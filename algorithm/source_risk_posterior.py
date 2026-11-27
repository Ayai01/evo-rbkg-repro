"""Gaussian source risks with a common-root term for matched fault injections."""

from math import sqrt

import numpy as np
from scipy.linalg import solve_triangular
from scipy.spatial.distance import cdist

from .root_block_knowledge_gradient import (
    GLOBAL_KERNEL_WEIGHT, LAYOUT_ROOT_STD, PRIOR_SIGMA, SHARED_ROOT_STD,
)


def components(genomes, source_axes, *, project):
    """Cache Hamming covariance matrices on the distinct physical projections."""
    all_axes = tuple(range(genomes.shape[1]))
    cache = {}

    def kernel(axes):
        if axes not in cache:
            values = np.ascontiguousarray(genomes[:, axes], dtype=float)
            cache[axes] = np.exp(cdist(values, values, 'hamming')*(-sqrt(len(axes)))) if axes else 1.
        return cache[axes]

    full = kernel(all_axes)
    for source, axes in source_axes.items():
        local = kernel(axes) if project else full
        yield source, PRIOR_SIGMA**2*((1.-GLOBAL_KERNEL_WEIGHT)*local+GLOBAL_KERNEL_WEIGHT*full)


def variance_removed(covariance, outcome_covariance, *, average_weights=None):
    """Reduction in target variance from a vector or its weighted scalar."""
    cov = np.asarray(covariance, dtype=float)
    observation = np.asarray(outcome_covariance, dtype=float)
    if average_weights is None:
        whitened = np.linalg.solve(np.linalg.cholesky(observation), cov)
        return float(whitened@whitened)
    weights = np.asarray(average_weights, dtype=float)
    return float((weights@cov)**2/(weights@observation@weights))


class SourcePosterior:
    def __init__(self, kernel, mean, measured, pending, available, counts, *, paired, noise_scale=1., root_sets=None,
                 predict_outcomes=False, physical_residual=None):
        """Rows use integer layout indices and immutable common-root block keys."""
        events = [*measured, *pending, *available]
        indices = [index for index, _ in events]
        shared = SHARED_ROOT_STD**2 if paired else 0.
        # Keep each action's marginal noise even when the cross-layout term is disabled.
        residual = LAYOUT_ROOT_STD**2+SHARED_ROOT_STD**2-shared
        n = np.asarray([counts[block] for _, block in events])
        blocks = [block for _, block in events]
        samples = kernel[np.ix_(indices, indices)]
        if root_sets is None or not paired:
            samples += np.equal.outer(blocks, blocks)*(shared*noise_scale/n)
        else:
            # Covariance of two source means is proportional to the size of
            # their root intersection, including when one is a multi-block mean.
            names = list(dict.fromkeys(blocks))
            groups = [set(root_sets[key]) for key in names]
            overlap = np.array([[len(a & b) for b in groups] for a in groups], dtype=float)
            lookup = {key: i for i, key in enumerate(names)}
            indices_by_block = [lookup[key] for key in blocks]
            if blocks:
                samples += shared*noise_scale*overlap[np.ix_(indices_by_block, indices_by_block)]/np.outer(n, n)
        samples.flat[::len(events)+1] += residual*noise_scale/n
        latent = kernel[:, indices]
        if physical_residual is not None:
            target_factors, outcome_factors = physical_residual
            samples += outcome_factors @ outcome_factors.T
            latent += target_factors @ outcome_factors.T
        n_seen, n_condition = len(measured), len(measured)+len(pending)
        self.mean = mean.copy()
        self.outcome_mean = mean[indices[n_condition:]].copy() if predict_outcomes else None
        self.cross = latent[:, n_condition:].copy()
        self.outcome = samples[n_condition:, n_condition:].copy()
        if n_condition:
            factor = np.linalg.cholesky(samples[:n_condition, :n_condition])
            # Preserve the separate matrix and vector solves used for strict acquisition ranking.
            left = solve_triangular(factor, latent[:, :n_condition].T, lower=True, check_finite=False)
            right = solve_triangular(factor, samples[:n_condition, n_condition:], lower=True, check_finite=False)
            self.cross -= left.T@right
            self.outcome -= right.T@right
            # The pending entries condition covariance. They carry no observed value.
            if n_seen:
                innovation = np.asarray(list(measured.values()))-mean[indices[:n_seen]]
                correction = solve_triangular(factor[:n_seen, :n_seen], innovation, lower=True, check_finite=False)
                self.mean += left[:n_seen].T@correction
                if predict_outcomes:
                    self.outcome_mean += right[:n_seen].T@correction
        self.index = {key: column for column, key in enumerate(available)}

    def standard_gain(self, columns):
        return self.cross[:, columns]/np.sqrt(self.outcome[columns, columns])

    def condition(self, key):
        j = self.index[key]
        covariance = self.outcome[:, j]/sqrt(self.outcome[j, j])
        self.cross -= np.multiply.outer(self.standard_gain(j), covariance)
        self.outcome -= np.multiply.outer(covariance, covariance)
