"""Source-specific residuals from matched native comparisons at known loci."""

from collections import Counter, defaultdict
from math import sqrt

import numpy as np

from .root_block_knowledge_gradient import PRIOR_SIGMA


def _identity(value):
    if isinstance(value, (list, tuple)):
        return tuple(map(_identity, value))
    if type(value) in (int, str):
        return value
    raise ValueError('A physical locus needs an integer, string or sequence of coordinates')


class PhysicalLocusResidual:
    """An opt-in covariance term on exact physical loci in paid source pairs.

    A design is represented by the intervention state at the struck gene,
    while roots retain their finer physical identities when one gene controls
    several registers. Evidence is counted by same-root pairs under the same
    recovery setting. No outcome is consumed twice to fit a separate prior.
    """

    def __init__(self, root_metadata, *, recovery_axes=()):
        self.recovery_axes = tuple(recovery_axes)
        self.roots = {}
        self.by_source = defaultdict(list)
        for root, row in root_metadata.items():
            source = row['source']
            loci = tuple(row.get('loci', ()))
            if any(type(axis) is not int or axis < 0 or axis in self.recovery_axes for axis in loci):
                raise ValueError('Physical loci must name registered non-recovery axes')
            loci = tuple(sorted(set(loci)))
            position = _identity(row.get('physical_locus', loci)) if loci else ()
            self.roots[root] = source, row.get('cluster') or root, loci, position
            self.by_source[source].append(root)

    def _paired_support(self, source, paid):
        comparisons = defaultdict(set)
        for genome, labels in paid.items():
            recovery = tuple(genome[axis] for axis in self.recovery_axes)
            for root in labels:
                row = self.roots.get(root)
                if row is None or row[0] != source:
                    continue
                _, _, loci, position = row
                for axis in loci:
                    comparisons[root, position, axis, recovery].add(genome[axis])
        support = defaultdict(set)
        independent_source_roots = set()
        for (root, position, axis, recovery), alleles in comparisons.items():
            if len(alleles) > 1:
                cluster = self.roots[root][1]
                support[position, axis, recovery].add(cluster)
                independent_source_roots.add(cluster)
        return support, len(independent_source_roots)

    def _exposures(self, roots, aggregate_by_cluster):
        groups = defaultdict(list)
        for root in roots:
            _, cluster, loci, position = self.roots[root]
            groups[cluster if aggregate_by_cluster else root].append((loci, position))
        weights = Counter()
        for group in groups.values():
            weight = 1. / (len(groups) * len(group))
            for loci, position in group:
                for axis in loci:
                    weights[position, axis] += weight / len(loci)
        return weights

    def factors(self, source, layouts, options, paid, events, block_roots, *, aggregate_by_cluster):
        """Return low-rank factors for the target risk and the individual outcomes."""
        roots = self.by_source.get(source, ())
        if not roots or not events or any(root not in self.roots for block in block_roots.values() for root in block):
            return None
        support, source_count = self._paired_support(source, paid)
        if source_count < 2:
            return None
        target_weights = self._exposures(roots, aggregate_by_cluster)
        by_block = {block: self._exposures(members, aggregate_by_cluster)
                    for block, members in block_roots.items()}
        layout_rows = tuple(layouts)
        recovery = [tuple(genome[k] for k in self.recovery_axes) for genome in layout_rows]
        event_layout = [i for i, _ in events]
        target_features, event_features = [], []
        for (position, axis, mode), clusters in support.items():
            # A source with several paired roots can begin specializing a
            # sparse locus. Repeating layouts on one root never adds roots.
            n = len(clusters)
            amplitude = PRIOR_SIGMA * sqrt(source_count / (source_count + 4.) * n / (n + 4.))
            alleles = {value: i for i, value in enumerate(options[axis])}
            if len(alleles) < 2:
                continue
            code = [alleles[genome[axis]] for genome in layout_rows]
            contrast = np.eye(len(alleles))[code] - 1. / len(alleles)
            scale = amplitude * sqrt(len(alleles) / (len(alleles) - 1))
            contrast *= np.asarray([scale if home == mode else 0. for home in recovery])[:, None]
            component = position, axis
            weights = np.asarray([by_block[block].get(component, 0.) for _, block in events])
            if not np.any(weights):
                continue
            target_features.append(contrast * target_weights.get(component, 0.))
            event_features.append(contrast[event_layout] * weights[:, None])
        if not event_features:
            return None
        return np.column_stack(target_features), np.column_stack(event_features)
