"""Allocate native source subsets to improve an area-constrained risk frontier."""

from collections import defaultdict
from math import fsum, isfinite

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .root_block_knowledge_gradient import PRIOR_RISK
from .source_risk_posterior import SourcePosterior, components


MODES = ('physical', 'global', 'unpaired', 'pooled')
CHUNK = 128


def _fit(genomes, sources, weights, counts, seen, pending, alternatives, mode, priors):
    if mode == 'pooled':
        return _pooled(genomes, sources, weights, counts, seen, pending, alternatives, priors)
    observed = defaultdict(dict)
    waiting = defaultdict(list)
    future = defaultdict(dict)
    for (layout, block, source), value in seen.items():
        observed[source][layout, block] = value
    for layout, block, source in pending:
        waiting[source].append((layout, block))
    for layout, block, subset in alternatives:
        for source in subset:
            future[source][layout, block] = None
    posts = {}
    for source, kernel in components(genomes, sources, project=mode != 'global'):
        source_counts = {block: n for (block, label), n in counts.items() if label == source}
        baseline = np.full(len(genomes), priors.get(source, PRIOR_RISK))
        posts[source] = SourcePosterior(kernel, baseline, observed[source], waiting[source], list(future[source]),
                                        source_counts, paired=mode != 'unpaired')
    mean = sum(weights[source]*post.mean for source, post in posts.items())
    return mean, posts, weights


def _pooled(genomes, sources, weights, counts, seen, pending, alternatives, priors):
    # Same marginal prior as the physical vector model. This arm observes one
    # weighted sum per full root block, leaving the compression as the change.
    seen_cells = dict.fromkeys((design, block) for design, block, _ in seen)
    pending_cells = list(dict.fromkeys((design, block) for design, block, _ in pending))
    future = list(dict.fromkeys(action[:2] for action in alternatives))
    if (len(seen) != len(seen_cells)*len(sources) or len(pending) != len(pending_cells)*len(sources)
            or any(set(subset) != sources.keys() for _, _, subset in alternatives)):
        raise ValueError('The pooled comparator accepts complete source bundles only')
    measurements = {(design, block): fsum(weights[s]*seen[design, block, s] for s in sources)
                    for design, block in seen_cells}
    mass = fsum(w*w for w in weights.values())
    blocks = {block for _, block in [*seen_cells, *pending_cells, *future]}
    effective_n = {block: mass/fsum(weights[s]**2/counts[block, s] for s in sources) for block in blocks}
    kernel = sum(weights[source]**2*k for source, k in components(genomes, sources, project=True))
    baseline = np.full(len(genomes), fsum(weights[s]*priors.get(s, PRIOR_RISK) for s in sources))
    post = SourcePosterior(kernel, baseline, measurements, pending_cells, future, effective_n,
                           paired=True, noise_scale=mass)
    return post.mean, {None: post}, {None: 1.}


def _front_gain(mean, posts, weights, offers, active, areas, reference, samples, *, cap=None, risk_floors=None,
                outside_span_floor=None, outside_gain_terms=None):
    active = sorted(active, key=lambda i: (areas[i], i))
    layouts = np.asarray([layout for layout, _, _ in offers])
    draws = len(samples)
    envelope = np.zeros((draws, len(offers), len(active)+1))
    envelope[:, :, 0] = 1.
    states = envelope[:, :, 1:]
    proposed = np.zeros((draws, len(offers)))
    for column, (source, post) in enumerate(posts.items()):
        indices = [i for i, (_, _, subset) in enumerate(offers) if source is None or source in subset]
        if not indices:
            continue
        keys = [post.index[offers[i][:2]] for i in indices]
        variance = post.standard_gain(keys).T*weights[source]
        random = samples[:, column, None]
        selected = slice(None) if len(indices) == len(offers) else indices
        states[:, selected, :] += random[:, :, None]*variance[:, active]
        proposed[:, selected] += random*variance[np.arange(len(indices)), layouts[selected]]
    states += mean[active]
    proposed += mean[layouts]
    left = np.r_[0., areas[active]]
    right = np.r_[areas[active], reference]
    widths = (right-left)/reference
    incumbent_mean = np.clip(mean[active], 0., 1.)
    np.clip(states, 0., 1., out=states)
    np.clip(proposed, 0., 1., out=proposed)
    if risk_floors is not None:
        by_layout, by_offer = risk_floors
        incumbent_floor = by_layout[active]
        incumbent_mean = np.maximum(incumbent_mean, incumbent_floor)
        np.maximum(states, incumbent_floor, out=states)
        np.maximum(proposed, by_offer, out=proposed)
    best = np.minimum.accumulate(np.r_[1., incumbent_mean])
    base = (1.-best)@widths
    np.minimum.accumulate(envelope, axis=2, out=envelope)
    improvement = 1.-envelope
    value = np.einsum('qaj,j->a', improvement, widths)/draws
    extra_width = np.maximum(right-np.maximum(left, areas[layouts, None]), 0.)/reference
    np.subtract(envelope, proposed[:, :, None], out=improvement)
    np.maximum(improvement, 0., out=improvement)
    direct = np.einsum('qaj,aj->a', improvement, extra_width)/draws
    value += direct
    raw_gain = value - base
    if outside_span_floor is not None or outside_gain_terms is not None:
        if cap is None or not 0. <= cap < reference:
            raise ValueError('Span balancing requires a cap below the area reference')
        outside_direct = np.where(areas[layouts] > cap, direct, 0.)
        if outside_gain_terms is not None:
            outside_gain_terms[:, 0] = raw_gain
            outside_gain_terms[:, 1] = outside_direct
        if outside_span_floor is not None:
            span = max(1. - cap/reference, outside_span_floor)
            raw_gain += (1. / span - 1.) * outside_direct
    gain = np.maximum(raw_gain, 0.)
    if cap is None:
        return gain
    at_cap = np.searchsorted(areas[active], cap, side='right')
    offer_cap = np.where(areas[layouts] <= cap, proposed, 1.)
    expected_cap = np.minimum(envelope[:, :, at_cap], offer_cap).mean(axis=0)
    cap_gain = np.maximum(best[at_cap] - expected_cap, 0.)
    return np.column_stack((gain, cap_gain))


def _bundle(layout, block, subset):
    return {(layout, block, source) for source in subset}


def _source_model(areas, genomes, axes, weights, mode, priors):
    indices = {d: i for i, d in enumerate(areas)}
    code = np.asarray([genomes[d] for d in areas])
    if code.ndim != 2 or not code.shape[1] or not np.issubdtype(code.dtype, np.integer):
        raise ValueError('Every layout needs a complete categorical genome')
    code = np.column_stack([np.unique(col, return_inverse=True)[1] for col in code.T])
    names = {source: tuple(sorted(set(cols))) for source, cols in axes.items()}
    if not names or set(names) != set(weights) or mode not in MODES or any(
            any(type(i) is not int or not 0 <= i < code.shape[1] for i in cols) for cols in names.values()):
        raise ValueError('Declare physical axes and risk mass for each source')
    if any(not isfinite(w) or w <= 0. for w in weights.values()):
        raise ValueError('Source mass must be positive and finite')
    if set(priors)-names.keys() or any(not 0. <= value <= 1. for value in priors.values()):
        raise ValueError('Source priors must be registered probabilities')
    total = fsum(weights.values())
    return indices, code, names, {source: float(weights[source])/total for source in names}


def choose(areas, genomes, source_axes, source_weights, root_counts, quotes, observed, pending, capacity, seed,
           *, reference, method='physical', cpu_budget=None, source_priors=None):
    """Choose offered (layout, root block, source subset) actions.

    A scientific atom is (layout, block, source). Results are per-root mean
    rates on the registered block, with identical roots across common layouts.
    Quotes may overlap, for example a full bundle and a cheaper single source.
    Pending and agreed actions affect covariance and eligibility only.
    """
    if type(capacity) is not int or capacity < 0 or not isfinite(reference) or reference <= 0.:
        raise ValueError('Expected a nonnegative batch size and positive area reference')
    limit = float('inf') if cpu_budget is None else cpu_budget
    if cpu_budget is not None and (not isfinite(cpu_budget) or cpu_budget < 0):
        raise ValueError('Remaining CPU must be finite and nonnegative')
    if set(areas) != set(genomes) or any(not isfinite(x) or not 0. <= x <= reference for x in areas.values()):
        raise ValueError('All candidates need registered admissible areas and genomes')
    if any(type(n) is not int or n < 1 for n in root_counts.values()):
        raise ValueError('Every source root block needs a positive integer number of roots')
    priors = source_priors or {}
    location, code, axes, weights = _source_model(areas, genomes, source_axes, source_weights, method, priors)
    keys = tuple(areas)
    area = np.asarray(list(areas.values()))
    pending = tuple(pending)
    held = set(pending)
    if (len(held) != len(pending) or held & observed.keys() or
            any(not isfinite(v) or not 0. <= v <= 1. for v in observed.values())):
        raise ValueError('Settled and pending source observations must be unique and disjoint')

    def atom(key):
        layout, block, source = key
        if layout not in location or (block, source) not in root_counts or source not in axes:
            raise ValueError('Source observation falls outside the registered root and design space')
        return location[layout], block, source

    seen = {atom(key): value for key, value in observed.items()}
    waiting = [atom(key) for key in pending]
    locked = set(seen) | set(waiting)
    alternatives, fees, wave = {}, {}, set()
    source_index = {name: i for i, name in enumerate(axes)}
    for offer, cost in quotes.items():
        layout, block, subset = offer
        if not subset or len(set(subset)) != len(subset):
            raise ValueError('Each offer needs a distinct set of sources')
        bundle = {atom(key) for key in _bundle(layout, block, subset)}
        expected, reserve = float(cost['median_cpu']), float(cost['reserve_cpu'])
        if not 0. < expected <= reserve or not isfinite(reserve):
            raise ValueError('Purchase quotes require positive CPU and sufficient reserve')
        target = location[layout], block, tuple(sorted(subset, key=source_index.__getitem__))
        if target in wave:
            raise ValueError('Equivalent source subsets need one combined offer')
        wave.add(target)
        if not bundle & locked and reserve <= limit:
            alternatives[target] = bundle
            fees[target] = expected, reserve

    mean, posts, model_weights = _fit(code, axes, weights, root_counts, seen, waiting, alternatives, method, priors)
    predictions = {d: float(np.clip(mu, 0., 1.)) for d, mu in zip(keys, mean, strict=True)}
    active = {layout for layout, _, _ in locked}
    rng = np.random.default_rng(seed)
    # Common draws across offers and methods, with paired antithetic values.
    samples = ndtri(np.clip(qmc.Sobol(len(posts), rng=rng).random_base2(3), 1.e-10, 1.-1.e-10))
    samples = np.vstack([samples, -samples])
    available = list(alternatives)
    ties = dict(zip(available, rng.random(len(available)), strict=True))
    selected, charges, actions = [], [], []
    for _ in range(capacity):
        if not available:
            break
        gains = np.concatenate([_front_gain(mean, posts, model_weights, available[i:i+CHUNK], active, area, reference, samples)
                                for i in range(0, len(available), CHUNK)])
        i = max(range(len(available)), key=lambda j: (gains[j]/fees[available[j]][0], ties[available[j]]))
        target = available[i]
        layout, block, subset = target
        expected, reserve = fees[target]
        selected.append((keys[layout], block, subset))
        charges.append(reserve)
        actions.append(dict(layout=keys[layout], block=block, sources=subset, expected_hv_gain=float(gains[i]),
                            median_cpu=expected, reserve_cpu=reserve))
        active.add(layout)
        available = [next_ for next_ in available if not alternatives[next_] & alternatives[target]
                     and fsum((*charges, fees[next_][1])) <= limit]
        for source in posts if method == 'pooled' else subset:
            if any(source is None or source in action[2] for action in available):
                posts[source].condition((layout, block))
    return dict(selected=selected, actions=actions, reserved_cpu=fsum(charges), predictions=predictions, method=method)
