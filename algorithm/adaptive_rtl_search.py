"""Bounded open proposals and separate short/native feedback for a root-block route."""

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import combinations, islice, product
from math import fsum, isfinite, sqrt

import numpy as np
from scipy.stats import spearmanr
from time import process_time

from .cross_circuit_search import Ablation, CrossCircuitSearch
from .paired_domain_transfer import PairedDomainTransfer
from .rtl_work_ledger import RtlWorkLedger, SearchAction
from .source_frontier import _front_gain
from .source_risk_posterior import components
from .root_block_knowledge_gradient import LAYOUT_ROOT_STD, PRIOR_RISK, SHARED_ROOT_STD
from .structural_search import propose_layouts


@dataclass(frozen=True)
class ForecastCost:
    cycles: int
    cpu_s: float

    def __post_init__(self):
        if type(self.cycles) is not int or self.cycles < 0 or not isfinite(self.cpu_s) or self.cpu_s < 0:
            raise ValueError('Action costs require nonnegative work and CPU')


def pareto_short_order(points, area, risk):
    """Order predicted Pareto layers before filling remaining promotion slots."""
    remaining = set(points)
    # Equal objective vectors share a layer; neither dominates the other.
    while remaining:
        best, front = float('inf'), []
        boundary = None
        for g in sorted(remaining, key=lambda x: (area[x], risk[x], x)):
            point = area[g], risk[g]
            if risk[g] < best or point == boundary:
                front.append(g)
                best = risk[g]
                boundary = point
        yield from front
        remaining.difference_update(front)


class AdaptiveRtlSearch(PairedDomainTransfer):
    """Rank a large physical pool before constructing the small RBKG posterior."""

    def __init__(self, circuit, *, root_metadata, seed, cpu_budget, reference_cycles,
                 max_cycles, proposal_pool=1024, shortlist=96, use_ridge=True, balanced_actions=False,
                 calibrated_feedback=False, pareto_short=False, allow_refinement=True, short_advice=False,
                 fast_proposals=False):
        if (circuit.candidates or not 4 <= shortlist <= 128 or proposal_pool < shortlist
                or type(max_cycles) is not int or max_cycles <= 0):
            raise ValueError('Use an open geometry, bounded shortlist and a positive cycle budget')
        super().__init__(circuit, root_metadata=root_metadata, seed=seed, cpu_budget=cpu_budget,
                         adaptive_blocks=False, ablation=Ablation(policy='joint_completion_kg',
                             coverage_trigger='unresolved' if balanced_actions else 'periodic'))
        self.balanced_actions = balanced_actions
        self.fast_proposals = fast_proposals
        self.calibrated_feedback = calibrated_feedback
        self.pareto_short = pareto_short
        self.allow_refinement = allow_refinement
        self.short_advice = short_advice
        self.short_advice_switches = 0
        self._short_calibration = None
        self._short_rank_cache = {}
        self._source_weight_cache = {}
        self.posterior_feedback = {}
        self._active_cost = None
        self._refinement_allowed = None
        self._cap_neighborhood = None
        self._cap_mutations = None
        self.ledger = RtlWorkLedger(reference_cycles)
        self.max_cycles = max_cycles
        self._cost_scales = cpu_budget, max_cycles
        self.pool_size, self.shortlist_size = proposal_pool, shortlist
        self.use_ridge = use_ridge
        self.explored = set()
        self.screen_log = []
        self.feedback = []
        self._shorts = {}
        self._short_signals = {}
        self._short_attempts = set()
        self._paired_signals = {}
        self._selected = {}
        self._prepared = False
        self._serial = 0
        self._native_quote = circuit.quote

    def _id(self, action):
        self._serial += 1
        return f'{action.value}/{self._serial}'

    def affordable(self, cost):
        bill = self.ledger.summary()
        return (bill['executed_cycles'] + cost.cycles <= self.max_cycles
                and bill['search_CPU_s'] + cost.cpu_s <= self.cpu_budget)

    def _band(self, area):
        return bisect_right(self.circuit.area_edges, area)

    def _matched_guidance(self, parents):
        # All guided candidates already compete for the bounded common shortlist.
        return {}

    def _standard(self, parents):
        if not self._prepared:
            raise ValueError('Refresh the dynamic L0 shortlist before the next root-block ask')
        return dict(self._selected)

    def _reserve_coverage_candidates(self, proposals, scores, selected):
        return selected

    def _local_open_moves(self, parents):
        # Cover every one-axis neighbor independent of incident-model confidence.
        # Swapping choices at comparable loci traverses area-neutral placements.
        options = self.options
        offered = {}
        for parent in parents:
            for axis, alleles in enumerate(options):
                for allele in alleles:
                    if allele != parent[axis]:
                        genome = (*parent[:axis], allele, *parent[axis+1:])
                        row = self._geometry(genome)
                        if row is not None and row[0] not in self._areas:
                            offered[row[0]] = row[1]
            for left, right in combinations(range(len(options)), 2):
                if options[left] != options[right] or parent[left] == parent[right]:
                    continue
                child = list(parent)
                child[left], child[right] = child[right], child[left]
                row = self._geometry(tuple(child))
                if row is not None and row[0] not in self._areas and row[1] <= parents[parent]:
                    offered[row[0]] = row[1]
        return offered

    def _capped_structural_moves(self):
        if not self.circuit.area_caps:
            return {}
        cap = self.circuit.area_caps[0]
        # Opening layouts and geometry are fixed. Occupancy changes after each native action.
        if self._cap_neighborhood is None:
            def mutations():
                for parent in dict.fromkeys(genome for genome, _ in self._opening):
                    recovery_options = [self.options[axis] for axis in self.recovery]
                    axes = [axis for axis in range(len(parent)) if axis not in self.recovery]
                    alternatives = {axis: tuple(a for a in self.options[axis] if a != parent[axis]) for axis in axes}
                    for distance in range(1, min(3, len(axes))+1):
                        for group in combinations(axes, distance):
                            for alleles in product(*(alternatives[axis] for axis in group), *recovery_options):
                                child = list(parent)
                                for axis, allele in zip((*group, *self.recovery), alleles, strict=True):
                                    child[axis] = allele
                                yield tuple(child)

            self._cap_neighborhood = {}
            self._cap_mutations = islice(mutations(), 12*self.pool_size)

        def available():
            yield from ((genome, area) for genome, area in self._cap_neighborhood.items()
                        if genome not in self._areas)
            # Extend the same ordered prefix only when the existing neighborhood
            # cannot refill the quota after newly occupied layouts are filtered out.
            for child in self._cap_mutations:
                row = self._geometry(child)
                if row is not None and row[1] <= cap and row[0] not in self._cap_neighborhood:
                    self._cap_neighborhood[row[0]] = row[1]
                    if row[0] not in self._areas:
                        yield row

        return dict(islice(available(), 4*self.pool_size))

    def _prepare_ridge_prediction(self, coefficients):
        """Sum fixed source and phase weights once per allele and recovery policy."""
        intercept, terms = defaultdict(float), defaultdict(float)
        for policy in product(*(self.options[axis] for axis in self.recovery)):
            for (source, axes, phase), weight in self.profile_weights.items():
                intercept[policy] += weight * coefficients.get(('policy', source, policy), 0.)
                if phase is not None:
                    intercept[policy] += weight * coefficients.get(('phase-policy', source, phase, policy), 0.)
                scale = weight / sqrt(max(1, len(axes)))
                for axis in axes:
                    for allele in self.options[axis]:
                        terms[policy, axis, allele] += scale * (
                            coefficients.get(('protection', source, axis, allele), 0.)
                            + coefficients.get(('interaction', source, axis, allele, policy), 0.))

        def predict(genome):
            policy = tuple(genome[axis] for axis in self.recovery)
            return intercept[policy] + fsum(terms.get((policy, axis, allele), 0.)
                                            for axis, allele in enumerate(genome))
        return predict

    def _source_kernel_components(self, codes, axes, layouts):
        if not self.calibrated_feedback or not self.mode.physical_domains:
            yield from super()._source_kernel_components(codes, axes, layouts)
            return
        whole = dict(components(codes, axes, project=False))
        for source, local in components(codes, axes, project=True):
            global_kernel = whole[source]
            observations = [(i, [loss for root, loss in self._labels.get(g, {}).items()
                                  if self.roots[root].source == source])
                            for i, g in enumerate(layouts)]
            observations = [(i, values) for i, values in observations if values]
            weight = .5
            if len(observations) >= 3:
                ordered = tuple(layouts)
                identity = tuple(sorted((ordered[i], tuple(row)) for i, row in observations))
                cached = self._source_weight_cache.get(source)
                if cached is not None and cached[0] == identity:
                    weight = cached[1]
                    self.posterior_feedback.setdefault(source, {})['physical_weight'] = weight
                    yield source, weight*local + (1.-weight)*global_kernel
                    continue
                ids, values = zip(*observations, strict=True)
                measured = np.asarray([fsum(row)/len(row) for row in values])
                noise = np.asarray([(LAYOUT_ROOT_STD**2+SHARED_ROOT_STD**2)/len(row) for row in values])

                def held_out(kernel):
                    covariance = kernel[np.ix_(ids, ids)] + np.diag(noise)
                    precision = np.linalg.solve(covariance, np.eye(len(ids)))
                    residual = precision @ (measured-PRIOR_RISK)
                    return measured-residual/np.diag(precision)

                local_fit, global_fit = held_out(local), held_out(global_kernel)
                direction = local_fit-global_fit
                # Shrink the leave-one-layout-out fit towards an equal kernel mixture.
                regularizer = 2.*np.mean(direction**2) + 1e-12
                weight = float(np.clip((direction @ (measured-global_fit) + .5*regularizer)
                                       / (direction @ direction + regularizer), 0., 1.))
                self._source_weight_cache[source] = identity, weight
            self.posterior_feedback.setdefault(source, {})['physical_weight'] = weight
            yield source, weight*local + (1.-weight)*global_kernel

    def _calibrated_windows(self):
        active = self._short_signals
        if self.short_advice:
            active = {(g, r): flag for g, cells in self._shorts.items() for r, flag in cells.items()}
        key = len(self._paired_signals), len(self._short_signals)
        if self._short_calibration is not None and self._short_calibration[0] == key:
            return self._short_calibration[1]
        counts, failures = Counter(), Counter()
        per_layout, per_layout_fail = Counter(), Counter()
        pairs = self._paired_signals
        if self.short_advice:
            cap = self.circuit.area_caps[0] if self.circuit.area_caps else self.circuit.reference
            matched_roots = {root for _, root in active}
            pairs = {(g, r): row for (g, r), row in pairs.items()
                     if r in matched_roots and self._areas[g] <= cap}
        state_roots = defaultdict(set)
        for (genome, root), (state, loss) in pairs.items():
            source = self.roots[root].source
            if self.short_advice:
                state_roots[source, state].add(self.cluster[root])
            for name in ((source, state), (source, None)):
                counts[name] += 1
                failures[name] += loss
                per_layout[genome, name] += 1
                per_layout_fail[genome, name] += loss

        def probability(name, genome=None):
            prior = (0.5 + failures[name[0], None] - per_layout_fail[genome, (name[0], None)]) / (
                1. + counts[name[0], None] - per_layout[genome, (name[0], None)])
            return (2.*prior + failures[name]-per_layout_fail[genome, name]) / (
                2. + counts[name]-per_layout[genome, name])

        gains, baseline = Counter(), Counter()
        size = Counter()
        for g, r in pairs:
            size[g, self.cluster[r]] += 1
        for (g, r), (state, loss) in pairs.items():
            source = self.roots[r].source
            raw = (loss-probability((source, None), g))**2
            short = (loss-probability((source, state), g))**2
            gain = (raw-short)/size[g, self.cluster[r]]
            gains[source] += gain
            baseline[source] += raw/size[g, self.cluster[r]]
        skill = {s: max(0., min(1., gains[s]/max(1e-12, baseline[s]))) for s in self.weights}
        shifts = defaultdict(list)
        for (g, r), flag in active.items():
            if r in self._labels.get(g, {}):
                continue
            source = self.roots[r].source
            if self.short_advice and len(state_roots[source, flag]) < 2:
                continue
            delta = probability((source, flag))-probability((source, None))
            shifts[source, g].append(delta)
        result = {k: skill[k[0]]*fsum(v)/len(v) for k, v in shifts.items()}
        self._short_calibration = key, result
        return result

    def _apply_short_source_evidence(self, source, layouts, posterior):
        if self.short_advice or not self.calibrated_feedback or not self._short_signals:
            return
        history = self._calibrated_windows()
        delta = np.asarray([history.get((source, g), 0.) for g in layouts])
        posterior.mean[:] = np.clip(posterior.mean+delta, 0., 1.)
        self.posterior_feedback.setdefault(source, {})['short_updated_layouts'] = int(np.count_nonzero(delta))

    def _cold_candidates(self, proposals, opener):
        genomes = sorted(proposals)
        design = np.asarray(genomes)
        origin = np.asarray(opener)
        exposures = Counter()
        if self.mode.physical_domains:
            for (_, axes, _), weight in self.profile_weights.items():
                for axis in axes:
                    exposures[axis] += weight/len(axes)
        prior = (design != origin) @ np.array([exposures[k] for k in range(len(opener))])
        distance = np.count_nonzero(design != origin, axis=1)
        bands = np.array([self._band(proposals[g]) for g in genomes])
        counts = np.zeros(len(self.circuit.area_edges) + 1, dtype=np.intp)
        remaining = np.ones(len(genomes), dtype=bool)
        for _ in range(min(len(genomes), 32, self.shortlist_size)):
            available = np.flatnonzero(remaining)
            occupancy = counts[bands[available]]
            eligible = available[occupancy == occupancy.min()]
            score = distance[eligible] / len(opener) + prior[eligible]
            tied = eligible[score == score.max()]
            # argmax retains the first (lexicographically smallest) tied genome.
            winner = tied[np.argmax(prior[tied])]
            remaining[winner] = False
            counts[bands[winner]] += 1
            distance = np.minimum(distance, np.count_nonzero(design != design[winner], axis=1))
            yield genomes[winner]

    def _frontier_gains(self, means, posts, items, incumbents, areas, samples):
        cap = self.circuit.area_caps[0] - self.circuit.area_origin if self.circuit.area_caps else None
        return _front_gain(means, posts, self.weights, items, incumbents, areas,
                           self.circuit.reference-self.circuit.area_origin, samples,
                           cap=cap if self.balanced_actions else None)

    def _score_offers(self, options, offers, owner, values, **kwargs):
        cost = self._active_cost
        if values.ndim == 2 and self.calibrated_feedback:
            cap = self.circuit.area_caps[0]
            span = self.circuit.reference - self.circuit.area_origin
            values = values[:, 0] + values[:, 1] * (self.circuit.reference-cap)/span
        elif values.ndim == 2:
            peaks = values.max(axis=0)
            # Comparable scales for the two different risk-frontier endpoints.
            normalized = np.divide(values, peaks, out=np.zeros_like(values), where=peaks > 1e-12)
            values = normalized.mean(axis=1)
        if not self.balanced_actions or cost is None or not self.mode.cost_aware:
            return super()._score_offers(options, offers, owner, values, **kwargs)
        cpu_scale, cycle_scale = self._cost_scales
        return {key: float(gain) / max(offers[key][3].median_cpu/cpu_scale,
                                     cost(owner[key], offers[key][2]).cycles/cycle_scale)
                for key, gain in zip(options, values, strict=True)}

    def _exploration_options(self, options, owner):
        if not self.balanced_actions:
            return options
        return [key for key in options if not self._labels.get(owner[key])]

    def _rank_knowledge_gradient(self, options, gains, tie, owner):
        native = super()._rank_knowledge_gradient(options, gains, tie, owner)
        if not self.short_advice or not self._shorts or owner[native] not in self._shorts:
            return native
        # Short evidence can break a near tie between measured cap candidates.
        # The native score, all other offers and the deployment posterior stay intact.
        candidates = [key for key in options if owner[key] in self._shorts
                      and gains[key] >= .90 * gains[native]]
        shifts = self._calibrated_windows()

        def correction(key):
            return fsum(weight * shifts.get((source, owner[key]), 0.) for source, weight in self.weights.items())

        best = min(candidates, key=lambda key: (correction(key), -gains[key], tie[key]))
        if correction(native) - correction(best) >= .01:
            self.short_advice_switches += 1
            return best
        return native

    def _choose(self, layouts, offers, capacity, limit, guided, *, joint_native=None):
        if self.balanced_actions and joint_native:
            anchored = {g: self._deployment(g) for g in self._labels if g in layouts}
            witnessed = [row for row in anchored.values() if row is not None]

            def valuable(key):
                row = anchored.get(key[0])
                if row is None:
                    return True  # Includes unfinished, partially evaluated layouts.
                if not self.allow_refinement:
                    return False
                if self._refinement_allowed is not None and not self._refinement_allowed(key[0], offers[key][2]):
                    return False
                return not any(other['area_ge'] <= row['area_ge']
                    and other['risk_interval'][1] < row['risk_interval'][0] for other in witnessed)

            offers = {key: offer for key, offer in offers.items() if valuable(key)}
            joint_native = {key: joint_native[key] for key in offers}
            if not offers:
                return []
        return super()._choose(layouts, offers, capacity, limit, guided, joint_native=joint_native)

    def screen(self):
        start = process_time()
        action = SearchAction.L0_SCREEN
        parents = self._parents()
        groups = tuple(dict.fromkeys(tuple(sorted(set(axes))) for axes in self.axes.values() if axes))
        proposals = propose_layouts(self.options, self.circuit.geometry, parents, self._areas,
                    count=self.pool_size, seed=self._seed(self._round, 'large-open-pool'),
                    area_edges=self.circuit.area_edges, groups=groups)
        local_moves = self._local_open_moves(parents) if self.balanced_actions else {}
        cap_moves = self._capped_structural_moves() if self.balanced_actions else {}
        proposals.update(local_moves)
        proposals.update(cap_moves)
        coefficients = self._coefficients() if self.use_ridge else {}
        profile_weights = self.profile_weights
        predict = self._prepare_ridge_prediction(coefficients) if coefficients and self.balanced_actions else None
        guided = {}
        if self.mode.directed_recovery:
            if self.use_ridge:
                guided = PairedDomainTransfer._matched_guidance(
                    self, parents, predict=predict if self.fast_proposals else None,
                )
            else:
                guided = CrossCircuitSearch._matched_guidance(self, parents)
        for genome in guided.keys() - self._areas.keys():
            proposals[genome] = self.circuit.geometry(genome)[1]
        if len(proposals) < self.pool_size:
            raise ValueError('The registered open geometry returned an undersized legal pool')
        if getattr(self, 'development_pool_size', None) is not None:
            supply = defaultdict(list)
            for genome, area in proposals.items():
                supply[self._band(area)].append(genome)
            queues = [iter(sorted(genomes, key=lambda genome: (
                genome not in guided, genome not in local_moves,
                self._seed(self._round, 'bounded-development-supply', genome))))
                for _, genomes in sorted(supply.items())]
            selected = {}
            while len(selected) < self.development_pool_size:
                for queue in queues:
                    genome = next(queue, None)
                    if genome is not None:
                        selected[genome] = proposals[genome]
                    if len(selected) == self.development_pool_size:
                        break
            proposals = selected

        bands = defaultdict(list)
        bases = tuple(parents)
        # Keep only one candidate-by-gene comparison live per parent.
        designs = np.asarray(tuple(proposals), dtype=object)
        distances = np.full(len(proposals), len(self.options), dtype=np.intp)
        for parent in bases:
            np.minimum(distances, np.count_nonzero(designs != np.asarray(parent, dtype=object), axis=1),
                       out=distances)
        scores = {}
        for (genome, area), distance in zip(proposals.items(), distances, strict=True):
            if predict is not None:
                ridge = predict(genome)
            else:
                ridge = fsum(weight * fsum(value * coefficients.get(term, 0.)
                             for term, value in self._features(genome, profile).items())
                             for profile, weight in profile_weights.items()) if coefficients else 0.
            scores[genome] = (ridge - guided.get(genome, 0.), int(distance), area)
            bands[self._band(area)].append(genome)

        # Interleave low estimated risk, near-cap and weak-support candidates per band.
        banks = []
        cap = self.circuit.area_caps[0] if self.circuit.area_caps else self.circuit.reference
        for band, members in sorted(bands.items()):
            banks.extend(iter(sorted(members, key=key)) for key in (
                lambda g: (*scores[g], g),
                lambda g: (abs(scores[g][2] - cap), scores[g][0], g),
                lambda g: (-scores[g][1], scores[g][0], g)))
        cold = (self.calibrated_feedback and self._opening and
                len({self.cluster[r] for r, rows in self._labels_by_root().items()
                     if len(set(rows.values())) > 1}) < 4)
        selected = {}
        if cold:
            selected.update((g, proposals[g]) for g in self._cold_candidates(proposals, self._opening[0][0]))
        elif self.balanced_actions and self.circuit.area_caps and self._opening:
            by_structure = defaultdict(list)
            opener = self._opening[0][0]
            for genome, area in proposals.items():
                if area <= cap:
                    policy = tuple(genome[axis] for axis in self.recovery)
                    edits = sum(gene != opener[axis] for axis, gene in enumerate(genome) if axis not in self.recovery)
                    by_structure[edits, policy].append(genome)
            # Reserve representatives across recovery settings and protection
            # distances. Otherwise tied cold scores fill the reserve from
            # a small set of high-area patterns.
            cap_banks = [iter(sorted(members, key=lambda g: (scores[g][0], -scores[g][1],
                                                            cap-scores[g][2],
                                                            self._seed(self._round, 'cap-shortlist', g))))
                         for _, members in sorted(by_structure.items())]
            while cap_banks and len(selected) < self.shortlist_size//4:
                active = []
                for bank in cap_banks:
                    choice = next(bank, None)
                    if choice is not None:
                        selected[choice] = proposals[choice]
                        active.append(bank)
                        if len(selected) == self.shortlist_size//4:
                            break
                cap_banks = active
        prioritized = () if cold else tuple(selected)
        while banks and len(selected) < self.shortlist_size:
            active = []
            for bank in banks:
                choice = next((genome for genome in bank if genome not in selected), None)
                if choice is not None:
                    selected[choice] = proposals[choice]
                    active.append(bank)
                    if len(selected) == self.shortlist_size:
                        break
            banks = active
        if prioritized:
            priority = set(prioritized)
            ordinary = iter(genome for genome in selected if genome not in priority)
            order = []
            for genome in prioritized:
                order.append(genome)
                order.extend(islice(ordinary, 3))
            order.extend(ordinary)
            selected = {genome: selected[genome] for genome in order}
        selected = self._reserve_coverage_candidates(proposals, scores, selected)
        self.explored.update(proposals)
        self._selected, self._shorts, self._prepared = selected, {}, True
        row = dict(pool_size=len(proposals), unique_seen=len(self.explored), shortlist_size=len(selected),
                   ridge_terms=len(coefficients), guided_in_shortlist=len(guided.keys() & selected.keys()),
                   maximum_dense_dimension=len(selected) + len(self._areas),
                   model_revision=len(self._paired_signals))
        if self.balanced_actions:
            row.update(cold_diversity=bool(cold), local_move_count=len(local_moves), capped_neighborhood_count=len(cap_moves),
                       capped_pool=sum(area <= cap for area in proposals.values()),
                       capped_shortlist=sum(area <= cap for area in selected.values()))
        self.screen_log.append(row)
        self.ledger.record(self._id(action), action, decision_cpu_s=process_time() - start)
        return tuple(selected)

    def short_candidates(self, block, size):
        cap = self.circuit.area_caps[0] if self.circuit.area_caps else self.circuit.reference
        return tuple(g for g, area in self._selected.items() if (not self.short_advice or area <= cap)
                     and (g, block) not in self._short_attempts and all(
            (g, r) not in self._short_signals and (g, r) not in self._paired_signals
            for r in self.blocks[block]))[:size]

    def held_out_short_quality(self, block):
        # Only completed short/terminal pairs on this exact root block count.
        ids = self.blocks[block]
        per_genome = defaultdict(dict)
        eligible = set(ids)
        for (genome, root), row in self._paired_signals.items():
            if root in eligible:
                per_genome[genome][root] = row
        cap = self.circuit.area_caps[0] if self.circuit.area_caps else self.circuit.reference
        rows = {g: cells for g, cells in per_genome.items() if cells.keys() == eligible
                and (not self.short_advice or self._areas[g] <= cap)}
        signature = tuple(sorted(rows))
        previous = self._short_rank_cache.get(block)
        if previous is not None and previous[0] == signature:
            return previous[1]
        counts, errors = Counter(), Counter()
        for cells in rows.values():
            for root, (flag, fail) in cells.items():
                source = self.roots[root].source
                for key in ((source, flag), (source, None)):
                    counts[key] += 1
                    errors[key] += fail
        predicted, observed, squared = [], [], [0., 0.]
        root_weights = Counter(self.roots[r].source for r in ids)
        total = fsum(self.weights[s] for s in root_weights)
        ordered_weights = {r: self.weights[self.roots[r].source] / root_weights[self.roots[r].source] / total
                           for r in ids}
        for cells in rows.values():
            own_counts, own_errors = Counter(), Counter()
            for root, (flag, fail) in cells.items():
                source = self.roots[root].source
                for key in ((source, flag), (source, None)):
                    own_counts[key] += 1
                    own_errors[key] += fail
            estimate, actual = 0., 0.
            for root, (flag, fail) in cells.items():
                source = self.roots[root].source
                pooled = source, None
                key = source, flag
                baseline = (0.5 + errors[pooled] - own_errors[pooled]) / (1. + counts[pooled] - own_counts[pooled])
                short = (2.*baseline + errors[key] - own_errors[key]) / (2. + counts[key] - own_counts[key])
                weight = ordered_weights[root]
                estimate += weight*short
                actual += weight*fail
                squared[0] += weight*(fail-baseline)**2
                squared[1] += weight*(fail-short)**2
            predicted.append(estimate)
            observed.append(actual)
        rho = None
        if len(set(predicted)) > 1 and len(set(observed)) > 1:
            rho = float(spearmanr(predicted, observed).statistic)
        result = dict(layouts=len(rows), spearman=rho,
                      brier_improvement=(squared[0]-squared[1])/max(1, len(rows)))
        self._short_rank_cache[block] = signature, result
        return result

    @staticmethod
    def short_saves_work(short_cost, anchor_cost, retain_fraction):
        discarded = 1 - retain_fraction
        return (short_cost.cycles < discarded * anchor_cost.cycles
                and short_cost.cpu_s < discarded * anchor_cost.cpu_s)

    def offer_short(self, block, *, short_cost, anchor_cost, retain_fraction=.5, coverage=True,
                    batch_size=None, reservation=None):
        """Offer one hot batch only when its projected savings cover its cost on both axes."""
        if not self._prepared or block not in self.blocks or not 0 < retain_fraction < 1:
            raise ValueError('Short offers need a current shortlist and a registered block')
        if (not coverage or not self._paired_signals or not self.affordable(reservation or short_cost)
                or not self.short_saves_work(short_cost, anchor_cost, retain_fraction)):
            return ()
        return (self.short_candidates(block, batch_size) if self.calibrated_feedback else
                tuple(self._selected)[:batch_size])

    def observe_short(self, action_id, block, signals, executions, *, keep, protected=(), decision_cpu_s=0.,
                      setup_cpu_s=0.):
        """Use pooled short risks to gate promotion; retain protected area/uncertainty candidates."""
        start = process_time()
        signals = {tuple(genome): dict(rows) for genome, rows in signals.items()}
        protected = tuple(map(tuple, protected))
        if (not self._prepared or self._shorts or block not in self.blocks or not set(signals) <= self._selected.keys()
                or not 1 <= keep <= len(signals) or len(set(protected)) > keep
                or any(g not in signals for g in protected)
                or any(set(values) != set(self.blocks[block]) for values in signals.values())):
            raise ValueError('Settle exactly one complete short batch on the registered shortlist and roots')
        self._shorts = signals
        self._short_signals.update({(g, root): value for g, rows in signals.items()
                                    for root, value in rows.items()})
        if self.short_advice:
            self.ledger.record(action_id, SearchAction.L1_SHORT_BATCH, executions, setup_cpu_s=setup_cpu_s,
                               decision_cpu_s=decision_cpu_s + process_time() - start)
            return tuple(signals)
        counts, failures = Counter(), Counter()
        for (_, root), (state, loss) in self._paired_signals.items():
            pair = self.roots[root].source, state
            counts[pair] += 1
            failures[pair] += loss

        def pooled(genome):
            by_source = defaultdict(list)
            for root, state in signals[genome].items():
                source = self.roots[root].source
                pair = source, state
                by_source[source].append((.5 + failures[pair]) / (1 + counts[pair]))
            return fsum(weight * (fsum(by_source[source]) / len(by_source[source]) if by_source[source] else .5)
                        for source, weight in self.weights.items())

        if self.calibrated_feedback:
            layouts = {g: self._areas[g] for g in self._labels}
            layouts.update({g: self._selected[g] for g in signals})
            location, means, _ = self._posteriors(layouts, {}, sorted(self._labels), root_overlap=True)
            if self.pareto_short:
                forecast = {g: float(means[location[g]]) for g in signals}
                ranked = tuple(pareto_short_order(signals, self._selected, forecast))
                cap = self.circuit.area_caps[0] if self.circuit.area_caps else self.circuit.reference
                if any(self._selected[g] <= cap for g in signals) and all(self._selected[g] > cap for g in ranked[:keep]):
                    protected = (min((g for g in signals if self._selected[g] <= cap),
                                     key=lambda g: (forecast[g], self._selected[g], g)),)
                else:
                    protected = ()
            else:
                ranked = sorted(signals, key=lambda g: (means[location[g]], pooled(g), self._selected[g], g))
        else:
            ranked = sorted(signals, key=lambda g: (pooled(g), self._selected[g], g))
        promoted = tuple(dict.fromkeys((*protected, *ranked)))[:keep]
        dropped = signals.keys() - set(promoted)
        self._selected = {genome: area for genome, area in self._selected.items() if genome not in dropped}
        self.ledger.record(action_id, SearchAction.L1_SHORT_BATCH, executions, setup_cpu_s=setup_cpu_s,
                           decision_cpu_s=decision_cpu_s + process_time() - start)
        return promoted

    def next_anchor(self, cost_for, *, refinement_allowed=None):
        """Select a new layout or a previously measured one with an additional root block."""
        start = process_time()
        action_id = self._id(SearchAction.L0_SCREEN)
        bill = self.ledger.summary()
        original = self.circuit
        room_cpu = self.cpu_budget - bill['search_CPU_s']
        room_cycles = self.max_cycles - bill['executed_cycles']
        if room_cpu <= 0 or room_cycles <= 0:
            return None

        def support(layout, roots):
            allowed = original.has_new_support
            return (allowed is None or allowed(layout, roots)) and cost_for(layout, roots).cycles <= room_cycles

        def price(layout, block, roots):
            quote = self._native_quote(layout, block, roots)
            reserved = max(cost_for(layout, roots).cpu_s, quote.reserve_cpu)
            if reserved > room_cpu:
                reserved = max(reserved, self.cpu_budget + 1)
            return replace(quote, reserve_cpu=reserved)

        self.circuit = replace(original, quote=price, has_new_support=support)
        self._active_cost = cost_for
        self._refinement_allowed = refinement_allowed
        try:
            chosen = self.ask(1)
        finally:
            self.circuit = original
            self._active_cost = None
            self._refinement_allowed = None
        self.ledger.record(action_id, SearchAction.L0_SCREEN, decision_cpu_s=process_time() - start)
        if not chosen:
            return None
        purchase, = chosen
        action = (SearchAction.L2_REFINE_ROOTS if self._labels.get(purchase.genome)
                  else SearchAction.L2_NEW_ANCHOR)
        return self._id(action), action, purchase

    def observe_anchor(self, ticket, outcomes, *, original, executions, setup_cpu_s=0., feasible=True,
                       signals=None, decision_cpu_s=0.):
        start = process_time()
        action_id, action, purchase = ticket
        key = purchase.genome
        if set(outcomes) != set(purchase.root_ids) or signals is not None and not set(signals) <= outcomes.keys():
            raise ValueError('Native feedback must settle the purchased roots and matching short signals')
        executions = tuple(executions)
        self.tell(purchase, outcomes, original_id=original,
                  paid_cpu=fsum(row.simulation_cpu_s for row in executions), feasible=feasible)
        before = len(self._paired_signals)
        for root, loss in outcomes.items():
            state = (signals or {}).get(root, self._short_signals.get((key, root)))
            if state is not None:
                self._paired_signals[key, root] = state, float(loss)
        self.feedback.append(dict(action=action.value, genome=key, roots=tuple(outcomes),
                                  paired_signals_added=len(self._paired_signals) - before))
        self._prepared = False
        self.ledger.record(action_id, action, executions, setup_cpu_s=setup_cpu_s,
                           decision_cpu_s=decision_cpu_s + process_time() - start)

    def admit_projection(self, donor, recipient, root, value, *, certificate_id, verify):
        """Import a proof only after the circuit adapter verifies its exact physical scope."""
        if not verify(donor, recipient, root, certificate_id):
            raise ValueError('The originating execution does not certify this target and root')
        start = process_time()
        accepted = self.seed_label_transfer(donor, recipient, root, value, certificate_id=certificate_id)
        action = SearchAction.L0_SCREEN
        self.ledger.record(self._id(action), action, decision_cpu_s=process_time() - start)
        return accepted
