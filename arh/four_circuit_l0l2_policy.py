"""FIR terminal search from the frozen implementation."""

from collections import Counter
from dataclasses import replace
from functools import lru_cache
from math import fsum, isfinite
from time import process_time

import numpy as np

from algorithm.adaptive_rtl_search import AdaptiveRtlSearch
from algorithm.cross_circuit_search import CrossCircuitSearch
from algorithm.source_frontier import _front_gain

BASE_BLOCK = 'development-0'


QUOTE_CYCLE_UNIT = 1_000_000  # Scale for quoted FTE inside the acquisition; never used to meter work.


class TerminalSearch(AdaptiveRtlSearch):
    """Area/recovery coverage and bounded refinement on native terminal evidence."""

    def __init__(self, spec, *, method, root_metadata, settings, max_fte, proposal_pool, shortlist,
                 root_evidence_policy=None, fast_proposals=False):
        if fast_proposals:
            spec = replace(spec, geometry=lru_cache(maxsize=8192)(spec.geometry))
        super().__init__(spec, root_metadata=root_metadata, **settings, reference_cycles=QUOTE_CYCLE_UNIT,
                         max_cycles=int(max_fte * QUOTE_CYCLE_UNIT), proposal_pool=proposal_pool,
                         shortlist=shortlist, balanced_actions=True, calibrated_feedback=True,
                         allow_refinement=method != 'no_root_refinement', fast_proposals=fast_proposals)
        self.method = method
        self._point_acquisition = method == 'no_rbkg'
        self.refinements = 0
        self._coverage_at = None
        self.fit_cpu_s = 0.
        self.kg_frontier_cpu_s = 0.
        self._expanded_root_policy = root_evidence_policy == 'root_capacity_v2'
        self._risk_floors = None
        self.outside_span_floor = None
        self.kg_above_cap_offers = 0
        self.kg_above_cap_positive_hv = 0
        self._opening_physical_roots = (len({
            (self.roots[r].source, self._physical_cluster(r)) for r in self.blocks[BASE_BLOCK]
        }) if spec.physical_root_shrinkage else 0)
        if method == 'no_physical_domains':
            self.mode = replace(self.mode, physical_domains=False)
            self.axes = {source: tuple(range(len(self.options))) for source in self.axes}

    def _posteriors(self, layouts, offers, active, *, root_overlap=False, offer_rows=None):
        start = process_time()
        outcome = super()._posteriors(layouts, offers, active, root_overlap=root_overlap, offer_rows=offer_rows)
        if self._expanded_root_policy and not self._point_acquisition:
            location = outcome[0]
            support = {g: self._physical_support(self._labels.get(g, ())) for g in layouts}
            by_layout = np.array([self._probability_floor(support[g]) for g in layouts])
            actions = offers.items() if offer_rows is None else offer_rows
            by_offer = {}
            for genome, (block, _, roots, _) in actions:
                observed = support[genome]
                fresh = self._physical_support(roots)
                combined = {s: observed.get(s, frozenset()) | fresh.get(s, frozenset())
                            for s in observed.keys() | fresh.keys()}
                by_offer[location[genome], block] = self._probability_floor(combined)
            self._risk_floors = by_layout, by_offer
        self.fit_cpu_s += process_time() - start
        return outcome

    def _physical_support(self, roots):
        support = {}
        for key in roots:
            support.setdefault(self.roots[key].source, set()).add(self._physical_cluster(key))
        return support

    def _probability_floor(self, support):
        # Worst empirical case for the floor is all successes, not perfect certainty.
        return fsum(self.weights[s] * .5 / (len(groups) + 1.) for s, groups in support.items())

    def _frontier_gains(self, means, posts, items, incumbents, areas, samples):
        start = process_time()
        if self._risk_floors is None and self.outside_span_floor is None:
            outcome = super()._frontier_gains(means, posts, items, incumbents, areas, samples)
        else:
            floors = None
            if self._risk_floors is not None:
                by_layout, by_offer = self._risk_floors
                floors = by_layout, np.array([by_offer[index, block] for index, block, _ in items])
            origin = self.circuit.area_origin
            cap = self.circuit.area_caps[0] - origin
            outcome = _front_gain(means, posts, self.weights, items, incumbents, areas,
                                  self.circuit.reference - origin, samples, cap=cap,
                                  risk_floors=floors, outside_span_floor=self.outside_span_floor)
            above = np.array([areas[index] > cap for index, _, _ in items])
            self.kg_above_cap_offers += int(np.count_nonzero(above))
            self.kg_above_cap_positive_hv += int(np.count_nonzero(above & (outcome[:, 0] > 1e-12)))
        self.kg_frontier_cpu_s += process_time() - start
        return outcome

    def _cell(self, genome, area):
        return self._band(area), tuple(genome[i] for i in self.recovery)

    def _reserve_coverage_candidates(self, proposals, scores, selected):
        """Keep rare joint cells visible to the scheduled coverage action."""
        measured = Counter(self._cell(g, self._areas[g]) for g in self._labels if self._labels[g])
        representatives = {}
        for genome, area in proposals.items():
            cell = self._cell(genome, area)
            priority = (-scores[genome][1], scores[genome][0], self._seed(self._round, 'cell', genome))
            if cell not in representatives or priority < representatives[cell][0]:
                representatives[cell] = priority, genome
        least_sampled = sorted(representatives, key=lambda cell: (measured[cell], representatives[cell][0]))
        reserve = [representatives[cell][1] for cell in least_sampled[:self.shortlist_size // 4]]
        order = tuple(dict.fromkeys((*reserve, *selected)))[:self.shortlist_size]
        return {genome: proposals[genome] for genome in order}

    def _needed_roots(self, genome, block):
        if not self._labels.get(genome) and block != BASE_BLOCK:
            return ()
        return super()._needed_roots(genome, block)

    def _choose(self, layouts, offers, capacity, limit, guided, *, joint_native=None):
        if joint_native is None or capacity != 1:
            raise ValueError('Terminal search ranks one native block at a time')
        measured = {g for g, labels in self._labels.items() if labels}
        candidates = {
            key: offer for key, offer in offers.items()
            if key[0] not in measured or (
                self.allow_refinement and 4 * (self.refinements + 1) <= self._issued + 1
                and self._refine_supported(key[0], measured)
            )
        }
        native = {key: joint_native[key] for key in candidates}
        if not candidates:
            return []
        if self._opening_physical_roots:
            confirm = self._confirmation_offer(layouts, native, measured)
            if confirm is not None:
                return [(confirm, 'optimistic_front_confirmation')]
        unseen = [key for key in candidates if key[0] not in measured]
        if unseen and len(measured) % 8 == 3 and self._coverage_at != len(measured):
            counts = Counter(self._cell(g, layouts[g]) for g in measured if g in layouts)
            winner = min(unseen, key=lambda key: (
                counts[self._cell(key[0], layouts[key[0]])],
                -min((sum(a != b for a, b in zip(key[0], g, strict=True)) for g in measured), default=0),
                self._seed(self._round, 'native-cell', key),
            ))
            self._coverage_at = len(measured)
            return [(winner, 'area_recovery_coverage')]
        if self._point_acquisition:
            return [self._point_policy(layouts, candidates, measured)]
        return super()._choose(layouts, candidates, capacity, limit, guided, joint_native=native)

    def _confirmation_offer(self, layouts, offers, measured):
        available = [key for key in offers if key[0] in measured]
        if not available:
            return None
        deployed = sorted(
            (row['area_ge'], row['empirical_risk'], genome, row)
            for genome in measured if (row := self._deployment(genome)) is not None
        )
        winner = next((row for _, risk, _, row in deployed if risk == 0.), None)
        if winner is None:
            return None
        genome = tuple(winner['genome'])
        support = sum(winner['roots_by_source'].values())
        if not self._expanded_root_policy and support >= 2 * self._opening_physical_roots:
            return None
        if not any(area > winner['area_ge'] for area in layouts.values()):
            return None
        prior = {(self.roots[r].source, self._physical_cluster(r)) for r in self._labels[genome]}
        choices = []
        for key in available:
            if key[0] != genome:
                continue
            _, _, roots, quote = offers[key]
            new = {(self.roots[r].source, self._physical_cluster(r)) for r in roots} - prior
            if new:
                choices.append((-len(new) / quote.median_cpu, key))
        return min(choices)[1] if choices else None

    def _refine_supported(self, genome, measured):
        row = self._deployment(genome)
        if row is None:
            return True
        return not any(
            other is not None and other['area_ge'] <= row['area_ge']
            and other['risk_interval'][1] < row['risk_interval'][0]
            for g in measured if g != genome
            for other in (self._deployment(g),)
        )

    def _point_policy(self, layouts, offers, measured):
        """Same terminal model and offers, with point improvement in place of KG."""
        locations, means, _ = self._posteriors(
            layouts, offers, sorted(measured), root_overlap=True,
            offer_rows=[(key[0], offer) for key, offer in offers.items()],
        )
        rows = [row for genome in measured if (row := self._deployment(genome)) is not None]
        cap = self.circuit.area_caps[0]
        span = self.circuit.reference - self.circuit.area_origin
        development = hasattr(self, 'arm')
        challenger = getattr(self, 'challenger', False)
        cap_risk = min((row['risk_interval'][1] if challenger else row['estimated_risk']
                        for row in rows if row['area_ge'] <= cap), default=1.)
        eligible = {key: self._deployment(key[0]) for key in offers}

        def value(key):
            genome, _ = key
            area = layouts[genome]
            pred = min(1., max(0., float(means[locations[genome]])))
            lower = min((r['estimated_risk'] for r in rows if r['area_ge'] <= area), default=1.)
            current = eligible[key]
            if current is not None:
                # Confidence width is a deterministic precision proxy, not a look-ahead fantasy.
                lo, hi = current['risk_interval']
                n = sum(current['roots_by_source'].values())
                precision = (hi - lo) * (1. - (n / (n + len(offers[key][2]))) ** .5)
                improvement = precision * (self.circuit.reference - area) / span
            else:
                improvement = max(0., lower - pred) * (self.circuit.reference - area) / span
                if area <= cap:
                    cap_weight = 1. if development else (self.circuit.reference - cap) / span
                    improvement += max(0., cap_risk - pred) * cap_weight
            cost = offers[key][3].median_cpu if self.mode.cost_aware else 1.
            if development and self._active_cost is not None and self.mode.cost_aware:
                cpu_scale, cycle_scale = self._cost_scales
                cost = max(cost / cpu_scale, self._active_cost(genome, offers[key][2]).cycles / cycle_scale)
            return (-improvement / cost,
                    self._seed(self._round, 'point-terminal', key))

        reason = 'cost_normalized_point_improvement' if self.mode.cost_aware else 'unpriced_point_improvement'
        return min(offers, key=value), reason
