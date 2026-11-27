"""Development-only terminal arms for testing the tight-cap deployment objective."""

from collections import defaultdict
from dataclasses import replace
from math import fsum, sqrt
from time import process_time

from algorithm.cross_circuit_search import CrossCircuitSearch, copy_deployment, deployment_frontier
from algorithm.rtl_work_ledger import SearchAction
from algorithm.structural_search import propose_layouts

from .four_circuit_l0l2_policy import TerminalSearch


ARMS = ('cap_kg', 'cap_point', 'cap_no_paired', 'cap_no_hierarchy', 'cap_no_acquisition', 'cap_no_targeted')
CLEAN_ARMS = ('cap_kg', 'cap_no_paired', 'cap_no_hierarchy', 'cap_no_acquisition')
CLEAN_PROFILE = 'rbkg_clean_v1'
SPAN_PROFILE = 'arh_clean_span_v1'
SPAN_ARMS = ('arh_clean', 'no_evolution', 'no_hierarchy', 'no_rbkg')
ADAPTIVE_CAP_ANCHORS = 'top_k_saturation_v1'


class CapTerminalSearch(TerminalSearch):
    """Give the deployment cap a share of native anchors and paired verification."""

    def __init__(self, *args, arm, challenger=False, point_profile=False, cap_anchor_policy=None,
                 clean=False, span=False, outside_span_floor=None, **kwargs):
        super().__init__(*args, **kwargs)
        if cap_anchor_policy not in (None, ADAPTIVE_CAP_ANCHORS):
            raise ValueError('Unknown cap anchor reservation policy')
        self.arm = arm
        self.clean = clean
        self.span = span
        self.challenger = challenger
        self.cap_anchor_policy = cap_anchor_policy
        if point_profile:
            self._point_acquisition = True
        self.development_pool_size = self.pool_size
        self._cap_share = .5
        self._top = 3
        if span or arm == 'cap_no_paired' or (arm == 'cap_no_hierarchy' and not clean):
            self.mode = replace(self.mode, directed_recovery=False)
            self.use_ridge = False
        if clean and arm in ('cap_no_hierarchy', 'no_hierarchy'):
            self.mode = replace(self.mode, physical_domains=False)
        if clean and arm in ('cap_no_acquisition', 'no_rbkg'):
            self.mode = replace(self.mode, cost_aware=False)
            self._point_acquisition = True
        if span and arm != 'no_rbkg':
            self.outside_span_floor = outside_span_floor

    def screen(self):
        if (self.clean or self.use_ridge) and self.arm != 'no_evolution':
            return super().screen()
        started = process_time()
        # The ablations have legal placement and the fixed opening geometry, and
        # use no measured parents, source groups, ridge coefficients or swaps.
        proposals = propose_layouts(self.options, self.circuit.geometry, (self._opening[0][0],), self._areas,
            count=self.pool_size, seed=self._seed(self._round, 'development-area-supply'),
            area_edges=self.circuit.area_edges)
        if len(proposals) != self.pool_size:
            raise ValueError('The geometry-only arm needs the same legal proposal budget')
        selected = dict(list(proposals.items())[:self.shortlist_size])
        self.explored.update(proposals)
        self._selected, self._shorts, self._prepared = selected, {}, True
        cap = self.circuit.area_caps[0]
        self.screen_log.append(dict(pool_size=len(proposals), unique_seen=len(self.explored),
            shortlist_size=len(selected), ridge_terms=0, guided_in_shortlist=0,
            maximum_dense_dimension=len(selected) + len(self._areas),
            capped_pool=sum(area <= cap for area in proposals.values()),
            capped_shortlist=sum(area <= cap for area in selected.values())))
        self.ledger.record(self._id(SearchAction.L0_SCREEN), SearchAction.L0_SCREEN,
                           decision_cpu_s=process_time() - started)
        return tuple(selected)

    def _cap_rows(self, measured, cap):
        return sorted((row for genome in measured if self._areas[genome] <= cap
                       and (row := self._deployment(genome)) is not None),
                      key=lambda row: (row['risk_interval'][1], row['estimated_risk'],
                                       -sum(row['roots_by_source'].values()), row['area_ge'], row['genome']))

    def _contenders(self, measured, cap):
        rows = self._cap_rows(measured, cap)
        if not self.challenger or len(rows) <= self._top:
            return rows[:self._top]
        leader = rows[0]
        eligible = [row for row in rows[2:] if row['risk_interval'][0] < leader['risk_interval'][1]]
        if not eligible:
            return rows[:self._top]
        # Select a challenger whose limited evidence still supports a lower
        # deployment risk. The conservative incumbent stays protected.
        challenger = min(eligible, key=lambda row: (row['risk_interval'][0], row['estimated_risk'],
            -sum(a != b for a, b in zip(row['genome'], leader['genome'], strict=True)),
            sum(row['roots_by_source'].values()), row['area_ge'], row['genome']))
        return [*rows[:2], challenger]

    def _targeted_offer(self, measured, offers):
        if not self.allow_refinement or 4 * (self.refinements + 1) > self._issued + 1:
            return None
        cap = self.circuit.area_caps[0]
        contenders = self._contenders(measured, cap)
        if len(contenders) < 2:
            return None
        leader = contenders[0]
        competitors = {tuple(row['genome']): row for row in contenders
                       if row['risk_interval'][0] <= leader['risk_interval'][1]
                       and row['risk_interval'][1] >= leader['risk_interval'][0]}
        if len(competitors) < 2:
            return None
        leader_genome = tuple(leader['genome'])
        challenger = tuple(contenders[-1]['genome'])
        registered = {g: self._physical_support(self._labels[g]) for g in competitors}
        choices = []
        for key, offer in offers.items():
            genome, _ = key
            if genome not in competitors:
                continue
            fresh = self._physical_support(offer[2])
            if not fresh:
                continue
            peers = (competitors.keys() - {genome} if genome == leader_genome else (leader_genome,))
            matched = max(fsum(self.weights[s] * len(roots & registered[peer].get(s, set()))
                               for s, roots in fresh.items()) for peer in peers)
            uncertainty = competitors[genome]['risk_interval'][1] - competitors[genome]['risk_interval'][0]
            support = sum(competitors[genome]['roots_by_source'].values())
            chase = self.challenger and genome == challenger and matched > 0 and (
                support < sum(leader['roots_by_source'].values()))
            priority = (not chase, -matched, -uncertainty, support,
                        self._seed(self._round, 'cap-matched-refinement', key))
            choices.append((priority, key))
        return min(choices)[1] if choices else None

    def _reserve_cap_anchor(self, measured, cap):
        if self.cap_anchor_policy == ADAPTIVE_CAP_ANCHORS:
            # Once Top-K deployment candidates exist, paired refinement addresses
            # their uncertainty. New anchors compete on both marginal gains.
            if len(self._cap_rows(measured, cap)) >= self._top:
                return False
        measured_in_cap = sum(self._areas[genome] <= cap for genome in measured)
        return measured_in_cap < self._cap_share * (len(measured) + 1)

    def _choose(self, layouts, offers, capacity, limit, guided, *, joint_native=None):
        measured = {g for g, labels in self._labels.items() if labels}
        if self.clean:
            return self._choose_clean(layouts, offers, capacity, limit, guided, joint_native, measured)
        verification = self._targeted_offer(measured, offers)
        if verification is not None:
            reason = 'cap_challenger_paired_refinement' if self.challenger else 'cap_top3_paired_refinement'
            return [(verification, reason)]
        new = {key: offer for key, offer in offers.items() if key[0] not in measured}
        if self.arm == 'cap_no_acquisition':
            order = {genome: i for i, genome in enumerate(self._selected)}
            if not new:
                return []
            winner = min(new, key=lambda key: (order.get(key[0], len(order)), key))
            return [(winner, 'proposal_order_only')]

        cap = self.circuit.area_caps[0]
        inside = {key: offer for key, offer in new.items() if layouts[key[0]] <= cap}
        outside = new.keys() - inside.keys()
        # The common opening counts toward the cap share. Only reserve a cap
        # anchor when both bands are affordable, so a sparse band cannot stall.
        if inside and outside and self._reserve_cap_anchor(measured, cap):
            offers = inside
        native = {key: joint_native[key] for key in offers}
        return super()._choose(layouts, offers, capacity, limit, guided, joint_native=native)

    def _choose_clean(self, layouts, offers, capacity, limit, guided, joint_native, measured):
        if joint_native is None or capacity != 1:
            raise ValueError('Terminal search ranks one native block at a time')
        cap = self.circuit.area_caps[0]
        if not any(self._areas[g] <= cap and self._deployment(g) is not None for g in measured):
            inside = {key: offer for key, offer in offers.items()
                      if key[0] not in measured and layouts[key[0]] <= cap}
            offers = inside or offers
        if self._point_acquisition:
            return [self._point_policy(layouts, offers, measured)] if offers else []
        native = {key: joint_native[key] for key in offers}
        return CrossCircuitSearch._choose(self, layouts, offers, capacity, limit, guided, joint_native=native)

    def _confirmation_offer(self, layouts, offers, measured):
        # Fault-root refinement remains cost-aware beyond the explicit cap
        # verification; the unconditional empirical-zero interrupt is disabled.
        return None

    def _score_offers(self, options, offers, owner, values, **kwargs):
        if values.ndim == 2:
            # Both endpoint improvements are normalized to the full [0, 1]
            # range. A smaller area span should not discount deployment risk.
            values = values[:, 0] + values[:, 1]
        return super()._score_offers(options, offers, owner, values, **kwargs)

    def _paired_upper_gap(self, contender, leader):
        common = self._labels[contender].keys() & self._labels[leader].keys()
        groups = defaultdict(list)
        for root in common:
            identity = self.roots[root].source, self._physical_cluster(root)
            groups[identity].append(self._labels[contender][root] - self._labels[leader][root])
        by_source = defaultdict(list)
        for (source, _), values in groups.items():
            by_source[source].append(fsum(values) / len(values))
        if by_source.keys() != self.weights.keys() or len(groups) < self._opening_physical_roots:
            return None
        # A symmetric, bounded prior supplies one virtual signed comparison.
        # Each source contributes variance even when all observed pairs agree.
        moments = []
        for source, values in by_source.items():
            support = len(values) + 1
            mean = fsum(values) / support
            variance = (fsum(value * value for value in values) + .5) / support - mean * mean
            weight = self.weights[source]
            moments.append((weight * mean, weight * weight * max(0., variance) / (support + 1)))
        return fsum(mean for mean, _ in moments) + 1.645 * sqrt(fsum(var for _, var in moments))

    def recommend(self, *, area_caps=None, conservative=None):
        if self.clean:
            if conservative is not None and type(conservative) is not bool:
                raise ValueError('The recommendation rule must be a boolean')
            caps = tuple(area_caps) if area_caps is not None else self.circuit.area_caps
            self._check_caps(caps)
            rows = []
            for genome in self._labels:
                row = self._deployment(genome)
                if row is not None and not self.circuit.physical_root_shrinkage:
                    row = self._summarize_physical_deployment(genome)
                if row is not None:
                    rows.append(row)
            frontier = deployment_frontier(rows, conservative=False)
            result = dict(frontier=frontier, recommendations=[dict(area_cap_ge=cap, layout=next(
                (row for row in reversed(frontier) if row['area_ge'] <= cap), None) if conservative is False else None)
                for cap in caps], paid_cpu=self.paid_cpu, reserved_cpu=self.reserved_cpu)
        else:
            result = super().recommend(area_caps=area_caps, conservative=conservative)
        if conservative is False:
            return result
        for item in result['recommendations']:
            contenders = self._contenders(self._labels, item['area_cap_ge'])
            if not contenders:
                continue
            leader = contenders[0]
            genome = tuple(leader['genome'])
            supported = [(0., 0)]
            for i, row in enumerate(contenders[1:], 1):
                gap = self._paired_upper_gap(tuple(row['genome']), genome)
                if gap is not None and gap < 0. and row['risk_interval'][0] <= leader['risk_interval'][1]:
                    supported.append((gap, i))
            chosen = contenders[min(supported)[1]]
            item['layout'] = copy_deployment(chosen)
        return result
