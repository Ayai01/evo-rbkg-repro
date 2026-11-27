"""Ask/tell optimization over declared native fault sources, costs and same-service area."""

from bisect import bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import KW_ONLY, dataclass
from hashlib import sha256
import json
from math import exp, fsum, isclose, isfinite, sqrt
from statistics import variance
from typing import Callable, Mapping

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .root_block_knowledge_gradient import PRIOR_RISK
from .source_frontier import CHUNK, _front_gain
from .source_risk_posterior import SourcePosterior, components
from .structural_search import propose_layouts


Genome = tuple[int, ...]


@dataclass(frozen=True)
class Root:
    source: str
    condition: str | None = None
    cluster: str | None = None


@dataclass(frozen=True)
class Quote:
    median_cpu: float
    reserve_cpu: float


@dataclass(frozen=True)
class Circuit:
    options: tuple[tuple[int, ...], ...]
    geometry: Callable[[Genome], tuple[Genome, float] | None]
    roots: Mapping[str, Root]
    blocks: Mapping[str, tuple[str, ...]]
    source_weights: Mapping[str, float]
    source_axes: Mapping[str, tuple[int, ...]]
    quote: Callable[[Genome, str, tuple[str, ...]], Quote]
    _: KW_ONLY
    reference: float
    max_area: float | None = None
    recovery_axes: tuple[int, ...] = ()
    area_origin: float = 0.
    area_caps: tuple[float, ...] = ()
    area_edges: tuple[float, ...] = ()
    anchors: tuple[Genome, ...] = ()
    candidates: tuple[Genome, ...] = ()
    opening: tuple[Genome, ...] = ()
    opening_blocks: tuple[str, ...] = ()
    proposal_count: int = 48
    min_source_roots: int = 1
    source_kernels: Callable | None = None
    has_new_support: Callable | None = None
    # Optional pre-purchase physical signatures.  A signature identifies the
    # actual hit locus, dose, and external/internal path for one registered
    # root.  The incident-conditioned model keeps those groups separate while
    # retaining the same shared-root accounting and native purchase ledger.
    incident_signatures: Mapping[str, tuple] | None = None
    incident_prior: Callable[[Genome, tuple], float] | None = None
    # Some registered catalogues contain layouts measured on different, but
    # source-balanced, root signatures.  The mapping records the roots that
    # are genuinely available for each finite-catalogue layout.  Omitting it
    # keeps the historical common-support contract.
    candidate_roots: Mapping[Genome, tuple[str, ...]] | None = None
    safe_anchors: tuple[Genome, ...] = ()
    # A deployment selected from a small, uneven root sample is not a
    # meaningful endpoint.  The incident route may require a fixed minimum
    # number of registered roots before a layout can enter its frontier.
    min_deployment_roots: int = 0
    incident_min_group_roots: int = 2
    # Optional training-only scores used to break acquisition ties.
    root_priority: Mapping[str, float] | None = None
    menu_candidate: Callable | None = None
    physical_root_shrinkage: bool = False


@dataclass(frozen=True)
class Ablation:
    physical_domains: bool = True
    paired_roots: bool = True
    directed_recovery: bool = True
    cost_aware: bool = True
    area_coverage: bool = True
    policy: str = 'kg'
    coverage_trigger: str = 'periodic'
    incident_conditioned: bool = False
    # Fraction of the grouped physical-signature gain retained by the
    # incident branch.  The complement is a source-level KG safeguard for
    # sparse signatures.
    incident_blend: float = 0.5
    # Let the settled native labels choose between the physical-signature and
    # source-level posteriors by leave-one-layout-out prediction error.
    incident_auto_blend: bool = False
    # Evaluate the physical and pooled acquisition surfaces together and keep
    # the larger per-offer gain.  This is a safe envelope, not an average that
    # can dilute a useful physical signal when the pooled model is stronger.
    incident_max_union: bool = False
    # Calibrate the structural incident prior from settled labels before using
    # it to generate directed proposals.  ``empirical`` removes the registered
    # directional prior entirely and uses only the hierarchical settled-label
    # model.  The area model is deliberately simple and monotone in measured
    # analytic area.  The factor model adds categorical protection/recovery
    # effects when enough layouts exist.
    incident_calibration: str = 'none'
    # Reserve finite proposal slots for layouts with purchasable support from
    # every required source. The opening and the underlying catalogue are shared.
    deployable_shortlist: bool = False


@dataclass(frozen=True)
class Purchase:
    id: str
    genome: Genome
    area_ge: float
    block: str
    sources: tuple[str, ...]
    root_ids: tuple[str, ...]
    quote: Quote
    reason: str


def _identity(*parts):
    return sha256(json.dumps(parts, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def _edit(genome, changes):
    result = list(genome)
    for axis, allele in changes:
        result[axis] = allele
    return tuple(result)


def _summary(values):
    n = len(values)
    mean = fsum(values)/n
    # A variance floor preserves uncertainty with one pair or identical observations.
    spread = max(variance(values) if n > 1 else .25, .25/n)
    return dict(mean=mean, stderr=sqrt(spread/n), roots=n)


def physical_deployment_summary(genome, area, labels, roots, weights, *, min_source_roots=1):
    """Summarize paid outcomes with one equally weighted vote per source and physical root."""
    clusters = {source: defaultdict(list) for source in weights}
    for root, outcome in labels.items():
        meta = roots[root]
        clusters[meta.source][meta.cluster or root].append(outcome)
    n = {source: len(groups) for source, groups in clusters.items()}
    if any(count < min_source_roots for count in n.values()):
        return None
    failures = {source: fsum(fsum(values) / len(values) for values in groups.values())
                for source, groups in clusters.items()}
    means = {source: (failures[source] + .5) / (count + 1.) for source, count in n.items()}
    mean = fsum(weights[source] * means[source] for source in n)
    half = 1.645 * sqrt(fsum(weights[s]**2 * means[s] * (1. - means[s]) / (n[s] + 2.) for s in n))
    return dict(genome=list(genome), area_ge=area, estimated_risk=mean,
                empirical_risk=fsum(weights[s] * failures[s] / n[s] for s in n),
                risk_interval=[max(0., mean-half), min(1., mean+half)],
                roots_by_source=n, risk_by_source=means)


def copy_deployment(row):
    return dict(row, genome=list(row['genome']), risk_interval=list(row['risk_interval']),
                roots_by_source=dict(row['roots_by_source']), risk_by_source=dict(row['risk_by_source']))


def deployment_frontier(rows, *, conservative):
    def metric(row):
        return row['risk_interval'][1] if conservative else row['estimated_risk']

    def order(row):
        support = sum(row['roots_by_source'].values()) if conservative else 0
        return row['area_ge'], metric(row), -support, row['genome']

    frontier, minimum = [], float('inf')
    for row in sorted(rows, key=order):
        value = metric(row)
        if value < minimum:
            frontier.append(copy_deployment(row))
            minimum = value
    return frontier


def _incident_shrinkage_weight(source_root_support, physical_root_support,
                               paired_loo_differences, *, support_scale=4.):
    """Return a continuous physical-posterior weight from retained evidence.

    ``paired_loo_differences`` contains physical-minus-pooled squared errors.
    Negative values favour the physical posterior.  Both root supports enter
    the reliability factor, so an attractive error contrast from a tiny root
    sample remains close to the pooled model.  The result is deliberately
    continuous and never changes the native label ledger.
    """
    source_support = float(source_root_support)
    physical_support = float(physical_root_support)
    if (not isfinite(source_support) or not isfinite(physical_support)
            or source_support < 0. or physical_support < 0.
            or support_scale <= 0. or not isfinite(support_scale)):
        raise ValueError('Incident root supports must be finite nonnegative values')
    differences = tuple(float(value) for value in paired_loo_differences)
    if any(not isfinite(value) for value in differences):
        raise ValueError('Paired leave-one-out errors must be finite')
    if not differences:
        mean_delta = 0.
        standard_error = 0.
    else:
        mean_delta = fsum(differences)/len(differences)
        spread = variance(differences) if len(differences) > 1 else 0.
        standard_error = sqrt(max(0., spread)/len(differences))
    # Support controls how much of the physical signal can be trusted.  The
    # geometric mean keeps both the source-wide and group-local counts active.
    source_factor = source_support/(source_support+support_scale)
    physical_factor = physical_support/(physical_support+support_scale)
    support_factor = sqrt(source_factor*physical_factor)
    # Scale the paired error contrast by its own uncertainty.  The fixed floor
    # only prevents a zero-variance tie from becoming a discontinuous switch.
    scale = max(standard_error, 1e-6)
    scaled_delta = mean_delta/scale
    evidence_factor = (0. if scaled_delta > 50. else
                       1. if scaled_delta < -50. else
                       1./(1.+exp(scaled_delta)))
    return dict(weight=max(0., min(1., support_factor*evidence_factor)),
                source_support=source_support, physical_support=physical_support,
                support_factor=support_factor, paired_loo_count=len(differences),
                paired_loo_mean_delta=mean_delta, paired_loo_stderr=standard_error,
                error_factor=evidence_factor)


class CrossCircuitSearch:
    """In-memory optimizer. The caller owns scientific registration and durable receipts.

    The known-label ledger contains only returned root results. An infeasible
    layout is excluded from acquisition and recommendation. Use ``feasible=None``
    for an unassessed transport or tool failure. Each receipt charges its DCO
    original once, even when it covers multiple purchases.
    """

    _predict_offer_outcomes = False

    def __init__(self, circuit, *, seed, cpu_budget, ablation=None):
        if type(seed) is not int or not isfinite(cpu_budget) or cpu_budget < 0.:
            raise ValueError('Use an integer seed and a finite nonnegative CPU budget')
        self.circuit = circuit
        self.mode = ablation or Ablation()
        if self.mode.policy not in ('kg', 'completion_kg', 'joint_completion_kg', 'coverage'):
            raise ValueError('Unknown root-block acquisition policy')
        if self.mode.coverage_trigger not in ('periodic', 'unresolved'):
            raise ValueError('Choose a periodic or unresolved-band coverage trigger')
        if not isfinite(self.mode.incident_blend) or not 0. <= self.mode.incident_blend <= 1.:
            raise ValueError('The incident/source acquisition blend must lie in [0, 1]')
        if type(self.mode.incident_auto_blend) is not bool:
            raise ValueError('The incident posterior blend gate must be boolean')
        if type(self.mode.incident_max_union) is not bool:
            raise ValueError('The incident acquisition envelope gate must be boolean')
        if self.mode.incident_calibration not in ('none', 'area', 'factor', 'hybrid', 'empirical'):
            raise ValueError('Choose none, area, factor, hybrid, or empirical incident calibration')
        self.seed, self.cpu_budget = seed, cpu_budget
        self.options = tuple(tuple(values) for values in circuit.options)
        self.roots, self.blocks = dict(circuit.roots), {k: tuple(v) for k, v in circuit.blocks.items()}
        self.recovery = tuple(circuit.recovery_axes)
        self.incident_signatures = (None if circuit.incident_signatures is None else
                                    {str(root): tuple(signature) for root, signature in circuit.incident_signatures.items()})
        self.candidate_roots = (None if circuit.candidate_roots is None else
                                {tuple(genome): tuple(str(root) for root in roots)
                                 for genome, roots in circuit.candidate_roots.items()})
        self.root_priority = ({} if circuit.root_priority is None else
                              {str(root): float(score) for root, score in circuit.root_priority.items()})
        self.safe_anchors = {tuple(genome) for genome in circuit.safe_anchors}
        axes = set(range(len(self.options)))
        roots = [root for block in self.blocks.values() for root in block]
        weights = dict(circuit.source_weights)
        if (not self.options or any(not values or len(values) != len(set(values))
                                    or any(type(value) is not int for value in values) for values in self.options)
                or not self.blocks or any(not block for block in self.blocks.values())
                or set(roots) != self.roots.keys() or len(roots) != len(set(roots))
                or not weights or set(weights) != circuit.source_axes.keys()
                or {root.source for root in self.roots.values()} != weights.keys()
                or any(not isfinite(w) or w <= 0. for w in weights.values())
                or len(set(self.recovery)) != len(self.recovery) or not set(self.recovery) <= axes
                or any(not set(path) <= axes for path in circuit.source_axes.values())
                or not isfinite(circuit.reference) or not 0. <= circuit.area_origin < circuit.reference
                or type(circuit.proposal_count) is not int or circuit.proposal_count < 1
                or type(circuit.min_source_roots) is not int or circuit.min_source_roots < 1):
            raise ValueError('Declare disjoint root blocks, categorical hardware axes, positive source weights and an area reference')
        if self.mode.incident_conditioned:
            if self.incident_signatures is None or set(self.incident_signatures) != set(self.roots):
                raise ValueError('Incident-conditioned acquisition needs one physical signature per registered root')
            if any(not signature for signature in self.incident_signatures.values()):
                raise ValueError('Incident signatures must be non-empty immutable tuples')
            if self.circuit.incident_prior is not None and not callable(self.circuit.incident_prior):
                raise ValueError('The incident prior must be a callable structural risk model')
            if circuit.candidates and not self.safe_anchors <= {tuple(g) for g in circuit.candidates}:
                raise ValueError('Every safe anchor must belong to the registered catalogue')
        if self.candidate_roots is not None:
            declared = {tuple(g) for g in circuit.candidates} if circuit.candidates else set()
            if not declared:
                declared = {tuple(g) for g in circuit.anchors}
            if set(self.candidate_roots) - declared:
                raise ValueError('Candidate root support must belong to the declared catalogue')
            if any(not roots or len(roots) != len(set(roots)) or not set(roots) <= self.roots.keys()
                   for roots in self.candidate_roots.values()):
                raise ValueError('Each candidate root support must contain distinct registered roots')
        if type(circuit.min_deployment_roots) is not int or circuit.min_deployment_roots < 0:
            raise ValueError('The deployment evidence threshold must be a nonnegative integer')
        self._supported_members = None
        self._completion_blocks = None
        if self.candidate_roots is not None:
            # Preserve the block's declared order, including for partial returns.
            position = {root: (block, i) for block, members in self.blocks.items()
                        for i, root in enumerate(members)}
            self._supported_members = {}
            self._completion_blocks = {}
            block_order = {block: i for i, block in enumerate(self.blocks)}
            for genome, members in self.candidate_roots.items():
                by_block = defaultdict(list)
                for root in members:
                    by_block[position[root][0]].append(root)
                self._supported_members[genome] = {
                    block: tuple(sorted(ids, key=lambda root: position[root][1]))
                    for block, ids in by_block.items()}
                counts = Counter(self.roots[root].source for root in members)
                if len(members) >= circuit.min_deployment_roots and all(
                        counts[source] >= circuit.min_source_roots for source in weights):
                    self._completion_blocks[genome] = tuple(sorted(by_block, key=block_order.__getitem__))
        self._shortlist_support = (self._completion_blocks if circuit.candidates and self.mode.deployable_shortlist
                                   and self.mode.policy in ('completion_kg', 'joint_completion_kg') else None)
        if any(root not in self.roots or not isfinite(score) or score < 0.
               for root, score in self.root_priority.items()):
            raise ValueError('Root priorities must be finite nonnegative scores for registered roots')
        if type(circuit.incident_min_group_roots) is not int or circuit.incident_min_group_roots < 1:
            raise ValueError('The incident group support threshold must be positive')
        self.cap = circuit.reference if circuit.max_area is None else circuit.max_area
        if not isfinite(self.cap) or not circuit.area_origin <= self.cap <= circuit.reference:
            raise ValueError('The legal search bound must lie within the declared area origin and reference')
        self._allele_codes = tuple({value: i for i, value in enumerate(values)} for values in self.options)
        self._check_caps(circuit.area_caps)
        area_edges = tuple(circuit.area_edges)
        if (tuple(sorted(set(area_edges))) != area_edges
                or any(not isfinite(x) or not circuit.area_origin < x <= self.cap for x in area_edges)):
            raise ValueError('Area edges must increase strictly within the registered search range')
        self.weights = {source: weight/fsum(weights.values()) for source, weight in weights.items()}
        condition_counts = Counter((root.source, root.condition) for root in self.roots.values())
        source_counts = Counter(root.source for root in self.roots.values())
        self.condition_mass = {key: count/source_counts[key[0]] for key, count in condition_counts.items()}
        self.axes = {source: tuple(sorted(set((*path, *self.recovery))))
                     for source, path in circuit.source_axes.items()}
        self.block_roots = {(block, source): tuple(root for root in members if self.roots[root].source == source)
                            for block, members in self.blocks.items() for source in self.weights}
        explicit = self._declared(circuit.candidates)
        anchors = self._declared(circuit.anchors)
        self._catalog = {**{g: area for g, area in anchors.items() if g in explicit}, **explicit} if circuit.candidates else anchors
        openings = self._declared(circuit.opening, required=True)
        if circuit.candidates and not openings.keys() <= explicit.keys():
            raise ValueError('The common opening must belong to the finite native purchase catalogue')
        if not openings:
            openings = dict(list(self._catalog.items())[:1])
            if not openings and not circuit.candidates:
                openings = self._ordinary({}, 0, count=1)
        opening_blocks = tuple(circuit.opening_blocks) or tuple(self.blocks)[:1]
        if len(set(opening_blocks)) != len(opening_blocks) or not set(opening_blocks) <= self.blocks.keys():
            raise ValueError('The shared opening must use registered development blocks')
        self._opening = [(genome, block) for genome in openings for block in opening_blocks]
        self._areas = dict(openings)
        self._labels = defaultdict(dict)
        self._deployment_cache = {}
        self._service, self._fees, self._receipts, self._pending = {}, {}, {}, {}
        self._decisions = []
        self._round = self._issued = 0
        self._incident_runtime_blend = self.mode.incident_blend
        # A data-driven gate may select the pooled completion model for a
        # decision.  Keep recommendation semantics aligned with that model
        # instead of retaining the conservative interval rule of the
        # incident branch after the acquisition has safely fallen back.
        self._incident_runtime_pooled = False
        self._incident_prior_override = None
        self._incident_calibration_cache_key = None
        self._incident_calibration_cache = {}
        self._incident_validation = {}
        self._transfer_history = []
        self._inferred_roots = set()

    def _check_caps(self, caps):
        if any(not isfinite(cap) or not self.circuit.area_origin <= cap <= self.circuit.reference for cap in caps):
            raise ValueError('Use a same-service area cap within the declared origin and reference')

    def _geometry(self, genome):
        result = self.circuit.geometry(tuple(genome))
        if result is None:
            return None
        code, area = tuple(result[0]), result[1]
        if (len(code) != len(self.options) or any(allele not in values or type(allele) is not int
                                                 for allele, values in zip(code, self.options, strict=True))
                or not isfinite(area) or area < self.circuit.area_origin):
            raise ValueError('Geometry must return a declared canonical genome and valid analytic area')
        return (code, area) if area <= self.cap else None

    def _declared(self, genomes, *, required=False):
        catalog = {}
        for genome in genomes:
            admitted = self._geometry(genome)
            if admitted is None:
                if required:
                    raise ValueError('Every registered opening genome must pass the search geometry')
                continue
            code, area = admitted
            if code in catalog and catalog[code] != area:
                raise ValueError('A canonical design must have one analytic area')
            catalog[code] = area
        return catalog

    @property
    def paid_cpu(self):
        return fsum(self._fees.values())

    @property
    def reserved_cpu(self):
        return fsum(row.quote.reserve_cpu for row in self._pending.values())

    def _seed(self, number, purpose, *parts):
        return int(_identity(self.seed, number, purpose, *parts)[:16], 16)

    def _ordinary(self, parents, number, *, count=None, occupied=()):
        groups, weights = self._proposal_groups()
        return propose_layouts(self.options, self._geometry, parents, occupied,
                               count=self.circuit.proposal_count if count is None else count,
                               seed=self._seed(number, 'proposals'), area_edges=self.circuit.area_edges,
                               groups=groups, group_weights=weights)

    def _incident_guidance(self, parents, occupied):
        raise ValueError('This FIR reproduction uses the frozen span-balanced profile')

    def _proposal_groups(self):
        """Return source-weighted physical mutation groups for new layouts.

        Source-conditioned acquisition is only useful when the proposal pool
        can reach the same physical domains.  Aggregate overlapping source
        paths into one group and use settled deployment spread as a bounded
        feedback term.  The round-robin proposal sampler still visits every
        group, so a temporarily quiet source cannot disappear from the pool.
        """
        if not self.mode.physical_domains:
            return (), None
        grouped = defaultdict(float)
        for source, weight in self.weights.items():
            group = tuple(sorted(set(self.axes[source])))
            if group:
                grouped[group] += weight
        if not grouped:
            return (), None
        groups = tuple(sorted(grouped))
        weights = [grouped[group] for group in groups]
        deployed = [self._deployment(genome) for genome in self._labels]
        deployed = [row for row in deployed if row is not None]
        if len(deployed) >= 4:
            risks = {tuple(row['genome']): row['estimated_risk'] for row in deployed}
            adapted = []
            for group, base in zip(groups, weights, strict=True):
                buckets = defaultdict(list)
                for genome, risk in risks.items():
                    buckets[tuple(genome[axis] for axis in group)].append(risk)
                if len(buckets) < 2:
                    adapted.append(base)
                    continue
                means = [fsum(values) / len(values) for values in buckets.values()]
                center = fsum(means) / len(means)
                spread = fsum((value - center) ** 2 for value in means) / len(means)
                adapted.append(base * (1. + 4. * spread))
            weights = adapted
        return groups, weights

    def _standard(self, parents):
        cold = {genome: area for genome, area in self._catalog.items() if genome not in self._areas
                and (self._shortlist_support is None or genome in self._shortlist_support)}
        if not self.circuit.candidates:
            cold.update(self._ordinary(parents, self._round, occupied=self._areas))
        bands = defaultdict(deque)
        for genome in sorted(cold, key=lambda g: self._seed(self._round, 'finite-proposals', g)):
            bands[bisect_right(self.circuit.area_edges, cold[genome])].append(genome)
        order = sorted(bands, key=lambda band: self._seed(self._round, 'area-bands', band))
        offered = {}
        while order and len(offered) < self.circuit.proposal_count:
            for band in order:
                if len(offered) == self.circuit.proposal_count:
                    break
                genome = bands[band].popleft()
                offered[genome] = cold[genome]
            order = [band for band in order if bands[band]]
        return offered

    def _parents(self):
        rows = [row for genome in self._labels if (row := self._deployment(genome)) is not None]
        if not rows:
            return {}

        def metric(row):
            return (row['risk_interval'][1] if self.mode.incident_conditioned or self.circuit.physical_root_shrinkage
                    else row['estimated_risk'])

        # A risk-only parent list collapses the proposal distribution around
        # the first lucky low-risk layout.  Keep one evidence-backed parent in
        # every registered area band, then fill the remaining slots with a
        # Hamming-diverse set of low-risk parents.  This preserves the middle
        # of the Pareto geometry while retaining exploitation pressure.
        bands = defaultdict(list)
        for row in rows:
            bands[bisect_right(self.circuit.area_edges, row['area_ge'])].append(row)
        selected = []
        chosen = set()
        for band in sorted(bands):
            row = min(bands[band], key=lambda item: (metric(item), item['area_ge'], item['genome']))
            selected.append(row)
            chosen.add(tuple(row['genome']))
            if len(selected) == 8:
                break
        remaining = [row for row in rows if tuple(row['genome']) not in chosen]
        while remaining and len(selected) < 8:
            def diversity(row):
                distance = min(sum(a != b for a, b in zip(row['genome'], other['genome'], strict=True))
                               for other in selected)
                return distance, -metric(row), -row['area_ge'], row['genome']
            row = max(remaining, key=diversity)
            selected.append(row)
            remaining.remove(row)
        return {tuple(row['genome']): row['area_ge'] for row in selected}

    def _labels_by_root(self):
        by_root = defaultdict(dict)
        for genome, labels in self._labels.items():
            if self._service.get(genome) is True:
                for root, value in labels.items():
                    by_root[root][genome] = value
        return by_root

    def _protection(self, source):
        axes = self.axes[source] if self.mode.physical_domains else range(len(self.options))
        return tuple(i for i in axes if i not in self.recovery)

    def _single_moves(self, genomes, axes):
        """Index observed one-axis neighbors in the original axis and allele order."""
        if len(genomes) > len(axes):
            support = set(genomes)
            return {g: {(axis, g[axis], allele): changed for axis in axes for allele in self.options[axis]
                        if allele > g[axis] and (changed := _edit(g, ((axis, allele),))) in support} for g in genomes}
        axis_order = {axis: i for i, axis in enumerate(axes)}
        neighbors = {g: {} for g in genomes}
        for i, left in enumerate(genomes):
            for right in genomes[i+1:]:
                differences = (axis for axis, (a, b) in enumerate(zip(left, right, strict=True)) if a != b)
                axis = next(differences, None)
                if axis not in axis_order or next(differences, None) is not None:
                    continue
                low, high = (left, right) if left[axis] < right[axis] else (right, left)
                neighbors[low][axis, low[axis], high[axis]] = high
        return {g: dict(sorted(offers.items(), key=lambda row: (axis_order[row[0][0]],
                                self._allele_codes[row[0][0]][row[0][2]]))) for g, offers in neighbors.items()}

    def effects(self):
        """Return exact matched-root marginal and four-cell protection × recovery estimates."""
        if not self.mode.paired_roots:
            return []
        paired = defaultdict(lambda: defaultdict(list))
        protection_by_source = {source: self._protection(source) for source in self.weights}
        move_cache = {}
        for root_id, measured in self._labels_by_root().items():
            root = self.roots[root_id]
            protection = protection_by_source[root.source]
            pattern = root.source, tuple(measured)
            if pattern not in move_cache:
                move_cache[pattern] = self._single_moves(pattern[1], (*protection, *self.recovery))
            for genome, moves in move_cache[pattern].items():
                original = measured[genome]
                neighbors = {move: measured[target] for move, target in moves.items()}
                for move, terminal in neighbors.items():
                    held = tuple(i for i in self.recovery if i != move[0])
                    context = tuple((i, genome[i]) for i in held)
                    key = (root.source, root.condition, (move,), context)
                    paired[key][root_id].append((terminal-original,))
                for p, protection_risk in neighbors.items():
                    if p[0] not in protection:
                        continue
                    for r, recovery_risk in neighbors.items():
                        if r[0] not in self.recovery:
                            continue
                        joint = _edit(genome, ((p[0], p[2]), (r[0], r[2])))
                        if joint in measured:
                            full = measured[joint]
                            context = tuple((i, genome[i]) for i in self.recovery if i != r[0])
                            key = (root.source, root.condition, (p, r), context)
                            paired[key][root_id].append((full-original, full-protection_risk-recovery_risk+original,
                                                        protection_risk-original, full-recovery_risk,
                                                        recovery_risk-original, full-protection_risk))
        result = []
        for (source, condition, moves, context), groups in sorted(paired.items(), key=lambda row: str(row[0])):
            columns = zip(*(tuple(fsum(part)/len(part) for part in zip(*rows, strict=True)) for rows in groups.values()),
                          strict=True)
            summaries = [_summary(list(col)) for col in columns]
            names = ('delta', 'interaction', 'protection_without_recovery', 'protection_with_recovery',
                     'recovery_without_protection', 'recovery_with_protection')
            result.append(dict(source=source, condition=condition, moves=moves, held_context=context,
                               **dict(zip(names, summaries))))
        return result

    def _matched_guidance(self, parents):
        guided = {}
        for row in self.effects():
            delta = row['delta']
            interaction = row.get('interaction')
            # Require a signed effect after a one-standard-error margin. Formulae and
            # support travel with the evidence instead of filling unseen factorial cells.
            if delta['roots'] < 2 or abs(delta['mean']) <= delta['stderr']:
                continue
            if interaction and interaction['mean']+interaction['stderr'] >= 0.:
                continue
            reverse = delta['mean'] > 0.
            moves = [(axis, high if reverse else low, low if reverse else high) for axis, low, high in row['moves']]
            for parent in parents:
                if (any(parent[axis] != allele for axis, allele in row['held_context'])
                        or any(parent[axis] != source for axis, source, _ in moves)):
                    continue
                offer = self._geometry(_edit(parent, [(axis, target) for axis, _, target in moves]))
                if offer is not None and (not self.circuit.candidates or offer[0] in self._catalog):
                    mass = self.weights[row['source']]*self.condition_mass[row['source'], row['condition']]
                    benefit = mass*(abs(delta['mean'])-delta['stderr'])*delta['roots']/(delta['roots']+4)
                    guided[offer[0]] = max(guided.get(offer[0], 0.), benefit)
        return guided

    def _unpaired_guidance(self, parents):
        raise ValueError('This FIR reproduction uses the frozen span-balanced profile')

    def _needed_roots(self, genome, block):
        known = self._labels.get(genome, {})
        support = (self.blocks[block] if self._supported_members is None else
                   self._supported_members.get(genome, {}).get(block, ()))
        missing = tuple(root for root in support if root not in known) if known else support
        if missing and self.circuit.has_new_support is not None and not self.circuit.has_new_support(genome, missing):
            return ()
        return missing

    def _offers(self, keys, limit, held):
        offers = {}
        for genome, block in keys:
            if genome in held or self._service.get(genome) is False or genome in offers:
                continue
            missing = self._needed_roots(genome, block)
            if not missing:
                continue
            quote = self.circuit.quote(genome, block, missing)
            if not isinstance(quote, Quote) or not 0. < quote.median_cpu <= quote.reserve_cpu or not isfinite(quote.reserve_cpu):
                raise ValueError('Each native purchase needs positive and conservative CPU quotes')
            if quote.reserve_cpu <= limit:
                sources = tuple(source for source in self.weights if any(self.roots[root].source == source for root in missing))
                offers[genome] = (block, sources, missing, quote)
        return offers

    def _source_kernel_components(self, codes, axes, layouts):
        return components(codes, axes, project=self.mode.physical_domains)

    def _apply_short_source_evidence(self, source, layouts, posterior):
        pass

    def _physical_cluster(self, root):
        return self.roots[root].cluster or root

    def _cluster_rates(self, labels, roots):
        clusters = defaultdict(list)
        for root in roots:
            clusters[self._physical_cluster(root)].append(labels[root])
        return [fsum(values) / len(values) for values in clusters.values()]

    def _posteriors(self, layouts, offers, active, *, root_overlap=False, offer_rows=None):
        location = {genome: i for i, genome in enumerate(layouts)}
        axes = dict(self.axes)
        means, posts = np.zeros(len(layouts)), {}
        residual = (getattr(self, 'physical_locus_residual', None)
                    if self.mode.physical_domains and self.mode.paired_roots else None)
        paid = ({g: self._labels[g] for g in active if self._service.get(g) is True}
                if residual is not None else None)
        if self.circuit.source_kernels is None:
            # Hamming covariance uses equality within each axis. Use the fixed option codebook.
            genome_codes = np.asarray([[codes[value] for value, codes in zip(g, self._allele_codes, strict=True)]
                                       for g in layouts])
            kernels = self._source_kernel_components(genome_codes, axes, layouts)
        else:
            kernels = self.circuit.source_kernels(tuple(layouts), axes, project=self.mode.physical_domains)
        for source, kernel in kernels:
            signatures, counts, aliases = {}, {}, {}
            block_roots = {}
            unambiguous = True

            def event(genome, roots):
                nonlocal unambiguous
                if residual is not None:
                    roots = tuple(sorted(roots))
                key = (tuple(sorted({self._physical_cluster(r) for r in roots}))
                       if self.circuit.physical_root_shrinkage else tuple(sorted(roots)))
                index = signatures.setdefault(key, len(signatures))
                counts[index] = len(key)
                if residual is not None and block_roots.setdefault(index, roots) != roots:
                    # One coarse noise key can combine alternative conditions of
                    # the same cluster. Keep it coarse if the exact roots differ.
                    unambiguous = False
                return location[genome], index

            known, waiting, available = {}, [], []
            for genome in active:
                if self._service.get(genome) is not True:
                    continue
                labels = self._labels[genome]
                for block in self.blocks:
                    present = [root for root in self.block_roots[block, source] if root in labels]
                    if present:
                        if self.circuit.physical_root_shrinkage:
                            rates = self._cluster_rates(labels, present)
                            observed = (fsum(rates) + .5) / (len(rates) + 1.)
                        else:
                            observed = fsum(labels[root] for root in present) / len(present)
                        known[event(genome, present)] = observed
            for purchase in self._pending.values():
                if purchase.genome in location and source in purchase.sources:
                    waiting.append(event(purchase.genome, (r for r in purchase.root_ids if self.roots[r].source == source)))
            for genome, (block, sources, roots, _) in (offers.items() if offer_rows is None else offer_rows):
                if source in sources:
                    key = event(genome, (r for r in roots if self.roots[r].source == source))
                    aliases[location[genome], block] = len(available)
                    available.append(key)
            root_sets = {i: roots for roots, i in signatures.items()} if root_overlap else None
            # The source-level posterior is the non-incident model.  A
            # structural incident prior is meaningful only inside
            # ``_incident_posteriors``, where the physical signature group is
            # defined.  Applying it here would leak an undefined group and
            # break every ordinary control route.
            baseline = np.full(len(layouts), PRIOR_RISK)
            effects = None
            if residual is not None and unambiguous:
                effects = residual.factors(source, layouts, self.options, paid, [*known, *waiting, *available],
                                           block_roots, aggregate_by_cluster=self.circuit.physical_root_shrinkage)
            extra = {'physical_residual': effects} if effects is not None else {}
            post = SourcePosterior(kernel, baseline, known, waiting, available,
                                   counts, paired=self.mode.paired_roots, root_sets=root_sets,
                                   predict_outcomes=self._predict_offer_outcomes, **extra)
            post.index.update(aliases)
            self._apply_short_source_evidence(source, layouts, post)
            means += self.weights[source]*post.mean
            posts[source] = post
        return location, means, posts

    def _incident_posteriors(self, layouts, offers, active, *, batch_covariance=True, joint_native=None):
        raise ValueError('This FIR reproduction uses the frozen span-balanced profile')

    def _incident_front_gain(self, mean, posts, group_weights, offers, offer_parts,
                             active, areas, reference, samples):
        """Expected front gain for the grouped physical-signature posteriors."""
        active = sorted(active, key=lambda genome: (areas[genome], genome))
        locations = {genome: i for i, genome in enumerate(areas)}
        area_values = np.asarray(list(areas.values()), dtype=float)
        active_indices = [locations[g] for g in active]
        offer_keys = list(offers)
        layouts = np.asarray([locations[offer_parts[key][0]] for key in offer_keys])
        draws = len(samples)
        states = np.zeros((draws, len(offer_keys), len(active_indices)))
        proposed = np.zeros((draws, len(offer_keys)))
        for column, (group, post) in enumerate(posts.items()):
            indices, columns = [], []
            for i, key in enumerate(offer_keys):
                if key not in post.index:
                    continue
                indices.append(i)
                columns.append(post.index[key])
            if not indices:
                continue
            variance = post.standard_gain(columns).T*group_weights[group]
            random = samples[:, column, None]
            states[:, indices, :] += random[:, :, None]*variance[:, active_indices]
            proposed[:, indices] += random*variance[np.arange(len(indices)), layouts[indices]]
        states += mean[active_indices]
        proposed += mean[layouts]
        left = np.r_[0., np.asarray([areas[g] for g in active])]
        right = np.r_[np.asarray([areas[g] for g in active]), reference]
        widths = (right-left)/reference
        best = np.minimum.accumulate(np.r_[1., np.clip(mean[active_indices], 0., 1.)])
        base = (1.-best)@widths
        envelope = np.ones((draws, len(offer_keys), len(active_indices)+1))
        envelope[:, :, 1:] = np.clip(states, 0., 1.)
        np.minimum.accumulate(envelope, axis=2, out=envelope)
        value = np.einsum('qaj,j->a', 1.-envelope, widths)/draws
        extra_width = np.maximum(right-np.maximum(left, area_values[layouts, None]), 0.)/reference
        drop = np.maximum(envelope-np.clip(proposed, 0., 1.)[:, :, None], 0.)
        value += np.einsum('qaj,aj->a', drop, extra_width)/draws
        return np.maximum(value-base, 0.)

    def _rank_knowledge_gradient(self, options, gains, tie, owner):
        return min(options, key=lambda key: (-gains[key], tie[key]))

    def _score_offers(self, options, offers, owner, values, *, layouts, location, posts, samples):
        return {key: float(gain)/(offers[key][3].median_cpu if self.mode.cost_aware else 1.)
                for key, gain in zip(options, values, strict=True)}

    def _frontier_gains(self, means, posts, items, incumbents, areas, samples):
        return _front_gain(means, posts, self.weights, items, incumbents, areas,
                           self.circuit.reference-self.circuit.area_origin, samples)

    def _exploration_options(self, options, owner):
        return options

    def _choose(self, layouts, offers, capacity, limit, guided, *, joint_native=None):
        active = {genome for genome in layouts if self._labels.get(genome)}
        active.update(p.genome for p in self._pending.values() if p.genome in layouts)
        joint = joint_native is not None
        owner = {key: key[0] if joint else key for key in offers}
        completion = self.mode.policy in ('completion_kg', 'joint_completion_kg')
        modeled = self.mode.policy != 'coverage'
        attained = {g for g in active if self._deployment(g) is not None} if completion else active
        # A periodic coverage purchase ignores posterior scores. A wider wave
        # still needs its posterior when any later purchase uses KG.
        need_posterior = modeled and not (capacity == 1 and self.mode.area_coverage
            and self.mode.coverage_trigger == 'periodic' and self._issued % 4 == 3)
        incident_weights = None
        incident_parts = None
        incident_source_means = None
        incident_source_posts = None
        if need_posterior:
            if self.mode.incident_conditioned:
                (location, means, posts, incident_weights,
                 incident_source_means, incident_source_posts) = self._incident_posteriors(
                    layouts, offers, sorted(active), batch_covariance=capacity > 1, joint_native=joint_native)
                incident_parts = {key: (owner[key], offer[0], offer[1], tuple(offer[2]))
                                  for key, offer in offers.items()}
                if self.mode.incident_auto_blend:
                    # Keep both posterior surfaces active.  The validated
                    # continuous weight below controls their acquisition
                    # contribution without changing the incident route or
                    # its future conditioning semantics.
                    self._incident_runtime_pooled = False
            elif joint:
                # Each option has a distinct (layout, block) key. For a multi-block
                # plan, separately index the first native return for pending fantasies.
                rows = [(owner[key], offer) for key, offer in offers.items()]
                for key, native in joint_native.items():
                    if native[2] != offers[key][2]:
                        rows.append((owner[key], (('native', native[0]), *native[1:])))
                location, means, posts = self._posteriors(layouts, offers, sorted(active),
                                                          root_overlap=True, offer_rows=rows)
            elif completion:
                location, means, posts = self._posteriors(layouts, offers, sorted(active), root_overlap=True)
            else:
                location, means, posts = self._posteriors(layouts, offers, sorted(active))
            rng = np.random.default_rng(self._seed(self._round, 'fantasies'))
            samples = ndtri(np.clip(qmc.Sobol(len(posts), rng=rng).random_base2(3), 1e-10, 1.-1e-10))
            samples = np.vstack((samples, -samples))
            areas = np.asarray(list(layouts.values()))-self.circuit.area_origin
        reserved, selected = 0., []
        available = list(offers)
        for index in range(capacity):
            available = [g for g in available if fsum((reserved, offers[g][3].reserve_cpu)) <= limit]
            if not available:
                break
            ordinal = self._issued+index
            force_coverage = self.mode.area_coverage and (
                not modeled or self.mode.coverage_trigger == 'periodic' and ordinal % 4 == 3)
            options = available
            tie = {key: (-fsum(self.root_priority.get(root, 0.) for root in offers[key][2]),
                         self._seed(self._round, 'ties', owner[key], offers[key][0]))
                   for key in options}
            gains = {}
            safe_winner = None
            if self.mode.incident_conditioned and self.safe_anchors:
                anchor_bands = {bisect_right(self.circuit.area_edges, layouts[genome])
                                for genome in attained if genome in self.safe_anchors}
                unfinished = [key for key in options
                              if owner[key] in self.safe_anchors and owner[key] not in attained
                              and bisect_right(self.circuit.area_edges, layouts[owner[key]]) not in anchor_bands]
                if unfinished:
                    anchor_bands = {key: bisect_right(self.circuit.area_edges, layouts[owner[key]])
                                    for key in unfinished}
                    anchor_counts = Counter(anchor_bands.values())
                    safe_winner = min(unfinished, key=lambda key: (
                        anchor_counts[anchor_bands[key]], layouts[owner[key]], tie[key]))
            if safe_winner is None and modeled and not force_coverage:
                if self.mode.incident_conditioned:
                    incident_values = np.concatenate([self._incident_front_gain(
                        means, posts, incident_weights, options[i:i+CHUNK],
                        incident_parts, attained,
                        {g: value-self.circuit.area_origin for g, value in layouts.items()},
                        self.circuit.reference-self.circuit.area_origin, samples)
                        for i in range(0, len(options), CHUNK)])
                    if incident_source_posts:
                        items = [(location[owner[key]], *offers[key][:2]) for key in options]
                        source_values = np.concatenate([_front_gain(
                            incident_source_means, incident_source_posts, self.weights,
                            items[i:i+CHUNK], {location[g] for g in attained}, areas,
                            self.circuit.reference-self.circuit.area_origin, samples)
                            for i in range(0, len(items), CHUNK)])
                        if self.mode.incident_max_union and not self.mode.incident_auto_blend:
                            values = np.maximum(incident_values, source_values)
                        else:
                            blend = (self._incident_runtime_blend if self.mode.incident_auto_blend
                                     else self.mode.incident_blend)
                            if self.mode.incident_max_union:
                                # Shrink only the proposed uplift over the source
                                # surface. A plain max discarded the learned weight.
                                incident_values = np.maximum(incident_values, source_values)
                            values = (blend*incident_values + (1.-blend)*source_values)
                    else:
                        values = incident_values
                else:
                    items = [(location[owner[key]], *offers[key][:2]) for key in options]
                    incumbents = {location[g] for g in attained}
                    values = np.concatenate([self._frontier_gains(means, posts, items[i:i+CHUNK],
                                                                  incumbents, areas, samples)
                                             for i in range(0, len(items), CHUNK)])
                outcome_posts = incident_source_posts if self.mode.incident_conditioned else posts
                gains = self._score_offers(options, offers, owner, values, layouts=layouts, location=location,
                                           posts=outcome_posts, samples=samples)
            if safe_winner is not None:
                winner, reason = safe_winner, 'safe_anchor'
            elif gains and max(gains.values()) > 1e-12:
                winner = self._rank_knowledge_gradient(options, gains, tie, owner)
                reason = 'knowledge_gradient'
            else:
                options = self._exploration_options(options, owner)
                if not options:
                    break
                coverage = ()
                if self.mode.area_coverage:
                    bands = {g: bisect_right(self.circuit.area_edges, layouts[g]) for g in layouts}
                    counts = Counter(bands[g] for g in attained)
                    coverage = (options if force_coverage or self.mode.coverage_trigger == 'periodic'
                                else [key for key in options if not counts[bands[owner[key]]]])
                if coverage:
                    winner = min(coverage, key=lambda key: (counts[bands[owner[key]]], -min(
                        (sum(x != y for x, y in zip(owner[key], other, strict=True)) for other in active),
                        default=0), tie[key]))
                    reason = 'area_coverage'
                else:
                    winner, reason = min(options, key=tie.__getitem__), 'seeded_exploration'
            genome = owner[winner]
            available = [key for key in available if owner[key] != genome]
            active.add(genome)
            block, sources, _, quote = offers[winner]
            if joint and joint_native[winner][2] != offers[winner][2]:
                native = joint_native[winner]
                block, sources = ('native', native[0]), native[1]
            else:
                attained.add(genome)
            reserved = fsum((reserved, quote.reserve_cpu))
            selected.append((winner, 'directed_'+reason if genome in guided else reason))
            if need_posterior and index + 1 < capacity and available:
                if self.mode.incident_conditioned:
                    selected_key = (genome,block) if joint else winner
                    # Condition both retained posterior surfaces.  A blended
                    # acquisition decision must advance the source and
                    # physical models together so its metadata remains
                    # traceable on the next decision.
                    for group, post in posts.items():
                        if selected_key in post.index:
                            post.condition(selected_key)
                    if incident_source_posts:
                        source_key = (location[genome], block)
                        for source in sources:
                            post = incident_source_posts.get(source)
                            if post is not None and source_key in post.index:
                                post.condition(source_key)
                else:
                    for source in sources:
                        if any(source in offers[g][1] for g in available):
                            posts[source].condition((location[genome], block))
        return selected

    def _purchase_blocks(self, picks, offers, capacity):
        return ((genome, reason, offers[genome]) for genome, reason in picks)

    def _replace_public_proposal(self, standard, targeted, limit, held):
        """Allow one paid-prefix proposal to replace the final ordinary slot."""
        provider = self.circuit.menu_candidate
        if provider is None or not standard:
            return None
        occupied = set(self._areas) | standard.keys() | targeted.keys()
        candidate = provider(self, occupied, limit, held)
        if candidate is None:
            return None
        if candidate in occupied or candidate not in self._catalog:
            raise ValueError('The proposed replacement must be an unseen legal public layout')
        displaced = next(reversed(standard))
        del standard[displaced]
        standard[candidate] = self._catalog[candidate]
        return candidate, displaced

    def ask(self, capacity):
        """Reserve requested CPU and return at most ``capacity`` independently purchasable root blocks."""
        if type(capacity) is not int or capacity < 0:
            raise ValueError('Capacity must count a nonnegative number of purchases')
        limit = self.cpu_budget-self.paid_cpu-self.reserved_cpu
        if not capacity or limit <= 0.:
            return ()
        held = {purchase.genome for purchase in self._pending.values()}
        opening = [(g, b) for g, b in self._opening if self._service.get(g) is not False and self._needed_roots(g,b)]
        replacement = None
        if opening:
            layouts = self._areas
            offers = self._offers(opening, limit, held)
            proposed = [(g, 'shared_opening') for g in offers]
            picks, fees = [], []
            for genome in offers:
                quote = offers[genome][3]
                if len(picks) < capacity and fsum((*fees, quote.reserve_cpu)) <= limit:
                    picks.append((genome, 'shared_opening'))
                    fees.append(quote.reserve_cpu)
        else:
            parents = self._parents()
            standard = self._standard(parents)
            guided = {}
            if self.mode.incident_conditioned:
                guided.update(self._incident_guidance(parents, self._areas))
            if self.mode.directed_recovery and self.recovery and parents:
                directed = (self._matched_guidance(parents) if self.mode.paired_roots
                            else self._unpaired_guidance(parents))
                for genome, score in directed.items():
                    guided[genome] = max(guided.get(genome, 0.), score)
            unseen = guided.keys() - self._areas.keys()
            if self._shortlist_support is not None:
                unseen.intersection_update(self._shortlist_support)
            shortlist = sorted(unseen, key=lambda g: (-guided[g], self._seed(self._round, 'guided-ties', g)))
            guided = {g: guided[g] for g in shortlist[:self.circuit.proposal_count//2]}
            targeted = {g: self._geometry(g)[1] for g in guided}
            ordinary = [(g, area) for g, area in standard.items() if g not in targeted]
            standard = dict(ordinary[:self.circuit.proposal_count-len(targeted)])
            replacement = self._replace_public_proposal(standard, targeted, limit, held)
            proposed = [*((g, 'exposure' if replacement and g == replacement[0] else 'ordinary') for g in standard),
                        *((g, 'directed') for g in targeted)]
            layouts = {g: area for g, area in {**self._areas, **standard, **targeted}.items()
                       if self._service.get(g) is not False}
            if self.mode.policy == 'joint_completion_kg':
                from .deployable_blocks import completion_candidates
                offers, native_blocks = {}, {}
                for genome, bundle, native in completion_candidates(
                        self, layouts, limit, held, all_blocks=True, blocks_by_layout=self._completion_blocks,
                        all_starts=self.mode.incident_conditioned):
                    key = (genome, native[0])
                    offers[key], native_blocks[key] = bundle, native
                actions = self._choose(layouts, offers, capacity, limit, guided,
                                       joint_native=native_blocks) if offers else []
                picks = [(key[0], reason) for key, reason in actions]
                offers = {key[0]: native_blocks[key] for key, _ in actions}
            elif self.mode.policy == 'completion_kg':
                from .deployable_blocks import completion_offers
                offers, native_blocks = completion_offers(self, layouts, limit, held,
                                                         blocks_by_layout=self._completion_blocks)
                picks = self._choose(layouts, offers, capacity, limit, guided) if offers else []
                offers = native_blocks
            else:
                offers = self._offers(((g, block) for g in layouts for block in self.blocks), limit, held)
                picks = self._choose(layouts, offers, capacity, limit, guided) if offers else []
        selected = []
        for genome, reason, (block, sources, roots, quote) in self._purchase_blocks(picks, offers, capacity):
            purchase = Purchase(_identity(self.seed, self._round, genome, block, roots), genome, layouts[genome],
                                block, sources, roots, quote, reason)
            self._areas[genome] = purchase.area_ge
            self._pending[purchase.id] = purchase
            selected.append(purchase)
        if proposed or selected:
            decision = dict(round=self._round, opening=bool(opening),
                proposed=[dict(genome=list(g), area_ge=layouts[g], route=route) for g, route in proposed],
                purchase_ids=[purchase.id for purchase in selected], purchases_requested=capacity,
                purchased=len(selected))
            if replacement is not None:
                decision['proposal_replacement'] = dict(added=list(replacement[0]), displaced=list(replacement[1]))
            if self.mode.incident_conditioned and self._incident_validation:
                decision['incident_validation'] = self.incident_validation()
            self._decisions.append(decision)
            self._round += 1
        if selected:
            self._issued += len(selected)
        return tuple(selected)

    def decision_history(self):
        """Serializable shortlist, purchase counts and completed receipt counts by wave."""
        return [dict(row, settled_purchases=sum(key in self._receipts for key in row['purchase_ids']))
                for row in self._decisions]

    def incident_validation(self):
        """Return the latest source/physical posterior selection evidence."""
        return json.loads(json.dumps(self._incident_validation, allow_nan=False))

    def tell(self, purchase, outcomes, *, original_id, paid_cpu, feasible=True):
        """Consume a terminal charge and any returned roots, leaving absent roots unknown."""
        token = purchase.id if isinstance(purchase, Purchase) else purchase
        values = dict(outcomes)
        if (not isinstance(original_id, str) or not original_id or not isfinite(paid_cpu) or paid_cpu < 0.
                or feasible is not None and type(feasible) is not bool
                or any(not isfinite(value) or not 0. <= value <= 1. for value in values.values())):
            raise ValueError('Report a real original charge, a valid service verdict and bounded terminal risks')
        record = (original_id, paid_cpu, feasible, values)
        if token in self._receipts:
            if self._receipts[token] != record:
                raise ValueError('A paid purchase cannot change its original receipt or labels')
            return
        pending = self._pending.get(token)
        if pending is None or (isinstance(purchase, Purchase) and purchase != pending) or not values.keys() <= set(pending.root_ids):
            raise ValueError('Only labels for roots in the outstanding purchase can be admitted')
        if original_id in self._fees and self._fees[original_id] != paid_cpu:
            raise ValueError('A shared original must retain the same total charge at every purchase')
        genome = pending.genome
        labels = self._labels[genome]
        if any(root in labels and labels[root] != value for root, value in values.items()):
            raise ValueError('A measured immutable layout and root cannot acquire a different terminal risk')
        if feasible is not None and genome in self._service and feasible != self._service[genome]:
            raise ValueError('A fixed layout needs a consistent same-service verdict')
        self._fees[original_id] = paid_cpu
        self._receipts[token] = record
        labels.update(values)
        self._inferred_roots.difference_update((genome, root) for root in values)
        self._deployment_cache.pop(genome, None)
        if feasible is not None:
            self._service[genome] = feasible
        del self._pending[token]

    def seed_label_transfer(self, source_genome, target_genome, root, value, *,
                            certificate_id, reason='unaffected_domain'):
        """Admit one zero-cost label under an externally checked certificate.

        This operation never creates a paid receipt.  It is deliberately
        separate from :meth:`tell` so a structural-equivalence proof cannot
        be confused with a native observation in the accounting ledger.
        Callers must provide a non-empty certificate identifier and a source
        label that is already present in this search state.
        """
        source = tuple(source_genome)
        target = tuple(target_genome)
        root = str(root)
        if not isinstance(certificate_id, str) or not certificate_id:
            raise ValueError('A zero-cost transfer needs a certificate identifier')
        if root not in self._labels.get(source, {}) or (source, root) in self._inferred_roots:
            raise ValueError('A transfer source must have a settled native label')
        value = float(value)
        if not isfinite(value) or not 0. <= value <= 1.:
            raise ValueError('Transferred labels must be bounded risks')
        physical = self._geometry(target)
        if value != self._labels[source][root]:
            raise ValueError('A transferred label must match its native donor')
        if (physical is None or physical[0] != target or root not in self.roots or
                self.circuit.candidates and target not in self._catalog or
                self.candidate_roots is not None and root not in self.candidate_roots.get(target, ())):
            raise ValueError('The transfer target and root must belong to the declared catalogue')
        existing = self._labels[target].get(root)
        if existing is not None:
            if not isclose(existing, value, rel_tol=0., abs_tol=1e-12):
                raise ValueError('A transferred label conflicts with an existing label')
            return False
        self._areas.setdefault(target, physical[1])
        self._labels[target][root] = value
        self._inferred_roots.add((target, root))
        self._deployment_cache.pop(target, None)
        self._transfer_history.append(dict(source=list(source), target=list(target), root=root,
                                           value=value, certificate_id=certificate_id, reason=reason))
        return True

    def transfer_history(self):
        return json.loads(json.dumps(self._transfer_history, allow_nan=False))

    def _deployment(self, genome):
        if self._service.get(genome) is not True:
            return None
        if self.circuit.min_deployment_roots:
            observed = len(self._labels.get(genome, {}))
            if observed < self.circuit.min_deployment_roots:
                return None
        if genome not in self._deployment_cache:
            self._deployment_cache[genome] = self._summarize_deployment(genome)
        return self._deployment_cache[genome]

    def _summarize_deployment(self, genome):
        if self.circuit.physical_root_shrinkage:
            return self._summarize_physical_deployment(genome)
        source_values = {source: [] for source in self.weights}
        for root, risk in self._labels[genome].items():
            source_values[self.roots[root].source].append(risk)
        if any(len(values) < self.circuit.min_source_roots for values in source_values.values()):
            return None
        summary = {source: _summary(values) for source, values in source_values.items()}
        mean = fsum(self.weights[s]*row['mean'] for s, row in summary.items())
        half = 1.96*sqrt(fsum((self.weights[s]*row['stderr'])**2 for s, row in summary.items()))
        return dict(genome=list(genome), area_ge=self._areas[genome], estimated_risk=mean,
                    risk_interval=[max(0., mean-half), min(1., mean+half)],
                    roots_by_source={s: row['roots'] for s, row in summary.items()},
                    risk_by_source={s: row['mean'] for s, row in summary.items()})

    def _summarize_physical_deployment(self, genome):
        return physical_deployment_summary(genome, self._areas[genome], self._labels[genome],
                                           self.roots, self.weights, min_source_roots=self.circuit.min_source_roots)

    def recommend(self, *, area_caps=None, conservative=None):
        """Return the observed frontier and a risk-safe deployment choice.

        Incident-conditioned acquisition deliberately starts with sparse,
        physically separated groups.  Selecting a layout by its point estimate
        at that stage rewards an unverified lucky root block.  Its registered
        confidence interval is already computed from the observed source
        roots, so the incident route uses the upper risk bound until a layout
        has enough evidence to displace a safer incumbent.  Other routes keep
        the historical point-estimate frontier unless they request the same
        conservative rule explicitly.
        """
        if conservative is None:
            conservative = self.circuit.physical_root_shrinkage or (
                self.mode.incident_conditioned and not self._incident_runtime_pooled
            )
        if type(conservative) is not bool:
            raise ValueError('The recommendation rule must be a boolean')
        caps = tuple(area_caps) if area_caps is not None else (self.circuit.area_caps or (self.cap,))
        self._check_caps(caps)
        rows = [row for genome in self._labels if (row := self._deployment(genome)) is not None]
        # Callers own mutable recommendations. Keep the cached summaries private.
        frontier = deployment_frontier(rows, conservative=conservative)
        return dict(frontier=frontier, recommendations=[dict(area_cap_ge=cap, layout=next(
            (row for row in reversed(frontier) if row['area_ge'] <= cap), None)) for cap in caps],
            paid_cpu=self.paid_cpu, reserved_cpu=self.reserved_cpu)
