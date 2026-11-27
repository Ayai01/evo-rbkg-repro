"""Price a nomination-ready bundle and dispatch its first native block."""

from collections import Counter
from math import fsum


def _minimum_completion_plan(candidates, need, *, total_needed=0, limit=float('inf')):
    """Choose the least-cost native block set that completes a deployment.

    A greedy source-gain ratio can buy a block that covers a common source
    cheaply and leave an uncovered source for an expensive tail block.  The
    dynamic programme keeps the source deficits capped at the registered
    deployment requirement and chooses the minimum remaining median cost,
    with reserve cost and block count as deterministic tie breakers.
    """
    source_order = tuple(need)
    target = tuple(need[source] for source in source_order)
    track_total = total_needed > sum(target)
    if track_total:
        target += (total_needed,)
    items = []
    for offer, support in candidates.values():
        gain = tuple(support.get(source, 0) for source in source_order)
        if track_total:
            gain += (len(offer[2]),)
        if any(gain):
            items.append((offer, gain))
    zero = (0,) * len(target)
    states = {zero: (0.0, 0.0, ())}
    for offer, gain in items:
        updated = dict(states)
        for covered, (median, reserve, plan) in states.items():
            next_covered = tuple(min(cap, n + step) for cap, n, step in zip(target, covered, gain))
            if next_covered == covered:
                continue
            candidate = (median + offer[3].median_cpu,
                         reserve + offer[3].reserve_cpu,
                         (*plan, offer))
            previous = updated.get(next_covered)
            if previous is None or candidate[:2] < previous[:2] or (
                    candidate[:2] == previous[:2] and len(candidate[2]) < len(previous[2])):
                updated[next_covered] = candidate
        states = updated
    result = states.get(target)
    if result is None or result[1] <= limit:
        return None if result is None else result[2]

    # The cheapest median plan may reserve more CPU than the remaining budget.
    # Keep nondominated (median, reserve) alternatives only when that cap binds.
    frontiers = {zero: [(0., 0., ())]}
    for offer, gain in items:
        updated = {key: list(values) for key, values in frontiers.items()}
        for covered, alternatives in frontiers.items():
            new = tuple(min(cap, n + step) for cap, n, step in zip(target, covered, gain))
            if new == covered:
                continue
            for median, reserve, plan in alternatives:
                fee = median + offer[3].median_cpu
                reserved = reserve + offer[3].reserve_cpu
                if reserved <= limit:
                    updated.setdefault(new, []).append((fee, reserved, (*plan, offer)))
        for key, plans in updated.items():
            kept = []
            for plan in sorted(plans, key=lambda p: (p[0], p[1], len(p[2]))):
                if not kept or plan[1] < kept[-1][1]:
                    kept.append(plan)
            updated[key] = kept
        frontiers = updated
    feasible = frontiers.get(target)
    return feasible[0][2] if feasible else None


def completion_candidates(search, layouts, limit, held, *, all_blocks, blocks_by_layout=None, all_starts=False):
    """Yield (layout, priced evidence, first native block) without reading unseen losses."""
    from .cross_circuit_search import Quote

    for genome in layouts:
        if genome in held or search._service.get(genome) is False:
            continue
        blocks = search.blocks if blocks_by_layout is None else blocks_by_layout.get(genome, ())
        if not blocks:
            continue
        labels = search._labels.get(genome, ())
        counts = Counter(search.roots[r].source for r in labels)
        total_needed = max(0, search.circuit.min_deployment_roots - len(labels))
        need = {s: search.circuit.min_source_roots - counts[s] for s in search.weights
                if counts[s] < search.circuit.min_source_roots}
        candidates = {}
        for block in blocks:
            offer = search._offers([(genome, block)], limit, held).get(genome)
            if offer is None:
                continue
            if not need and not total_needed:
                yield genome, offer, offer
                if not all_blocks:
                    break
                continue
            candidates[block] = (offer, Counter(search.roots[r].source for r in offer[2]))
        if not need and not total_needed:
            continue
        if all_starts:
            plans = []
            for block, (first, support) in candidates.items():
                remaining = {s: n-support[s] for s,n in need.items() if n > support[s]}
                tail = _minimum_completion_plan(
                    {b: row for b,row in candidates.items() if b != block}, remaining,
                    total_needed=max(0,total_needed-len(first[2])), limit=limit-first[3].reserve_cpu)
                if tail is not None:
                    plans.append((first, *tail))
        else:
            plan = _minimum_completion_plan(candidates, need, total_needed=total_needed, limit=limit)
            plans = () if plan is None else (plan,)
        for plan in plans:
            price = Quote(fsum(o[3].median_cpu for o in plan), fsum(o[3].reserve_cpu for o in plan))
            if price.reserve_cpu <= limit:
                members = tuple(r for offer in plan for r in offer[2])
                sources = tuple(dict.fromkeys(search.roots[r].source for r in members))
                yield genome, (plan[0][0], sources, members, price), plan[0]


def completion_offers(search, layouts, limit, held, *, blocks_by_layout=None):
    valued, first = {}, {}
    for genome, bundle, native in completion_candidates(search, layouts, limit, held, all_blocks=False,
                                                         blocks_by_layout=blocks_by_layout):
        valued[genome], first[genome] = bundle, native
    return valued, first
