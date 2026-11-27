"""Bounded proposals in a legal categorical design space without a catalog."""

from bisect import bisect_right
from collections import deque
from math import fsum, isfinite
from random import Random

from .optimizers import _Space


def propose_layouts(options, geometry, parents, occupied, *, count, seed, area_edges,
                    groups=(), radius=3, gene_families=(), active_loci=None,
                    group_weights=None):
    """Return canonical genomes and areas from a mixture of sparse and global moves.

    Geometry is the adapter's deterministic admission and canonicalization rule.
    Area edges come from the registered constraints. Parents are chosen from
    settled evidence by the caller. The pool can be rebuilt after each receipt.
    """
    if type(count) is not int or count < 0 or type(radius) is not int or radius < 1:
        raise ValueError('Proposal count and radius must be nonnegative and positive integers, respectively')
    edges = tuple(area_edges)
    if any(not isfinite(edge) or edge <= 0 for edge in edges) or tuple(sorted(set(edges))) != edges:
        raise ValueError('Area band edges must increase strictly and be positive')
    rng = Random(seed)
    space = _Space(options, (), 'fine', rng, gene_families=gene_families, active_loci=active_loci)
    parents = tuple(space.validate(parent) for parent in parents)
    groups = tuple(dict.fromkeys(tuple(sorted(set(group))) for group in groups))
    if any(not group or any(type(i) is not int or not 0 <= i < len(space.options) for i in group) for group in groups):
        raise ValueError('Physical mutation groups must name valid genome axes')
    if group_weights is None:
        weights = None
    else:
        weights = tuple(float(weight) for weight in group_weights)
        if (len(weights) != len(groups) or not weights or any(not isfinite(weight) or weight <= 0 for weight in weights)
                or not fsum(weights) > 0):
            raise ValueError('Group weights must be positive finite values in group order')
    if not count:
        return {}

    if active_loci is None:
        mutable = set(space.mutable)
        groups = tuple(tuple(axis for axis in group if axis in mutable) for group in groups)
    bands = [deque() for _ in range(len(edges)+1)]
    quota = (count+len(bands)-1)//len(bands)
    occupied = set(occupied)
    tried = set()
    for attempt in range(max(64, 12*count)):
        route = attempt % 4
        if not parents or route == 0:
            raw = space.random(0.)
        else:
            parent = rng.choice(parents)
            active = space.active_mutable(parent)
            raw = list(parent)
            if groups and route == 1:
                # Keep every registered physical domain reachable while using
                # its frozen exposure mass to focus part of the mutation budget.
                # The alternating uniform draw prevents a high-mass domain from
                # starving smaller domains in a finite proposal pool.
                if weights is None or attempt // 4 % 2 == 0:
                    selected_group = groups[(attempt // 4) % len(groups)]
                else:
                    selected_group = rng.choices(groups, weights=weights, k=1)[0]
                active = (selected_group if active_loci is None else
                          tuple(axis for axis in selected_group if axis in active))
                changes = len(active)
            else:
                changes = rng.randint(1, min(radius, len(active))) if active else 0
            for axis in rng.sample(active, changes):
                space._change_gene(raw, axis)
            raw = tuple(raw)
        if raw in tried:
            continue
        tried.add(raw)
        admitted = geometry(raw)
        if admitted is None:
            continue
        genome, area = admitted
        genome = space.validate(genome)
        if not isfinite(area) or area < 0:
            raise ValueError('Admitted layouts require a finite nonnegative area')
        if genome in occupied:
            continue
        occupied.add(genome)
        band = bands[bisect_right(edges, area)]
        if len(band) < count:
            band.append((genome, area))
        if all(len(band) >= quota for band in bands):
            break

    bands = [band for band in bands if band]
    rng.shuffle(bands)
    selected = {}
    while len(selected) < count and bands:
        for band in bands:
            genome, area = band.popleft()
            selected[genome] = area
            if len(selected) == count:
                break
        bands = [band for band in bands if band]
    return selected
