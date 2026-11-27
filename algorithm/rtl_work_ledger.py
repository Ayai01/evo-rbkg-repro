"""Track independent execution and search-cost axes for one open-search route."""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from math import fsum, isfinite


class SearchAction(StrEnum):
    L0_SCREEN = 'L0_SCREEN'
    L1_SHORT_BATCH = 'L1_SHORT_BATCH'
    L2_NEW_ANCHOR = 'L2_NEW_ANCHOR'
    L2_REFINE_ROOTS = 'L2_REFINE_ROOTS'


def full_task_equivalents(cycles, service_references):
    """Convert actual per-service native cycle totals to an exact FTE fraction."""
    if not cycles.keys() <= service_references.keys() or any(
            type(value) is not int or value <= 0 for value in service_references.values()):
        raise ValueError('Every charged cycle requires a frozen, positive service reference')
    if any(type(work) is not int or work < 0 for work in cycles.values()):
        raise ValueError('Native work is measured in nonnegative integer cycles')
    return sum((Fraction(work, service_references[service]) for service, work in cycles.items()), Fraction(0))


@dataclass(frozen=True)
class Execution:
    original: str
    segment: str
    kind: str
    executed_cycles: int
    simulation_cpu_s: float

    def __post_init__(self):
        if (not self.original or not self.segment or self.kind not in {'clean', 'short', 'fault'}
                or type(self.executed_cycles) is not int or self.executed_cycles < 0
                or not isfinite(self.simulation_cpu_s) or self.simulation_cpu_s < 0):
            raise ValueError('Invalid native execution receipt')


class RtlWorkLedger:
    """Deduplicate references to physical executions, preserving actual repeated work."""

    def __init__(self, reference_cycles):
        if type(reference_cycles) is not int or reference_cycles <= 0:
            raise ValueError('Freeze a positive reference cycle count per service contract')
        self.reference_cycles = reference_cycles
        self._executions = {}
        self._actions = {}

    def record(self, action_id, kind, executions=(), *, decision_cpu_s=0., setup_cpu_s=0.,
               failure_cpu_s=0., unmetered_failure=False):
        kind = SearchAction(kind)
        executions = tuple(executions)
        if (not action_id or any(not isfinite(v) or v < 0 for v in
                (decision_cpu_s, setup_cpu_s, failure_cpu_s))
                or kind is SearchAction.L0_SCREEN and executions):
            raise ValueError('Invalid search action or cost')
        event = (kind, executions, decision_cpu_s, setup_cpu_s, failure_cpu_s, unmetered_failure)
        if action_id in self._actions:
            if self._actions[action_id] != event:
                raise ValueError('A settled action cannot change')
            return
        pending = {}
        for row in executions:
            key = (row.original, row.segment)
            previous = pending.get(key, self._executions.get(key))
            if previous is not None and previous != row:
                raise ValueError('Shared execution references must preserve their original counters')
            pending[key] = row
        self._executions.update(pending)
        self._actions[action_id] = event

    def summary(self):
        groups = {k: 0 for k in ('clean', 'short', 'fault')}
        for row in self._executions.values():
            groups[row.kind] += row.executed_cycles
        cycles = sum(groups.values())
        sim = fsum(row.simulation_cpu_s for row in self._executions.values())
        actions = tuple(self._actions.values())
        decisions = fsum(row[2] for row in actions)
        unknown_failures = sum(bool(row[5]) for row in actions)
        return dict(reference_cycles=self.reference_cycles, executed_cycles=cycles, cycles_by_kind=groups,
                    fte=cycles / self.reference_cycles, simulation_CPU_s=sim, decision_CPU_s=decisions,
                    search_CPU_s=sim + decisions, setup_CPU_s=fsum(row[3] for row in actions),
                    failure_CPU_s=fsum(row[4] for row in actions), unmetered_failures=unknown_failures,
                    native_executions=len(self._executions), actions=len(actions))
