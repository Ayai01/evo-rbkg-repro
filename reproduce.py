'''Run the frozen FIR evidence prefix and a bounded local Evo-RBKG search.'''

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
from queue import Queue
import subprocess
import shutil
import sys
from tempfile import TemporaryDirectory
from time import perf_counter, process_time

import psutil

from algorithm.adaptive_rtl_search import ForecastCost
from algorithm.cross_circuit_search import Circuit, Quote, Root
from algorithm.physical_locus_residual import PhysicalLocusResidual
from algorithm.rtl_work_ledger import full_task_equivalents
from arh.four_circuit_l0l2_policy import BASE_BLOCK, QUOTE_CYCLE_UNIT
from arh.terminal_cap_development import CapTerminalSearch
from fir_model import VARIABLES, design, geometry, observe, packed_hardware, payload_position
from framework.text_transport import TextTransport


ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expected_fte(cases, ids):
    clean = {(cases[k]["seed"], cases[k]["period"], cases[k]["deadline"]) for k in ids}
    return len(ids) + len(clean)


def actual_fte(segments):
    cycles = Counter()
    for spent, reference in segments:
        cycles[reference] += spent
    return full_task_equivalents(cycles, {reference: reference for reference in cycles})


def design_area(spec, genome):
    if len(genome) != len(spec["options"]) or any(g not in o for g, o in zip(genome, spec["options"], strict=True)):
        return None
    legal = geometry(genome)
    return legal if legal and legal[1] <= spec["max_area"] else None


def optimizer(data):
    spec = data["spec"]
    metadata = spec["roots"]
    origin, cap, reference = (spec[k] for k in ("area_origin", "cap", "reference"))

    def quote(_genome, _block, ids):
        return Quote(len(ids) * spec["quote_cpu_per_condition"],
                     len(ids) * spec["reserve_cpu_per_condition"])

    circuit = Circuit(
        tuple(map(tuple, spec["options"])), lambda g: design_area(spec, g),
        {key: Root(row["source"], key, row["cluster"]) for key, row in metadata.items()},
        {name: tuple(ids) for name, ids in data["blocks"].items()},
        spec["source_weights"], spec["source_axes"], quote,
        reference=reference, max_area=spec["max_area"], recovery_axes=tuple(spec["recovery_axes"]),
        area_origin=origin, area_caps=(cap,),
        area_edges=(cap, cap + (reference - cap) / 3, cap + 2 * (reference - cap) / 3),
        opening=(tuple(spec["opening"]),), opening_blocks=(BASE_BLOCK,),
        proposal_count=data["shortlist"], min_deployment_roots=len(data["blocks"][BASE_BLOCK]),
        physical_root_shrinkage=True,
    )
    search = CapTerminalSearch(
        circuit, arm="arh_clean", method="arh_dse", root_metadata=metadata,
        settings={"seed": data["seed"], "cpu_budget": data["paper_CPU_s"]},
        max_fte=data["paper_FTE"], proposal_pool=data["proposal_pool"], shortlist=data["shortlist"],
        root_evidence_policy=data["evidence_policy"], fast_proposals=True,
        clean=True, span=True, outside_span_floor=data["outside_span_floor"],
    )
    search.physical_locus_residual = PhysicalLocusResidual(metadata, recovery_axes=tuple(spec["recovery_axes"]))
    return search


def select(engine, data, *, has_observations):
    if has_observations:
        engine.screen()

    def forecast(_genome, roots):
        quote = data["spec"]["quote_fte_per_condition"] * len(roots)
        return ForecastCost(max(1, round(quote * QUOTE_CYCLE_UNIT)),
                            data["spec"]["quote_cpu_per_condition"] * len(roots))

    engine._active_cost = forecast
    try:
        choices = engine.ask(1)
    finally:
        engine._active_cost = None
    return choices[0] if choices else None


def settle(engine, purchase, receipt, idx, paid_cpu):
    previously_measured = bool(engine._labels.get(purchase.genome))
    engine.tell(purchase, {k: int(v) for k, v in receipt["outcomes"].items()},
                original_id=f"local/{idx}", paid_cpu=paid_cpu, feasible=receipt["feasible"])
    engine.refinements += int(previously_measured)
    return engine.recommend()


def recommendation(nomination):
    return nomination["recommendations"][0]["layout"]


def smoke(data):
    engine = optimizer(data)
    fte = Fraction()
    for i, expected in enumerate(data["prefix"]):
        offer = select(engine, data, has_observations=bool(i))
        require(offer is not None, f"Search made no offer at step {i + 1}")
        require(list(offer.genome) == expected["genome"] and offer.block == expected["block"],
                f"Frozen proposal differs at step {i + 1}: {offer.genome} {offer.block}")
        require(offer.reason == expected["decision"], f"Decision reason differs at step {i + 1}")
        require(set(offer.root_ids) == set(expected["outcomes"]), f"Root coverage differs at step {i + 1}")
        require(engine.circuit.geometry(offer.genome)[1] == expected["area_ge"], "FIR area proxy differs")
        fte += actual_fte(expected["segments"])
        require(fte == 100 * (i + 1), "FTE does not match the frozen budget trace")
        nominal_cpu = data["spec"]["quote_cpu_per_condition"] * len(offer.root_ids)
        nominee = recommendation(settle(engine, offer, expected, i + 1, nominal_cpu))
        require(list(nominee["genome"]) == expected["nominee"], f"Nominee differs at step {i + 1}")
    return {"check": "frozen optimizer prefix", "seed": data["seed"], "blocks": len(data["prefix"]),
            "FTE": str(fte), "fixed_layout_area_GE": nominee["area_ge"]}


def build(data, work, jobs):
    source = work / "src"
    for relative in ("projects/fir_stream/rtl.sv", "projects/fir_stream/sim.cpp",
                     "framework/rtl/ClockedInput.h", "framework/rtl/JointCapture.h"):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    sources = [source / "projects/fir_stream" / name for name in ("rtl.sv", "sim.cpp")]
    objects = work / "obj"
    for path, key in zip(sources, ("rtl_sha256", "harness_sha256"), strict=True):
        require(sha256(path.read_bytes()).hexdigest() == data[key], f"Source fingerprint mismatch: {path.name}")
    installed = Path(sys.executable).with_name("verilator-dspsim")
    executable = str(installed) if installed.is_file() else shutil.which("verilator")
    require(executable, "Run bash setup.sh or install Verilator 5.026 or newer")
    subprocess.run(
        [executable, "--cc", str(sources[0]), "--exe", str(sources[1]),
         "--top-module", "FirStream", "--Mdir", str(objects), "-o", "fir",
         "--output-split", "2000", "--output-split-cfuncs", "400", "-Wno-fatal",
         "-CFLAGS", "-std=c++17 -DVL_TIME_CONTEXT -fwrapv -fno-strict-aliasing",
         *[f"-G{k}={v}" for k, v in sorted(data["parameters"].items())]],
        check=True, stdout=subprocess.DEVNULL,
    )
    subprocess.run(["make", "-s", "-C", str(objects), "-f", "VFirStream.mk", "-j", str(jobs),
                    "VM_PARALLEL_BUILDS=1", "OPT_FAST=-O1", "OPT_SLOW=-O0", "OPT_GLOBAL=-O0"], check=True)
    return objects / "fir"


class Receiver:
    def __init__(self, binary):
        self.binary = binary

    def __enter__(self):
        self.process = subprocess.Popen([str(self.binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, encoding="utf-8", bufsize=1)
        self.pid = psutil.Process(self.process.pid)
        self.transport = TextTransport(self.process, Queue(maxsize=8), timeout=30)
        self.transport.start_readers(self.read, self.read_errors)
        return self

    def __exit__(self, *_):
        self.transport.close()

    def read(self):
        for row in self.process.stdout:
            self.transport.emit(row)
        self.transport.emit(None)

    def read_errors(self):
        for line in self.process.stderr:
            print(line.rstrip(), file=sys.stderr)

    def cpu(self):
        times = self.pid.cpu_times()
        return times.user + times.system

    def evaluate(self, data, genome, block):
        spec = data["spec"]
        legal = design_area(spec, genome)
        require(legal is not None and legal[0] == tuple(genome), "Require a legal canonical FIR layout")
        genes, banks = packed_hardware(genome)
        layout = design(genes)
        hardware = [layout[k] if isinstance(layout[k], int) else choices.index(layout[k])
                    for k, choices in VARIABLES]
        hardware[9] = genome[18]
        starts, outcomes, segments = {}, {}, []

        def run(case, config, events):
            msg = ["T", case["seed"], *hardware, *layout["schemes"], 64, case["period"], 256, len(events), 1, banks]
            msg.extend(value for event in events for value in event)
            result = json.loads(self.transport.exchange(",".join(map(str, msg))))
            require("error" not in result, f"FIR receiver error: {result.get('error')}")
            require(result.get("seed") == case["seed"] and result.get("input_frames") == 64
                    and result.get("physical_xor_checks") == len(events), "FIR event or workload mismatch")
            reference = 64 * case["period"] + 256
            require(result.get("cycles") == reference, "Executed FIR horizon differs from its fixed task reference")
            segments.append([result["cycles"], reference])
            return observe(result, genes, config, case["seed"])

        child_start, host_start = self.cpu(), process_time()
        for key in data["blocks"][block]:
            case = data["cases"][key]
            seed, period, deadline = (case[k] for k in ("seed", "period", "deadline"))
            config = {"stream_frames": 64, "input_period_cycles": period, "drain_cycles": 256,
                      "sample_deadline_cycles": deadline, "fault_error_limit": .025}
            state = (seed, period, deadline)
            if state not in starts:
                clean = run(case, config, ())
                starts[state] = not clean["failed"] and not clean["incorrect_output_frames"] and clean["sustained_rate"] >= 1.
            events = [[case["tick"], domain, word, bit if axis == len(genome) else payload_position(bit, genome[axis])]
                      for domain, word, axis, bit in case["sites"]]
            fault = run(case, config, events)
            outcomes[key] = bool(not starts[state] or fault["failed"])
        cpu = max(0., self.cpu() - child_start) + process_time() - host_start
        return {"area_ge": legal[1], "outcomes": outcomes, "feasible": all(starts.values()), "segments": segments,
                "cycles": sum(c for c, _ in segments), "simulation_CPU_s": cpu}


def rtl_check(data, rtl):
    frozen = (data["prefix"][0], data["refine"])
    fte = Fraction()
    cases = 0
    for row in frozen:
        received = rtl.evaluate(data, tuple(row["genome"]), row["block"])
        for key in ("area_ge", "feasible", "segments", "cycles", "outcomes"):
            require(received[key] == row[key], f"{row['block']}: native {key} differs from archived witness")
        fte += actual_fte(received["segments"])
        cases += len(received["outcomes"])
    return {"check": "frozen FIR native witnesses", "fault_cases": cases, "FTE": str(fte)}


def live_search(data, rtl, budget):
    engine = optimizer(data)
    blocks = data["blocks"]
    block_work = {b: expected_fte(data["cases"], ids) for b, ids in blocks.items()}
    require(len(set(block_work.values())) == 1, "This quick runner expects equal whole-block FTE")
    quantum = block_work[BASE_BLOCK]
    require(budget >= quantum, f"Use a budget of at least {quantum} FTE")
    results, fte = [], Fraction()
    decision_cpu = simulation_cpu = 0.
    while fte + quantum <= budget:
        started = process_time()
        offer = select(engine, data, has_observations=bool(results))
        decision_cpu += process_time() - started
        if offer is None:
            break
        row = rtl.evaluate(data, offer.genome, offer.block)
        simulation_cpu += row["simulation_CPU_s"]
        used = actual_fte(row["segments"])
        require(used == quantum and fte + used <= budget, "Whole-block FTE accounting disagrees")
        fte += used
        start = process_time()
        result = recommendation(settle(engine, offer, row, len(results) + 1, row["simulation_CPU_s"]))
        decision_cpu += process_time() - start
        results.append({"block": offer.block, "genome": offer.genome, "area_GE": row["area_ge"],
                        "failures": sum(row["outcomes"].values()), "cases": len(row["outcomes"]),
                        "FTE": str(fte), "nominee": None if result is None else result["genome"]})
    require(results, "The quick search did not complete its opening block")
    return {"check": "local FIR search", "budget_FTE": budget, "used_FTE": str(fte), "actions": results,
            "simulation_CPU_s": simulation_cpu, "decision_CPU_s": decision_cpu,
            "warm_CPU_s": fsum((simulation_cpu, decision_cpu))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "quick"))
    parser.add_argument("--fte", type=int, default=300, help="Exact cycle budget for quick mode (default: 300)")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel C++ build jobs (default: 2)")
    args = parser.parse_args()
    if args.fte < 1 or args.jobs < 1:
        parser.error("--fte and --jobs must be positive")
    data = json.loads((ROOT / "fir.json").read_bytes())
    if args.mode == "smoke":
        print(json.dumps(smoke(data), indent=2))
        return
    with TemporaryDirectory(prefix="evo-rbkg-") as path:
        started = perf_counter()
        binary = build(data, Path(path), args.jobs)
        build_seconds = perf_counter() - started
        with Receiver(binary) as rtl:
            witnesses = rtl_check(data, rtl)
            result = live_search(data, rtl, args.fte)
        result["native_witnesses"] = witnesses
        result["build_wall_s"] = build_seconds
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
