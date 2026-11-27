# Evo-RBKG reproduction

This repository provides a CI-verified standalone Streaming FIR run and an offline replay of the principal four-benchmark results for **Task-Level Protection and Recovery Co-Design for Digital Circuits via Evolutionary Root-Block Knowledge Gradient**. Standalone runners for RV32I, Conv-Pool and Rocket DMA W4 and the complete frozen evaluation inputs are planned for the final version at this address.

## Run

On x86-64 Ubuntu or Debian, including WSL:

```sh
bash setup.sh
.venv/bin/python -B reproduce.py smoke
.venv/bin/python -B verify_paper.py
.venv/bin/python -B reproduce.py quick --fte 300 --jobs 2
```

Setup installs missing build tools with APT and the pinned Python dependencies and Verilator wheel into `.venv`. It uses `sudo` for missing system tools. On another architecture, `quick` can use a system-installed Verilator 5.026 or newer. The two offline commands also run on Windows after `python -m pip install -r requirements.txt`.

| Check | What it runs |
| --- | --- |
| `reproduce.py smoke` | Replays the first three registered FIR optimizer decisions from archived labels. Checks selected layouts, blocks, nominations, geometry and cumulative FTE. |
| `reproduce.py quick` | Builds the FIR receiver once, verifies the archived outcomes of 160 faulted evaluations from two root blocks, then runs a fresh short decision loop under an exact FTE cap. |
| `verify_paper.py` | Recomputes the full four-circuit main analysis from compact archived measurements. Optional `--json results.json` exports calculated endpoint and comparison rows. |

## What the offline paper check reproduces

- **Figs. 2–3 and Table II** — integrates 384 matched NHV budget trajectories, reconstructs terminal NHV and area-cap risk, calculates paired main effects, and derives 50% empirical attainment at each cap.
- **Table III** — reconstructs the pooled nondominated reference fronts, recomputes 384 terminal IGD+ scores and all 24 paired comparisons, and recalibrates their 4,096-draw joint bootstrap intervals.
- **Fig. 4 and Table IV** — recomputes 72 paired ablation effects from frozen per-seed endpoints on their respective evaluation panels.
- **Fig. 5 and sensitivity tables** — recomputes 480 paired point effects, identifies 12 structural invariants, and checks the significance and practical-gain counts against the frozen intervals.
- **Hardware checks in Secs. VI-C and VI-D** — checks reported matched-test arithmetic, recomputes FIR mapped-area and power changes, verifies the archived MAC RTL/SDF labels, and recomputes the Rocket deadline bound and measured execution savings.

The compressed evidence contains budget trajectories, terminal fronts, paired endpoints, and the measurements used by these checks. Main, ablation and sensitivity intervals use the frozen whole-root calibration. IGD+ intervals are recomputed because their inference holds blind scores fixed and resamples optimizer seeds. The offline command uses saved outputs for ablations, sensitivity and hardware summaries. Local RTL execution is provided for FIR.

## Budget and provenance

One full-task equivalent (FTE) charges each simulation segment by its own reference horizon. FIR uses 640- and 1,024-cycle references. A full block charges 80 faulted and 20 clean executions, for exactly 100 FTE. The quick runner admits whole blocks under the requested cap. Warm CPU reports active native simulation, Python harness and optimizer time on the current host. Build wall time is reported separately.

`fir.json` records the frozen optimizer revision and source fingerprints for the portable FIR receiver. The native check rebuilds the receiver and matches archived labels and cycles. `paper-evidence.json.gz` records the selected paper evidence digest and retains the nominal FTE and warm-CPU coordinates and pairings.
