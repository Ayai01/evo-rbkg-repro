# Evo-RBKG minimal reproduction

This repository reproduces a bounded Streaming FIR run for **Task-Level Protection and Recovery Co-Design for Digital Circuits via Evolutionary Root-Block Knowledge Gradient**. It ships the optimizer path, two registered root blocks, FIR RTL and its fault-injection harness, and a small set of archived correctness witnesses.

## Run

On x86-64 Ubuntu or Debian, including WSL, install the environment and run both checks:

```sh
bash setup.sh
.venv/bin/python -B reproduce.py smoke
.venv/bin/python -B reproduce.py quick --fte 300 --jobs 2
```

The setup script installs missing build tools with APT and the pinned Python dependencies and Verilator wheel into `.venv`. It uses `sudo` only for missing system tools. On another architecture, `quick` can use a system-installed Verilator 5.026 or newer. For the Python-only smoke check on Windows, run `python -m pip install -r requirements.txt` followed by `python -B reproduce.py smoke`.

- `smoke` replays the first three registered optimizer decisions for seed 11. It checks chosen layouts, block IDs, terminal nominations, analytic area and exact cumulative FTE using archived labels and segment counts.
- `quick` builds the FIR receiver once in a temporary directory. It first checks two archived hardware witnesses, one from each root block. Together these cover 160 faulted evaluations at 200 FTE including clean checks. It then starts a fresh three-block search and reports measured FTE, selected layouts, terminal failures and active CPU time.

## Budget and scope

One full-task equivalent (FTE) charges each executed simulation segment by its own reference horizon. FIR uses both 640- and 1,024-cycle references. A full block charges 80 faulted and 20 clean executions, for exactly 100 FTE. The local search admits whole blocks under the requested FTE cap. For example, a 250-FTE cap can admit two blocks. Build wall time is reported separately. Warm CPU is measured native simulation and Python harness CPU plus optimizer CPU on the current host. FTE and archived task labels are the portable checks.

The optimizer and geometry come from the frozen code revision recorded in `fir.json`. The portable RTL is rebuilt from the released carrier sources. The archived C++ harness is byte-matched, and the terminal witnesses check the portable RTL against the recorded run. This package covers the FIR method and budget checks. The published multi-circuit tables use the full frozen run and blind panels.
