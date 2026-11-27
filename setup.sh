#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

packages=()
for tool in python3 make g++ perl; do
    command -v "$tool" >/dev/null || packages+=("$tool")
done
if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
    packages+=(python3-venv)
fi
if (("${#packages[@]}")); then
    if ! command -v apt-get >/dev/null; then
        printf 'Install these tools with your package manager: %s\n' "${packages[*]}" >&2
        exit 1
    fi
    apt=(apt-get)
    if ((EUID != 0)); then
        apt=(sudo apt-get)
    fi
    "${apt[@]}" update -q
    "${apt[@]}" install -y --no-install-recommends "${packages[@]}"
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    printf 'Python 3.11 or newer is required.\n' >&2
    exit 1
fi
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
if [[ "$(uname -m)" == x86_64 ]]; then
    .venv/bin/python -m pip install --disable-pip-version-check --only-binary=:all: verilator-dspsim==5.54.0
    .venv/bin/verilator-dspsim --version
elif command -v verilator >/dev/null; then
    verilator --version
else
    printf 'The smoke check is ready. Provide Verilator 5.026 or newer on this architecture for quick mode.\n'
fi
