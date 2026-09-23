#!/usr/bin/env bash
# Install CPU PyTorch + amd-pace and run PACE GEMM (Linear / TPP / AOCL).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV="${PACE_VENV:-$PWD/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "=== creating $VENV ==="
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "=== python $(python --version) @ $(which python) ==="

if ! python -c "import torch" 2>/dev/null; then
  echo "=== installing CPU torch ==="
  pip install --upgrade pip
  pip install --extra-index-url https://download.pytorch.org/whl/cpu "torch==2.13.0+cpu"
fi

if ! python -c "import pace" 2>/dev/null; then
  echo "=== installing amd-pace ==="
  pip install amd-pace
fi

# EPYC has no AMX; force libXSMM onto the AVX-512 (CPX) path if it probes AMX.
export LIBXSMM_TARGET="${LIBXSMM_TARGET:-cpx}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PACE_LOG_LEVEL="${PACE_LOG_LEVEL:-info}"
# Dump TPP layers 1–3 (pack / Y+tiles / JIT). Set PACE_INSPECT=0 to skip.
export PACE_INSPECT="${PACE_INSPECT:-1}"
export PACE_KERNEL_DUMP="${PACE_KERNEL_DUMP:-$PWD/kernel_dump}"
# Negative = write raw JIT binaries; positive = atexit stats table only.
export LIBXSMM_VERBOSE="${LIBXSMM_VERBOSE:--1}"
mkdir -p "$PACE_KERNEL_DUMP"

echo "=== running GEMM (inspect=$PACE_INSPECT dump=$PACE_KERNEL_DUMP) ==="
python run_gemm.py
