#!/usr/bin/env bash
# AMX lowering is the "new matrix instruction" hook.
# AMD EPYC has no AMX tiles — this script records that and, if AMX
# ever appears (or you copy the IR to an SPR+ Xeon), runs enable-amx.
set -euo pipefail
# Same tool search as run_steps.sh (MLIR_BIN overrides).
for _d in ${MLIR_BIN:-} /opt/rocm/llvm/bin /opt/rocm-*/llvm/bin; do
  [[ -x "$_d/mlir-opt" && -x "$_d/llc" ]] || continue
  export PATH="$_d:${PATH}"
  break
done
cd "$(dirname "$0")"
mkdir -p out

FLAGS=$(grep -oE 'amx_tile|amx_bf16' /proc/cpuinfo | sort -u | tr '\n' ' ' || true)
{
  echo "host: $(grep -m1 'model name' /proc/cpuinfo)"
  echo "amx flags: ${FLAGS:-none}"
  echo
} | tee out/14_amx_cpuid.txt

if ! grep -q amx_tile /proc/cpuinfo; then
  cat <<'EOF' | tee -a out/14_amx_cpuid.txt
SKIP: this CPU has no Intel AMX (typical for EPYC).
PACE on this box uses AVX-512 BF16 (vdpbf16ps), not tdpbf16ps.

When a CPU with AMX (or a future AMD tile ISA) is available:
  mlir-opt vector_contract.mlir \\
    --convert-vector-to-llvm=enable-amx \\
    --convert-func-to-llvm --reconcile-unrealized-casts
  then grep the dump for tdpbf16ps / tileloadd.

Do not change linalg.matmul / vector.contract — only this last flag.
EOF
  if [[ -f out/INDEX.txt ]]; then
    echo "14_amx_cpuid.txt written (AMX SKIP)" >> out/INDEX.txt
  fi
  {
    echo
    echo "=== lower_amx.sh ==="
    cat out/14_amx_cpuid.txt
  } >> out/RUN_LOG.txt 2>/dev/null || true
  exit 0
fi

echo "=== AMX present: convert-vector-to-llvm=enable-amx ===" | tee -a out/14_amx_cpuid.txt
mlir-opt vector_contract.mlir \
  --convert-vector-to-llvm="enable-amx" \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts | tee out/14_amx_llvm.mlir
mlir-translate --mlir-to-llvmir out/14_amx_llvm.mlir | tee out/14_amx.ll
llc -O3 -mattr=+amx-tile,+amx-bf16,+amx-int8 out/14_amx.ll -o out/14_amx.s
grep -E 'tdpbf16ps|tileloadd|amx' out/14_amx.s | tee out/14_amx_isa.txt || true
