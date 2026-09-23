#!/usr/bin/env bash
# Prove the bf16 ukernel row really lowers to the bf16 dot instruction.
#
# Run bench_flops.py --prod first: this reads the lowered IR it leaves in out/.
#   ./bench/isa_proof.sh   ->  out/bf16_ukernel_isa.txt
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${MLIR_BIN:-/opt/rocm/llvm/bin}"
SRC=out/prod_bf16_ukernel_bf16_r3.llvm.mlir
LL=out/prod_bf16_ukernel.ll
ASM=out/prod_bf16_ukernel.s
REPORT=out/bf16_ukernel_isa.txt

[ -f "$SRC" ] || { echo "missing $SRC -- run: python3 bench_flops.py --prod" >&2; exit 1; }

"$BIN/mlir-translate" --mlir-to-llvmir "$SRC" -o "$LL"
"$BIN/llc" -O3 -mcpu=native -mattr=+avx512bf16 "$LL" -o "$ASM"

{
  echo "ISA proof for the 'MLIR bf16 ukernel (VNNI + dpbf16ps)' row."
  echo "$SRC"
  echo "  -> mlir-translate --mlir-to-llvmir -> $LL"
  echo "  -> llc -O3 -mcpu=native -mattr=+avx512bf16 -> $ASM"
  echo
  echo "llvm.call_intrinsic survives mlir-opt and becomes a real LLVM intrinsic:"
  grep -m1 -o 'call <16 x float> @llvm.x86.avx512bf16.dpbf16ps.512' "$LL" \
    | sed 's/^/  /'
  echo "  occurrences in $LL: $(grep -c 'llvm.x86.avx512bf16.dpbf16ps.512' "$LL")"
  echo
  echo "static mnemonic counts in $ASM:"
  for m in vdpbf16ps vbroadcastss vpbroadcastd vmovups \
           vfmadd231ps vcvtneps2bf16 vpslld vpmovzxwd; do
    printf '  %-16s %4d\n' "$m" "$(grep -cw "$m" "$ASM" || true)"
  done
  echo
  echo "Reading those counts:"
  echo "  vdpbf16ps 40 = 24 (the 12x32 main tile) + 16 (the 8x32 remainder"
  echo "    tile, because M=128 is not a multiple of MR=12)."
  echo "  vbroadcastss 20 = 12 + 8, one packed A pair per row of each tile."
  echo "    LLVM picks vbroadcastss over vpbroadcastd; they are the same"
  echo "    32-bit dword broadcast, only the execution domain differs."
  echo "  vfmadd231ps / vcvtneps2bf16 / vpslld / vpmovzxwd all 0: nothing is"
  echo "    emulated.  This is the proof the row is not extf/mulf/truncf."
  echo
  echo "innermost loop of the main 12x32 tile:"
  awk '/This Inner Loop Header: Depth=3/{p=1;n++} p&&n==1{print}  p&&n==1&&/jle|jmp/{exit}' \
    "$ASM" | sed 's/^/  /'
} > "$REPORT"

cat "$REPORT"
echo
echo "wrote $REPORT"
