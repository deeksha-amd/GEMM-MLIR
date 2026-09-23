#!/usr/bin/env bash
# Step-by-step MLIR GEMM lowering. Saves every dump under mlir/out/.
# Uses ROCm mlir-opt (same as mlir-learn/run_week1.sh). Does NOT call PACE.
#
# Usage:
#   ./run_steps.sh
#   ./lower_amx.sh     # after this; writes 14_amx_cpuid.txt
set -euo pipefail
# Needs mlir-opt + mlir-translate + mlir-runner + llc in ONE bin dir.
# The ROCm docker images and some compute nodes ship only part of that set,
# so search instead of hardcoding. Override with MLIR_BIN=/path/to/llvm/bin.
for _d in ${MLIR_BIN:-} /opt/rocm/llvm/bin /opt/rocm-*/llvm/bin; do
  [[ -x "$_d/mlir-opt" && -x "$_d/mlir-translate" \
     && -x "$_d/mlir-runner" && -x "$_d/llc" ]] || continue
  export PATH="$_d:${PATH}"
  break
done
cd "$(dirname "$0")"
OUT=out
mkdir -p "$OUT"

# Capture a full log without hiding stdout. Skip RUN_LOG in the wipe so
# the outer tee/redirect can keep the file open.
if [[ -z "${MLIR_STEPS_LOGGING:-}" ]]; then
  export MLIR_STEPS_LOGGING=1
  set +e
  "$0" "$@" >"$OUT/RUN_LOG.txt" 2>&1
  rc=$?
  cat "$OUT/RUN_LOG.txt"
  exit "$rc"
fi

# Keep 14_amx_* if lower_amx.sh already ran; wipe the rest for a clean re-run.
find "$OUT" -maxdepth 1 -type f ! -name '14_amx*' ! -name 'RUN_LOG.txt' -delete 2>/dev/null || true

need() {
  command -v "$1" >/dev/null && return 0
  cat >&2 <<EOF
missing $1

This step needs mlir-opt, mlir-translate, mlir-runner and llc together.
Searched: \${MLIR_BIN}, /opt/rocm/llvm/bin, /opt/rocm-*/llvm/bin
Here:     $(hostname)

These dumps are CPU-only (no GPU, no srun, no ROCm container needed).
Run them on the login node, or point MLIR_BIN at a full LLVM/MLIR build:
  MLIR_BIN=/path/to/llvm/bin ./run_steps.sh
EOF
  exit 1
}
need mlir-opt
need mlir-translate
need mlir-runner
need llc

log() { echo "=== $* ==="; }

# 00 source
cp -f matmul.mlir "$OUT/00_source.mlir"
log "00 source (linalg.matmul 2x2, C[0,0] should be 19)"
cat "$OUT/00_source.mlir"

log "01 parse"
mlir-opt matmul.mlir | tee "$OUT/01_parse.mlir" >/dev/null
wc -l "$OUT/01_parse.mlir"

log "02 canonicalize"
mlir-opt matmul.mlir --canonicalize | tee "$OUT/02_canonicalize.mlir" >/dev/null
wc -l "$OUT/02_canonicalize.mlir"

log "03 bufferize (tensor -> memref)"
mlir-opt matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  | tee "$OUT/03_bufferize.mlir" >/dev/null
wc -l "$OUT/03_bufferize.mlir"
echo "  look for: memref.global  /  memref.get_global  /  linalg.matmul on memrefs"

log "04 linalg -> loops (the naive GEMM: 3 nested scf.for + mulf/addf)"
mlir-opt matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  | tee "$OUT/04_linalg_to_loops.mlir" >/dev/null
wc -l "$OUT/04_linalg_to_loops.mlir"
echo "  look for: scf.for  arith.mulf  arith.addf"

log "05 scf -> cf (cond_br, week-1 loop form)"
mlir-opt matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  | tee "$OUT/05_scf_to_cf.mlir" >/dev/null
wc -l "$OUT/05_scf_to_cf.mlir"
echo "  look for: cf.cond_br  cf.br"

log "06 llvm dialect (still MLIR, LLVM ops)"
mlir-opt matmul.mlir \
  --one-shot-bufferize="bufferize-function-boundaries" \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --expand-strided-metadata \
  --lower-affine \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-index-to-llvm \
  --reconcile-unrealized-casts \
  > "$OUT/06_llvm_dialect.mlir"
wc -l "$OUT/06_llvm_dialect.mlir"
echo "  look for: llvm.fmul  llvm.fadd  llvm.br   (no linalg / scf left)"

log "07 llvm ir (mlir-translate)"
mlir-translate --mlir-to-llvmir "$OUT/06_llvm_dialect.mlir" > "$OUT/07_llvmir.ll"
wc -l "$OUT/07_llvmir.ll"
echo "  look for: fmul float  /  fadd float"

log "08 llc assembly (x86-64-v4 = AVX-512 capable, still scalar 2x2)"
llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16 \
  "$OUT/07_llvmir.ll" -o "$OUT/08_avx512.s"
wc -l "$OUT/08_avx512.s"

log "09 ISA hits from 08 (expect vmulss + vaddss, NOT vdpbf16ps)"
grep -E 'vmulss|vaddss|vmulps|vfmadd|vdpbf16|tdpbf16' "$OUT/08_avx512.s" \
  | tee "$OUT/09_isa_hits.txt" || true
if [[ ! -s "$OUT/09_isa_hits.txt" ]]; then
  echo "(no mul/add mnemonics grepped — inspect 08_avx512.s)" | tee "$OUT/09_isa_hits.txt"
fi

log "10 mlir-runner (must print 19)"
mlir-runner "$OUT/06_llvm_dialect.mlir" -e main -entry-point-result=i32 \
  | tee "$OUT/10_runner.txt"
got=$(tr -d '[:space:]' < "$OUT/10_runner.txt")
if [[ "$got" != "19" ]]; then
  echo "FAIL: runner printed '$got', expected 19" >&2
  exit 1
fi
echo "  PASS: C[0,0] = 19"

log "11 vector.contract parse (ISA hook, 8x8)"
mlir-opt vector_contract.mlir | tee "$OUT/11_contract_parse.mlir" >/dev/null
wc -l "$OUT/11_contract_parse.mlir"

log "12 vector.contract -> llvm dialect (enable-x86vector + arith/func)"
# arith.constant / arith.addf must become llvm before mlir-translate.
mlir-opt vector_contract.mlir \
  --convert-vector-to-llvm="enable-x86vector" \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --convert-index-to-llvm \
  --reconcile-unrealized-casts \
  > "$OUT/12_contract_llvm.mlir"
wc -l "$OUT/12_contract_llvm.mlir"
if grep -q 'arith\.' "$OUT/12_contract_llvm.mlir"; then
  echo "WARN: arith dialect still present; mlir-translate may fail" >&2
fi

log "13 vector.contract llvm ir + asm"
mlir-translate --mlir-to-llvmir "$OUT/12_contract_llvm.mlir" \
  > "$OUT/13_contract.ll"
wc -l "$OUT/13_contract.ll"
llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f \
  "$OUT/13_contract.ll" -o "$OUT/13_contract.s"
wc -l "$OUT/13_contract.s"
grep -E 'vfmadd|vmulps|vaddps|mulps|addps|mulss|addss' "$OUT/13_contract.s" \
  | head -40 | tee "$OUT/13_contract_isa.txt" || true
if [[ ! -s "$OUT/13_contract_isa.txt" ]]; then
  echo "(no vector mul/add grepped — inspect 13_contract.s)" | tee "$OUT/13_contract_isa.txt"
fi

log "16 llvm intrinsic parse (vdpbf16ps; not linalg / not vector.contract)"
mlir-opt dpbf16ps.mlir | tee "$OUT/16_dpbf16_parse.mlir" >/dev/null
wc -l "$OUT/16_dpbf16_parse.mlir"

log "17 llvm ir of dpbf16ps.mlir"
mlir-translate --mlir-to-llvmir "$OUT/16_dpbf16_parse.mlir" > "$OUT/17_dpbf16.ll"
wc -l "$OUT/17_dpbf16.ll"
echo "  look for: llvm.x86.avx512bf16.dpbf16ps.512"

log "18 llc assembly (must contain vdpbf16ps)"
llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16 \
  "$OUT/17_dpbf16.ll" -o "$OUT/18_dpbf16.s"
wc -l "$OUT/18_dpbf16.s"
grep -E 'vdpbf16ps' "$OUT/18_dpbf16.s" | tee "$OUT/18_dpbf16_isa.txt" || true
if [[ ! -s "$OUT/18_dpbf16_isa.txt" ]]; then
  echo "FAIL: llc did not emit vdpbf16ps (need +avx512bf16 on this llc)" >&2
  exit 1
fi
echo "  PASS: MLIR LLVM intrinsic lowered to vdpbf16ps"

log "20 portable source (recommended MLIR shape: linalg.matmul, no ISA)"
if grep -qE 'vdpbf16|call_intrinsic|avx512' portable_matmul.mlir; then
  echo "FAIL: portable_matmul.mlir named an ISA — that is not the production shape" >&2
  exit 1
fi
cp -f portable_matmul.mlir "$OUT/20_portable_source.mlir"
grep -n 'linalg.matmul' "$OUT/20_portable_source.mlir"
echo "  PASS: layer 1 is linalg.matmul only"

log "21 tile (layer 2, still no ISA — transform-interpreter)"
set +e
mlir-opt portable_matmul.mlir --transform-interpreter --canonicalize \
  > "$OUT/21_portable_tiled.mlir" 2> "$OUT/21_portable_tile_err.txt"
tile_rc=$?
set -e
if [[ "$tile_rc" -ne 0 ]]; then
  echo "  transform-interpreter failed; affine-loop-tile fallback"
  cat "$OUT/21_portable_tile_err.txt" || true
  mlir-opt portable_matmul.mlir \
    --one-shot-bufferize="bufferize-function-boundaries" \
    --convert-linalg-to-affine-loops \
    --affine-loop-tile="tile-sizes=8,8,8" \
    > "$OUT/21_portable_tiled.mlir"
  echo "fallback (affine-loop-tile after dropping linalg)" > "$OUT/21_portable_tile_method.txt"
else
  echo "transform-interpreter" > "$OUT/21_portable_tile_method.txt"
  : > "$OUT/21_portable_tile_err.txt"
fi
wc -l "$OUT/21_portable_tiled.mlir"
echo "  look for: scf.for wrapping linalg.matmul (or affine.for if fallback)"

log "22 portable proof (no vdpbf16ps in recommended IR)"
{
  echo "method: $(cat "$OUT/21_portable_tile_method.txt")"
  echo
  echo "linalg.matmul count (source): $(grep -c 'linalg.matmul' "$OUT/20_portable_source.mlir" || true)"
  echo "linalg.matmul count (tiled):  $(grep -c 'linalg.matmul' "$OUT/21_portable_tiled.mlir" || true)"
  echo "scf.for / affine.for (tiled): $(grep -cE 'scf.for|affine.for' "$OUT/21_portable_tiled.mlir" || true)"
  echo
  echo "PASS: recommended path has no vdpbf16ps / call_intrinsic"
} | tee "$OUT/22_portable_proof.txt"
if grep -qE 'vdpbf16|call_intrinsic' "$OUT/20_portable_source.mlir" \
   "$OUT/21_portable_tiled.mlir"; then
  echo "FAIL: ISA leaked into portable IR" >&2
  exit 1
fi

log "15 four-way ISA (PACE vs 2x2 vs contract vs llvm intrinsic)"
{
  echo "# Compare MNEMONICS, not GEMM GFLOPS."
  echo "# Recommended MLIR design is portable_matmul.mlir (no ISA in IR)."
  echo "# dpbf16ps.mlir is an ISA decoder only — not the production shape."
  echo "# Unfair timing: PACE is a tiled Linear; 18 is one instruction."
  echo
  echo "## 1) PACE TPP libXSMM JIT  (full BF16 Linear / BRGEMM)"
  echo "expect: vdpbf16ps   source: ../kernel_dump/*br3*.mxm.s"
  if ls ../kernel_dump/*br3*.mxm.s >/dev/null 2>&1; then
    grep -h -n 'vdpbf16ps' ../kernel_dump/*br3*.mxm.s 2>/dev/null | head -8 || true
    echo
    echo "files:"
    ls -1 ../kernel_dump/*br3*.mxm.s | sed 's|.*/|  |' | head -20 || true
  else
    echo "no ../kernel_dump/*br3*.mxm.s yet — run ../setup_and_run.sh with PACE_INSPECT=1"
  fi
  echo
  echo "## 2) MLIR linalg.matmul 2x2  (matmul.mlir -> 09_isa_hits.txt)"
  echo "expect: vmulss / vaddss   (FP32 scalar; NOT vdpbf16ps)"
  cat "$OUT/09_isa_hits.txt"
  echo
  echo "## 3) MLIR vector.contract 8x8  (vector_contract.mlir -> 13_contract_isa.txt)"
  echo "expect: vmulps / vaddss   (FP32 SIMD; NOT vdpbf16ps)"
  cat "$OUT/13_contract_isa.txt"
  echo
  echo "## 4) MLIR llvm.call_intrinsic  (dpbf16ps.mlir -> 18_dpbf16_isa.txt)"
  echo "expect: vdpbf16ps   ISA decoder, NOT the production shape"
  cat "$OUT/18_dpbf16_isa.txt"
  echo
  echo "## 5) Recommended MLIR shape  (portable_matmul.mlir -> 22_portable_proof.txt)"
  echo "expect: linalg.matmul + tiles; NO vdpbf16ps in the IR"
  cat "$OUT/22_portable_proof.txt"
  echo
  echo "## Summary"
  echo "path                         role                   instruction in IR"
  echo "portable_matmul.mlir         PRODUCTION SHAPE       none (linalg.matmul)"
  echo "PACE TPP                     ship Linear on EPYC    vdpbf16ps (JIT, not MLIR)"
  echo "matmul.mlir                  learn naive loops      vmulss/vaddss"
  echo "vector_contract.mlir         vector dialect hook    vmulps (YMM)"
  echo "dpbf16ps.mlir                decode one ISA         vdpbf16ps (hardcoded)"
} | tee "$OUT/15_pace_vs_mlir.txt"

log "NOTES (extracted proof from the dumps)"
{
  echo "# Extracted from the dumps so you do not have to grep by hand."
  echo
  echo "## 04 linalg -> loops (naive GEMM mul+add)"
  grep -n -E 'scf.for|arith.mulf|arith.addf' "$OUT/04_linalg_to_loops.mlir" || true
  echo
  echo "## 09 2x2 llc ISA"
  cat "$OUT/09_isa_hits.txt"
  echo
  echo "## 10 mlir-runner (expect 19)"
  cat "$OUT/10_runner.txt"
  echo
  echo "## 13 8x8 vector.contract ISA (first 15 lines)"
  head -15 "$OUT/13_contract_isa.txt"
  echo
  echo "## 18 llvm intrinsic ISA (expect vdpbf16ps)"
  cat "$OUT/18_dpbf16_isa.txt"
  echo
  echo "## 22 portable proof (recommended shape, no ISA)"
  cat "$OUT/22_portable_proof.txt"
} | tee "$OUT/NOTES.txt"

log "INDEX"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOST=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')
MLIR_BIN=$(command -v mlir-opt)
MLIR_VER=$("$MLIR_BIN" --version 2>&1 | head -1 || true)
{
  echo "mlir/out  generated ${STAMP}"
  echo "host:     ${HOST}"
  echo "mlir-opt: ${MLIR_BIN}"
  echo "version:  ${MLIR_VER}"
  echo
  echo "10_runner.txt must be 19  (C[0,0] of 2x2 GEMM = 1*5+2*7)"
  echo "18_dpbf16_isa.txt must contain vdpbf16ps  (decoder only)"
  echo "22_portable_proof.txt: recommended path must NOT contain vdpbf16ps"
  echo
  echo "file                         what"
  echo "----                         ----"
  echo "00_source.mlir               copy of matmul.mlir (linalg.matmul)"
  echo "01_parse.mlir                parse / verify"
  echo "02_canonicalize.mlir         --canonicalize"
  echo "03_bufferize.mlir            tensors -> memref"
  echo "04_linalg_to_loops.mlir      triple scf.for + arith.mulf/addf"
  echo "05_scf_to_cf.mlir            cf.cond_br loops"
  echo "06_llvm_dialect.mlir         LLVM dialect (input to runner + translate)"
  echo "07_llvmir.ll                 LLVM IR"
  echo "08_avx512.s                  llc asm of the 2x2 (read mul/add here)"
  echo "09_isa_hits.txt              grep of 08 (vmulss/vaddss)"
  echo "10_runner.txt                mlir-runner result (19)"
  echo "11_contract_parse.mlir       vector.contract 8x8"
  echo "12_contract_llvm.mlir        convert-vector-to-llvm=enable-x86vector"
  echo "13_contract.ll               LLVM IR of the contract"
  echo "13_contract.s                llc asm of the contract"
  echo "13_contract_isa.txt          grep of 13_contract.s"
  echo "14_amx_cpuid.txt             from lower_amx.sh (SKIP on EPYC)"
  echo "15_pace_vs_mlir.txt          ISA table + portable shape (section 5)"
  echo "16_dpbf16_parse.mlir         dpbf16ps.mlir (ISA decoder, not product)"
  echo "17_dpbf16.ll                 LLVM IR of the intrinsic"
  echo "18_dpbf16.s                  llc asm (read vdpbf16ps here)"
  echo "18_dpbf16_isa.txt            grep of 18 (must be vdpbf16ps)"
  echo "20_portable_source.mlir      recommended: linalg.matmul, no ISA"
  echo "21_portable_tiled.mlir       layer 2 tiles"
  echo "22_portable_proof.txt        no vdpbf16ps in portable IR"
  echo "NOTES.txt                    greps of 04 / 09 / 10 / 13 / 18 / 22"
  echo "INDEX.txt                    this file"
  echo "RUN_LOG.txt                  stdout of run_steps.sh + lower_amx.sh"
  echo
  echo "files now in this directory:"
  ls -1 "$OUT"
} | tee "$OUT/INDEX.txt"

echo
echo "Dumps in $PWD/$OUT"
echo "Walkthrough:  $PWD/STEPS.md"
echo "Overview:     $PWD/README.md"
echo "Next:         ./lower_amx.sh"
echo "PACE JIT:     grep vdpbf16ps ../kernel_dump/*br3*.mxm.s | head"
