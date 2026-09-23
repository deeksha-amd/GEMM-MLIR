# Comparing MLIR and PACE by instruction

Start from the [root README](../README.md). This page is reference: how to tell which
instruction each path emits, and what to change when a new ISA arrives.

## Read the mul/add

**MLIR 2×2 loops** (`out/08_avx512.s`): separate multiply and add

```
vmulss  ... %xmm0, %xmm0
vaddss  ... %xmm0, %xmm0
```

That is `C += A * B` in FP32, one element at a time.

**PACE TPP** (`../kernel_dump/*br3*.mxm.s`): fused BF16 dot

```
vpbroadcastd  (%rsi), %zmm0
vdpbf16ps     %zmm1, %zmm0, %zmm6
```

Two BF16 multiplies plus an add into FP32, 16 lanes.

**`vector.contract`** is where an ISA-specific lowering attaches
(`--convert-vector-to-llvm=enable-x86vector`, `enable-amx`, or the x86vector packed-dot
pattern in `chain/`). `linalg.matmul` stays unchanged.

**`portable_matmul.mlir`:** `linalg.matmul` + tile schedule, no ISA in the IR.

**`dpbf16ps.mlir`:** calls the LLVM intrinsic directly. Same mnemonic as PACE, but one
instruction, not a GEMM.

## Compare the teaching paths by instruction

The 2×2 and 8×8 files are teaching IR, so time them against nothing. Compare which
instruction landed:

```bash
PACE_INSPECT=1 ./setup_and_run.sh     # once, from the repo root: PACE JIT dumps
cd mlir
./run_steps.sh
cat out/15_pace_vs_mlir.txt
```

| Path | File to grep | What you should see |
|---|---|---|
| PACE TPP | `../kernel_dump/*br3*.mxm.s` | many `vdpbf16ps` + loads/broadcasts |
| `matmul.mlir` | `out/09_isa_hits.txt` | `vmulss` / `vaddss` |
| `vector_contract.mlir` | `out/13_contract_isa.txt` | `vmulps` (YMM) |
| `portable_matmul.mlir` | `out/20` / `21` / `22` | `linalg.matmul` + tiles, **no** `vdpbf16ps` |
| `dpbf16ps.mlir` | `out/18_dpbf16_isa.txt` | `vdpbf16ps` (decoder only) |

`run_steps.sh` fails if the 2×2 runner does not print `19` or if `18` has no `vdpbf16ps`.

Intrinsic only:

```bash
cd mlir
mlir-opt dpbf16ps.mlir | mlir-translate --mlir-to-llvmir \
  | llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16 -o -
# expect: vdpbf16ps  %zmm..., %zmm..., %zmm...
```

## New instruction checklist

1. `lscpu | grep -E 'amx|avx512_bf16'`
2. Keep `matmul.mlir` / `vector_contract.mlir` / `portable_matmul.mlir`. Do not put new ISAs in the math.
3. Add or enable one conversion (e.g. `--convert-vector-to-llvm=enable-amx`).
4. Diff `09_isa_hits.txt` / `13_contract_isa.txt` against the new mnemonic (`tdpbf16ps`, etc.).
5. PACE: upgrade `amd-pace` / libXSMM, dump the JIT, grep the same mnemonic.
