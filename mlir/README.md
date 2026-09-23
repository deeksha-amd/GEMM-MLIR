# MLIR GEMM beside PACE TPP

**Recommended MLIR design:** `portable_matmul.mlir` — keep `linalg.matmul`, tile in a schedule, swap a **backend** when the machine changes. Do **not** ship `dpbf16ps.mlir` as the product kernel.

Two compilers, same math `C = A @ B` (PACE Linear is `Y = X @ W.T` with `B = W.T`).

| Stack | Entry | Role |
|---|---|---|
| **MLIR portable** | `portable_matmul.mlir` | **Production shape.** Math + tiles; no ISA in IR |
| **PACE** | `../run_gemm.py` `TPPLinear` | Ship Linear on this EPYC (libXSMM `vdpbf16ps`) |
| **MLIR 2×2** | `matmul.mlir` | Learn naive loops (`vmulss`) |
| **MLIR contract** | `vector_contract.mlir` | Vector dialect hook (`vmulps`) |
| **MLIR intrinsic** | `dpbf16ps.mlir` | **Decoder only** — hardcoded `vdpbf16ps` |

This is **not** plugged into `torch.ops.pace.libxsmmlinear_plain`. New ISA → new **backend** (or PACE upgrade), not a rewrite of `@matmul`.

Numbered dumps: **[STEPS.md](STEPS.md)**. After a run: **`out/INDEX.txt`**.

## Commands

```bash
cd ./mlir
./run_portable.sh       # layer 1–2 only (recommended shape)
./run_steps.sh          # all dumps including portable 20–22
./bench_flops.sh          # tiny FP32 table → out/flops.txt
./bench_flops.sh --prod   # 128×4096×4096 bf16 like-to-like → out/flops_prod.txt
./lower_amx.sh          # no-op on EPYC; records CPUID

# both stacks
cd .
./run_both.sh
```

Need `mlir-opt`, `mlir-translate`, `mlir-runner` and `llc` **in one bin dir**. The scripts search `$MLIR_BIN`, then `/opt/rocm/llvm/bin`, then `/opt/rocm-*/llvm/bin`.

These dumps are **CPU-only**: no GPU, no `srun`, no ROCm container. Run them on the **login node**. The `rocm/dev-ubuntu-*` images and the MI300X compute nodes ship an incomplete set (`mlir-opt` or `llc` missing), which is why `./run_steps.sh` aborts there. Custom build:

```bash
MLIR_BIN=/path/to/llvm/bin ./run_steps.sh
```

## What each dump is (`mlir/out/`)

| File | Step | What to look at |
|---|---|---|
| `00_source.mlir` | copy of `matmul.mlir` | `linalg.matmul`, `@main` returns `C[0,0]` |
| `01_parse.mlir` | parse | same IR, verified |
| `02_canonicalize.mlir` | `--canonicalize` | little change on this example |
| `03_bufferize.mlir` | `--one-shot-bufferize` | tensors → `memref` + `memref.global` |
| `04_linalg_to_loops.mlir` | `--convert-linalg-to-loops` | triple `scf.for` = naive GEMM |
| `05_scf_to_cf.mlir` | `--convert-scf-to-cf` | `cf.cond_br` (week-1 loop lesson) |
| `06_llvm_dialect.mlir` | arith/func/memref/cf → LLVM | still MLIR, LLVM dialect |
| `07_llvmir.ll` | `mlir-translate --mlir-to-llvmir` | LLVM IR |
| `08_avx512.s` | `llc -O3 -mcpu=x86-64-v4` | **read the mul/add here** |
| `09_isa_hits.txt` | grep of `08` | expect `vmulss` + `vaddss` (scalar GEMM) |
| `10_runner.txt` | `mlir-runner -e main` | must print **`19`** |
| `11_contract_parse.mlir` | `vector.contract` | ISA hook (not loops) |
| `12_contract_llvm.mlir` | `--convert-vector-to-llvm=enable-x86vector` + `--convert-arith-to-llvm` | no leftover `arith.` |
| `13_contract.ll` / `.s` / `_isa.txt` | translate + llc of the contract | wider SIMD than the 2×2 loops |
| `14_amx_cpuid.txt` | `lower_amx.sh` | AMX skip on EPYC |
| `15_pace_vs_mlir.txt` | four-way ISA | PACE / 2×2 / contract / intrinsic |
| `16_dpbf16_parse.mlir` | `dpbf16ps.mlir` | `llvm.x86.avx512bf16.dpbf16ps.512` |
| `17_dpbf16.ll` | `mlir-translate` | LLVM IR of the intrinsic |
| `18_dpbf16.s` / `_isa.txt` | `llc +avx512bf16` | decoder; **not** the product path |
| `20_portable_source.mlir` | copy of `portable_matmul.mlir` | `linalg.matmul` only |
| `21_portable_tiled.mlir` | `--transform-interpreter` | tiles around matmul |
| `22_portable_proof.txt` | greps | no `vdpbf16ps` in portable IR |
| `NOTES.txt` | greps of 04 / 09 / 10 / 13 / 18 | proof snippets |
| `INDEX.txt` | written last | this table, plus host / date |
| `RUN_LOG.txt` | copied after the run | full stdout |

## How to read the mul/add (MLIR vs PACE)

**MLIR 2×2 loops** (`08_avx512.s`): separate multiply and add

```
vmulss  ... %xmm0, %xmm0
vaddss  ... %xmm0, %xmm0
```

That is `C += A * B` in FP32, one element at a time. Fine for learning; **not** competitive with TPP.

**PACE TPP** (`../kernel_dump/*br3*.mxm.s`): fused BF16 dot

```
vpbroadcastd  (%rsi), %zmm0
vdpbf16ps     %zmm1, %zmm0, %zmm6
```

That is two BF16 muls + add into FP32, 16 lanes. Different compiler, same GEMM idea.

**`vector.contract`** is where you would later attach `--convert-vector-to-llvm=enable-amx` or a future AMD tile pass. `linalg.matmul` stays.

**MLIR `portable_matmul.mlir` (recommended):** `linalg.matmul` + tile schedule. New box → new backend, same `@matmul`.

**MLIR `dpbf16ps.mlir`:** skip linalg; call the LLVM intrinsic. Same **mnemonic** as PACE; **not** the production shape.

## How to compare the four paths (correctly)

Do **not** time them against each other as GEMM. PACE is a tiled BF16 Linear. The 2×2 and 8×8 files are teaching IR. `dpbf16ps.mlir` is **one** `vdpbf16ps`.

Fair comparison is **which instruction landed**:

```bash
cd .
# PACE dumps (once): needed for column 1 of the table
PACE_INSPECT=1 ./setup_and_run.sh     # or ./run_both.sh

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

`run_steps.sh` fails if the 2×2 runner is not `19` **or** if `18` has no `vdpbf16ps`.

Manual intrinsic-only (existing files unchanged):

```bash
export PATH="/opt/rocm/llvm/bin:${PATH}"
cd ./mlir
mlir-opt dpbf16ps.mlir | mlir-translate --mlir-to-llvmir \
  | llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16 -o -
# you should see:  vdpbf16ps  %zmm..., %zmm..., %zmm...
```

## New instruction checklist

1. `lscpu | grep -E 'amx|avx512_bf16'`
2. Keep `matmul.mlir` / `vector_contract.mlir` / `portable_matmul.mlir`. Do not put new ISAs in product IR.
3. Add or enable one conversion (see `lower_amx.sh`)
4. Diff `09_isa_hits.txt` / `13_contract_isa.txt` vs the new mnemonic (`tdpbf16ps`, etc.)
5. PACE: upgrade `amd-pace` / libXSMM; dump JIT; grep that mnemonic there too
