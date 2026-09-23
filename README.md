# GEMM-MLIR

Faster matmul with MLIR on AMD CPUs. PACE TPP GEMM beside an MLIR lowering.

Two compilers, same math. **PACE does not consume the MLIR dumps.**

| Stack | What you run | Kernel you read |
|---|---|---|
| **MLIR portable** | `./mlir/run_portable.sh` or `run_steps.sh` | `linalg.matmul` + tiles; **no** ISA in IR |
| **PACE** (amd-pace + libXSMM) | `./setup_and_run.sh` | `kernel_dump/*br3*.mxm.s` → `vdpbf16ps` |
| **MLIR 2×2** | `./mlir/run_steps.sh` | `mlir/out/09_isa_hits.txt` → `vmulss` |
| **MLIR contract** | same script | `mlir/out/13_contract_isa.txt` → `vmulps` |
| **MLIR intrinsic** | same script | decoder only: `18_dpbf16_isa.txt` |

Linear GEMM is `Y = X @ W.T` (+ optional bias). The MLIR file is `C = A @ B`; match PACE with `A = X`, `B = W.T`.

## Quick start

```bash
cd GEMM-MLIR

# PACE on EPYC CPU (not the MI300X GPU). Needs the CPU venv from setup_and_run.sh.
./setup_and_run.sh

# MLIR recommended shape (no ISA in source): see mlir/DESIGN.md
./mlir/run_portable.sh
./mlir/run_steps.sh
./mlir/lower_amx.sh

# Both (PACE first; inspect is slow on the 4096 GEMM).
./run_both.sh
```

Skip PACE inspect (benchmark only):

```bash
PACE_INSPECT=0 ./setup_and_run.sh
```

## Docs

| File | Contents |
|---|---|
| `mlir/LEARNINGS.txt` | **Start here.** Concepts, benchmarking bugs, all measurements, recommendation |
| `mlir/DESIGN.md` | **Recommended production shape** (linalg + tiles + backends) |
| `mlir/README.md` | PACE vs MLIR, ISA comparison |
| `mlir/STEPS.md` | What each `mlir/out/NN_*` file is, in order |
| `mlir/out/INDEX.txt` | Generated manifest after a run |
| `mlir/out/15_pace_vs_mlir.txt` | ISA table + portable shape |

## Layout

```
run_gemm.py          TPP / AOCL / JIT / NATIVE Linear + 3-layer inspect
setup_and_run.sh     venv, torch CPU, amd-pace, LIBXSMM_VERBOSE=-1
run_both.sh          PACE then MLIR
kernel_dump/         libXSMM JIT .mxm / .s  (PACE)
mlir/
  matmul.mlir        2x2 linalg.matmul, @main returns 19
  vector_contract.mlir   8x8 vector.contract (ISA hook)
  dpbf16ps.mlir      ISA decoder (not the product path)
  portable_matmul.mlir   recommended: linalg.matmul + tile schedule
  DESIGN.md          production-shaped MLIR layers
  run_portable.sh    layer 1–2 only
  bench_flops.sh     PACE vs 2x2 vs contract vs portable GFLOP/s
  run_steps.sh       writes out/00 .. out/22, INDEX, 15
  lower_amx.sh       writes out/14_amx_cpuid.txt (SKIP on EPYC)
  STEPS.md           walkthrough of the dumps
  README.md          overview
```
