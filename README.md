# GEMM-MLIR

Faster matmul with MLIR on AMD CPUs, measured against PACE (amd-pace + libXSMM).

Same math everywhere: `C = A @ B`. PACE Linear is `Y = X @ W.T`, so `A = X`, `B = W.T`.
PACE does not consume the MLIR output; the two stacks run side by side.

## Results: bf16, 128×4096×4096, one core (AMD EPYC 9655)

| Path | GFLOP/s |
|---|---|
| MLIR, straight lowering (`prod_naive`) | 0.34 |
| MLIR, tiled + vectorized, no BF16 instruction (`prod_portable`) | 11.1 |
| **MLIR, pack + 12×32 tile → `vdpbf16ps`** (`chain/bench_prod_12x32.mlir`) | **~525** |
| PACE TPP / libXSMM | 501 |

FLOPs per GEMM = 2·M·N·K = 4.29 G. The 12×32 row pads M to 132 because 128 is not a multiple of 12.

## 1. Learn MLIR

Straight lowering of `linalg.matmul` to assembly, one dump per step.
Needs `mlir-opt`, `mlir-translate`, `mlir-runner` and `llc` in one bin dir
(`$MLIR_BIN`, else `/opt/rocm/llvm/bin`). CPU only; run on the login node.

```bash
cd mlir
./run_steps.sh        # writes out/00 .. out/22
./run_portable.sh     # portable_matmul.mlir only: tiles, no ISA in the IR
```

Read the dumps in order with [mlir/STEPS.md](mlir/STEPS.md).

| File | What it teaches |
|---|---|
| `mlir/matmul.mlir` | 2×2 `linalg.matmul` → naive loops → `vmulss` |
| `mlir/vector_contract.mlir` | `vector.contract` → SIMD |
| `mlir/dpbf16ps.mlir` | one `vdpbf16ps` via the LLVM intrinsic (decoder, not a GEMM) |
| `mlir/portable_matmul.mlir` | `linalg.matmul` + tile schedule, no ISA in the source |

## 2. PACE

```bash
./setup_and_run.sh                    # venv, CPU torch, amd-pace; dumps JIT kernels to kernel_dump/
PACE_INSPECT=0 ./setup_and_run.sh     # benchmark only
./run_both.sh                         # PACE, then mlir/run_steps.sh
```

`run_gemm.py` runs TPP / AOCL / JIT / NATIVE Linear. The libXSMM kernel is in
`kernel_dump/*br3*.mxm.s` (grep `vdpbf16ps`).

## 3. Production MLIR: `linalg.matmul` → pack → tile → `vdpbf16ps`

Needs LLVM 22 (`x86vector.avx512.dot` and the packed-dot pattern are not in ROCm LLVM 20),
so it runs in Docker. The image is built on first run.

```bash
cd mlir
./chain/run.sh        # 01 and 02: contract -> x86vector.avx512.dot -> vdpbf16ps

docker run --rm --security-opt seccomp=unconfined -v "$PWD":/work -w /work \
  pace-learn-mlir:llvm22 bash /work/chain/bench_in_container.sh
                      # times bench_prod_12x32.mlir -> chain/out/prod12_flops.txt
```

| File | Step |
|---|---|
| `chain/01_contract_to_dot.mlir` | `vector.contract` → `x86vector.avx512.dot` |
| `chain/02_matmul_pack.mlir` | pack K into pairs, 1×16 tile, vectorize, packed-dot |
| `chain/04_matmul_12x32.mlir` | pack `[12,32,2]` from `linalg.matmul`, 12×32 register tile |
| `chain/bench_prod_12x32.mlir` | 128×4096×4096 timed run of the 12×32 schedule |
| `chain/reshape_vnni_contract.py` | reshapes the contract so the packed-dot pattern matches |

## Like-to-like benchmark (naive / portable / PACE)

```bash
cd mlir
./bench_flops.sh --prod   # 128×4096×4096, one core -> out/flops_prod.txt
./bench_flops.sh          # tiny teaching kernels  -> out/flops.txt
```

## More

[mlir/COMPARE.md](mlir/COMPARE.md): how to compare MLIR and PACE by instruction, and what to change for a new ISA.
