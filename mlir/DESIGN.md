# Recommended production shape (MLIR advantage)

Keep **one** GEMM: `linalg.matmul`. When a new architecture ships, add or swap a
**backend**. Do not put `vdpbf16ps` (or MFMA, AMX, …) in the source IR.

`dpbf16ps.mlir` is **not** this design. It is a decoder for one x86 instruction.

## Layers

```
Your model (PyTorch Linear, ONNX MatMul, …)
        │
        ▼
  linalg.matmul          portable_matmul.mlir  ← you stay here
        │
        ▼
  tile / pack / vectorize   transform in that file (layer 2)
        │
        ├── CPU  → LLVM, or call PACE / libXSMM / oneDNN
        ├── GPU  → IREE / Triton / hipBLASLt (MFMA on MI300X)
        └── next → new last-mile only
```

| Layer | What changes on a new box | This repo |
|---|---|---|
| 1. Math | **Nothing** | `linalg.matmul` in `portable_matmul.mlir` |
| 2. Schedule | Tile sizes, maybe packing | `--transform-interpreter` → `out/20_*.mlir` |
| 3. Backend | New pass or library | CPU LLVM generic; PACE is today’s EPYC Linear |

## What to run

```bash
cd ./mlir
./run_steps.sh          # includes portable dumps 20–22
cat out/20_portable_source.mlir   # still linalg.matmul, no ISA
cat out/21_portable_tiled.mlir    # scf.for around linalg.matmul
```

Layer 2 only (no PACE, no intrinsic):

```bash
export PATH="/opt/rocm/llvm/bin:${PATH}"
mlir-opt portable_matmul.mlir --transform-interpreter --canonicalize
```

## Why this is the MLIR advantage

- New EPYC / Xeon / MI300X: **keep** `@matmul`. Change tiles or the last compiler.
- Handwriting `llvm.call_intrinsic "…dpbf16ps…"` means you **do** rewrite GEMM per ISA.
- Naive `--convert-linalg-to-loops` on the 2×2 (`matmul.mlir`) is the math with **no**
  schedule. Fine for learning; not a product kernel.

## What to ship today vs later

| Goal | Use |
|---|---|
| Fast Linear on this EPYC **now** | PACE TPP (`../run_gemm.py`) — a backend, not MLIR IR |
| Portable compiler for the **next** box | This shape, then a real compiler (IREE, torch-mlir, Triton) |
| Understand `vdpbf16ps` | `dpbf16ps.mlir` dumps only |

A full product stack is IREE (or similar): it already does layer 2–3 for CPU and GPU.
This folder shows the **shape** with stock `mlir-opt`, not a replacement for IREE/PACE.

## FLOPs / GFLOP/s (PACE vs the three MLIR kernels)

```bash
cd ./mlir
./bench_flops.sh          # writes out/flops.txt
```

Formula: **FLOPs = 2×M×N×K** per GEMM (multiply + add). **GFLOP/s = FLOPs × inner_iters / seconds / 1e9**.

| Row | Size | What you measure |
|---|---|---|
| `matmul_2x2` | 2×2×2 = 16 FLOPs | naive FP32 loops (`vmulss`) |
| `vector_contract_8x8` | 8×8×8 = 1024 FLOPs | FP32 `vector.contract` |
| `portable_16x16` | 16×16×16 = 8192 FLOPs | tiled `linalg.matmul`, generic LLVM |
| PACE 16³ / 32×64×128 / 128×4096² | 2MNK | TPP BF16 Linear |

Like-to-like **128×4096×4096 bf16** (does not rewrite the 2×2 teaching files):

```bash
./bench_flops.sh --prod     # out/flops_prod.txt   (naive MLIR may take minutes)
```

| Row | Same 2MNK | How it is implemented |
|---|---|---|
| matmul-style | yes | one `linalg.matmul`, `--convert-linalg-to-loops` |
| vector.contract | yes | **8×8** `vector.contract` tiles (a 128×4096 vector is not a register) |
| portable | yes | `linalg.matmul` + `tile_sizes [32,64,64]` |
| PACE | yes | TPP BF16 Linear |

`mlir-runner` is typically **1 thread**; PACE uses `OMP_NUM_THREADS`.
