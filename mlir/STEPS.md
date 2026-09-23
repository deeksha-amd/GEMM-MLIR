# Step-by-step: MLIR GEMM dumps

Each numbered file lives in `mlir/out/`. Re-generate with:

```bash
cd ./mlir
./run_steps.sh
```

Need `/opt/rocm/llvm/bin` on `PATH` (`mlir-opt`, `mlir-translate`, `mlir-runner`, `llc`).

This path does **not** feed PACE. PACE JIT dumps are in `../kernel_dump/`.

---

## 00 `00_source.mlir`

Copy of `matmul.mlir`. Tiny 2×2 FP32 GEMM so you can read every op:

```
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C[0,0] = 1*5 + 2*7 = 19    ← @main returns this as i32
```

Look at: `linalg.matmul`. That is the whole kernel. Later steps only *lower* it.

---

## 01 `01_parse.mlir`

`mlir-opt matmul.mlir` with no passes. Confirms the file is valid MLIR.

---

## 02 `02_canonicalize.mlir`

`--canonicalize`. Little change on this example (constants already folded).

---

## 03 `03_bufferize.mlir`

`--one-shot-bufferize="bufferize-function-boundaries"`

Tensors become `memref`. Constants become `memref.global`. `linalg.matmul` now writes into a buffer.

Look at: `memref.global`, `memref.get_global`, still `linalg.matmul`.

---

## 04 `04_linalg_to_loops.mlir`  ← first place you see mul+add

`--convert-linalg-to-loops`

Naive GEMM: three nested `scf.for` (i, j, k) and

```
%prod = arith.mulf %a, %b : f32
%sum  = arith.addf %c, %prod : f32
```

That is `C[i,j] += A[i,k] * B[k,j]`. Same math as PACE Linear (`Y = X @ W.T`) if you set `A=X`, `B=W.T`.

---

## 05 `05_scf_to_cf.mlir`

`--convert-scf-to-cf`

Structured `scf.for` becomes `cf.cond_br` / `cf.br` (the week-1 loop form).

---

## 06 `06_llvm_dialect.mlir`

Full CPU pipeline down to the LLVM *dialect* (still a `.mlir` file):

```
--expand-strided-metadata --lower-affine
--convert-arith-to-llvm --convert-func-to-llvm
--finalize-memref-to-llvm --convert-cf-to-llvm
--convert-index-to-llvm --reconcile-unrealized-casts
```

Look at: `llvm.fmul`, `llvm.fadd`. No `linalg` / `scf` left.

This file is the input to both `mlir-runner` (step 10) and `mlir-translate` (step 07).

---

## 07 `07_llvmir.ll`

`mlir-translate --mlir-to-llvmir`

Real LLVM IR. Look at: `fmul float`, `fadd float`.

---

## 08 `08_avx512.s`  ← assembly of the 2×2

```
llc -O3 -mcpu=x86-64-v4 -mattr=+avx512f,+avx512bf16
```

`-mcpu=x86-64-v4` means “AVX-512-capable generic x86”. This ROCm `llc` does not usefully target `znver5`.

The 2×2 is too small to emit `vdpbf16ps`. You should see scalar `vmulss` / `vaddss`.

---

## 09 `09_isa_hits.txt`

Grep of `08_avx512.s`. Expected on this box:

```
vmulss
vaddss
```

Not `vdpbf16ps` (that is PACE / libXSMM) and not `tdpbf16ps` (Intel AMX).

---

## 10 `10_runner.txt`  ← correctness check

```
mlir-runner 06_llvm_dialect.mlir -e main -entry-point-result=i32
```

Must print **`19`**. `run_steps.sh` fails if it does not.

---

## 11–13  `vector.contract` (the ISA hook)

`linalg.matmul` is the math. New matrix instructions attach **here**, not by rewriting the 2×2.

| File | Command | What to look at |
|---|---|---|
| `11_contract_parse.mlir` | `mlir-opt vector_contract.mlir` | `vector.contract` 8×8, `kind = add` |
| `12_contract_llvm.mlir` | `--convert-vector-to-llvm=enable-x86vector` plus `--convert-arith-to-llvm` | LLVM dialect; **no `arith.` left** (otherwise translate fails) |
| `13_contract.ll` | `mlir-translate --mlir-to-llvmir` | LLVM IR of the 8×8 |
| `13_contract.s` | `llc -mcpu=x86-64-v4` | wider SIMD than the 2×2 |
| `13_contract_isa.txt` | grep of `.s` | `vmulps` / `vfmadd` / similar |

When a new ISA lands, change only the convert flag (e.g. `enable-amx`), keep `vector_contract.mlir`.

---

## 16–18  `dpbf16ps.mlir` (LLVM intrinsic → `vdpbf16ps`)

Does **not** change `matmul.mlir` or `vector_contract.mlir`. Already LLVM dialect; skip linalg.

| File | Command | What to look at |
|---|---|---|
| `16_dpbf16_parse.mlir` | `mlir-opt dpbf16ps.mlir` | `llvm.call_intrinsic "llvm.x86.avx512bf16.dpbf16ps.512"` |
| `17_dpbf16.ll` | `mlir-translate --mlir-to-llvmir` | the LLVM intrinsic |
| `18_dpbf16.s` | `llc -mattr=+avx512f,+avx512bf16` | **`vdpbf16ps`** |
| `18_dpbf16_isa.txt` | grep of `.s` | must be non-empty; `run_steps.sh` fails otherwise |

This is one instruction, **not** the production shape (see 20–22). Not TPP Linear.

---

## 20–22  `portable_matmul.mlir` (recommended MLIR shape)

Keep `linalg.matmul`. Tile in a transform schedule. **No** `vdpbf16ps` in this IR.
Quick run: `./run_portable.sh`.

| File | Command | What to look at |
|---|---|---|
| `20_portable_source.mlir` | copy | `@matmul` + transform tiles `[8,8,8]` |
| `21_portable_tiled.mlir` | `--transform-interpreter` (or affine-loop-tile fallback) | loops around matmul |
| `22_portable_proof.txt` | greps | `PASS: recommended path has no vdpbf16ps` |

New architecture: change tile sizes or the **backend**, not `@matmul`.

---

## 15 `15_pace_vs_mlir.txt`

ISA dump **plus** the recommended portable path (not a GFLOPS race):

1. PACE TPP: `vdpbf16ps` from `../kernel_dump/*br3*.mxm.s`
2. `matmul.mlir` 2×2: `vmulss` + `vaddss`
3. `vector_contract.mlir` 8×8: `vmulps` / similar
4. `dpbf16ps.mlir`: `vdpbf16ps` (decoder only)
5. `portable_matmul.mlir`: **no ISA** in IR (`22_portable_proof.txt`)

If the PACE dumps are missing, run `../setup_and_run.sh` with `PACE_INSPECT=1`. Then `./run_steps.sh` again so 15 picks them up.

---

## `NOTES.txt`

Short greps pulled out of 04 / 09 / 10 / 13 / 18 / 22 so the dump folder is readable without opening every IR file.

## `INDEX.txt`

Manifest written at the end of `run_steps.sh`: date, host, `mlir-opt` path, and a one-line description of every file.

## `RUN_LOG.txt`

Full stdout of `run_steps.sh` (copied after the script finishes).
