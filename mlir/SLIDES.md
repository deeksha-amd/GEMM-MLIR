# Deck: bf16 GEMM on AMD EPYC — MLIR vs PACE

5 slides. One story: same GEMM, bf16, one core, 0.34 -> 527 GFLOP/s.
The difference is data packing and the right instruction.

Everything is bf16, M=128 N=4096 K=4096, single core, AMD EPYC 9655.
No f32 numbers anywhere in the main deck. Backup material is at the bottom.

================================================================================

## Slide 1 — Title

**Same GEMM, 1,500x apart**

bf16 128x4096x4096 on one AMD EPYC core
What actually makes a matmul fast

================================================================================

## Slide 2 — The problem

Title: **The math is fixed. The speed is not.**

- One definition: `C[i,j] += A[i,k] * B[k,j]`
- One FLOP count: `2 x M x N x K` = 4.29 GFLOP
- Same source math, same core, measured:

```
        MLIR, straight lowering ........    0.34 GFLOP/s
        MLIR, tiled + vectorized .......   11.1
        PACE / libXSMM .................  501.0
```

- **Question for the deck: where do the other 490 come from?**

Speaker note: everyone assumes the gap is "library vs compiler". It is not.
It is data layout plus one instruction. Next slide shows exactly that.

================================================================================

## Slide 3 — The answer   [MAIN DIAGRAM]

Title: **Pack the data, use the BF16 instruction**

```
   1. PACK  (done once, like PACE's preprocess - not timed)

         Ap : each i32 = A[m][2k] , A[m][2k+1]
         Bp : each i32 = B[2k][n] , B[2k+1][n]        <- VNNI layout
                                |
                                v
   2. TILE   12 rows x 32 columns of C, held in 24 zmm registers
                                |
                                v
   3. INNER LOOP   (2048 steps, one per k-pair)

         2  x  load B      ->  bitcast, free
         12 x  broadcast A ->  bitcast, free
         24 x  vdpbf16ps   ->  accumulate in registers
                                |
                                v
   4. STORE  the 24 registers back to C, once per tile


   RESULT   526.7 GFLOP/s      PACE 501.0      hardware ceiling 577.7
```

Speaker note: build as 4 stacked boxes with down arrows. Highlight box 3.
C is read and written ONCE per tile, not once per k-step - that is what
"held in registers" means.

================================================================================

## Slide 4 — Why it is fast

Title: **One number explains it**

```
        per inner-loop iteration

           straight lowering  :     2 FLOPs  from 4 memory ops
           packed ukernel     :  1536 FLOPs  from 2 cache-line loads
```

Four decisions that produce that ratio:

- **VNNI packing** — the bf16 pairs are already in lane order, so the
  reinterpret is free. No conversion instructions at all.
- **B stored in panels** — the k-loop reads consecutive cache lines.
  Worth 4.2x on its own.
- **24 accumulators in registers** — C touched twice per tile, not per step.
- **`vdpbf16ps`** — one instruction does 32 multiply-accumulates.

Speaker note: without packing, that instruction cannot be used at all.
Layout and ISA are one decision, not two.

================================================================================

## Slide 5 — The numbers

Title: **bf16, one core, 4.29 GFLOP per GEMM**

Bar chart, **log scale**, ceiling as a dashed line:

| Implementation | GFLOP/s |
|---|---|
| MLIR, straight lowering | 0.34 |
| MLIR, tiled + vectorized | 11.1 |
| PACE / libXSMM | 501.0 |
| **MLIR, packed bf16 ukernel** | **526.7** |
| *hardware ceiling* | *577.7* |

Two takeaways to say out loud:

- The jump from 11 to 527 is **packing + instruction**, nothing else
- PACE is at 87% of the ceiling, the ukernel at 91% — **both are at the
  same wall**

Speaker note: do not claim victory over PACE. 5% on one hardcoded shape.

================================================================================

## Slide 6 — Recommendation

Title: **Beating it once is not owning it**

- **Ship PACE** for CPU Linear — any shape, already in PyTorch, 87% of peak
- **Keep `linalg.matmul`** as the source of truth; the schedule is data
- **Don't** hand-write instructions in product code — our kernel is 390
  lines for one shape, one ISA, with a hand-written remainder path
- **Close the gap with a backend**: tpp-mlir (`linalg` -> libXSMM) or IREE
- GPU is a separate stack: Triton or FlyDSL

Speaker note: libXSMM generates our kernel automatically for arbitrary
M/N/K. That generality is the product, not the 5%.

================================================================================
================================================================================
BACKUP SLIDES — only if asked
================================================================================

## Backup A — How we measured

Title: **Four rules that changed the numbers**

- Time **inside** the kernel, not the process — 0.755 s of JIT and setup
  was being counted as GEMM time
- `mlir-runner` JITs at `-O0` by default — switching to `--O3` was 7.6 -> 13.1
- Pin threads — every row at `OMP_NUM_THREADS=1`
- Check the **whole** output — a kernel that skipped a block of rows
  reported **587 GFLOP/s, above the hardware ceiling**, and still passed a
  single-element check

## Backup B — f32 numbers

Same shape, same core, f32 instead of bf16:

| Implementation | f32 GFLOP/s |
|---|---|
| MLIR, straight lowering | 0.38 |
| MLIR, tiled + vectorized | 13.1 |
| torch (reference library) | 111.2 |

Point: in f32, MLIR tiled+vectorized (13.1) beats MLIR bf16 (11.1).
bf16 is SLOWER until you have the bf16 instruction, because the compiler
emulates it as extend / multiply / truncate.

## Backup C — the 8x8 vector.contract path

A third MLIR variant measured 3.4 (bf16) / 11.6 (f32). It is left out of the
main deck on purpose: it differs from the tiled+vectorized row in schedule,
memory model and cleanup passes at the same time, so the comparison says
nothing about which dialect is better. It is a third implementation, not a
controlled experiment.

## Backup D — what PACE still does that our kernel does not

- JITs the equivalent kernel for arbitrary M/N/K
- Handles remainders automatically (ours needs a hand-written 8-row block,
  because 128 is not divisible by our 12-row tile)
- Does the packing itself
- Adds K-blocking, prefetch, epilogue fusion
- Does the f32 -> bf16 output downconvert that our benchmark skips
