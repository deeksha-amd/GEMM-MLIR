#!/usr/bin/env python3
"""GEMM FLOPs / GFLOP/s: PACE vs matmul.mlir vs vector.contract vs portable MLIR.

  ./bench_flops.sh            tiny teaching kernels  -> out/flops.txt
  ./bench_flops.sh --prod     128x4096x4096, 1 core  -> out/flops_prod.txt

FLOPs per GEMM = 2*M*N*K. GFLOP/s = FLOPs * reps / seconds / 1e9.

--prod is the apples-to-apples run:
  * every row is pinned to ONE core (OMP/MKL threads = 1)
  * MLIR rows time only the GEMM (rtclock inside @main), not JIT or fills
  * mlir-runner JITs at --O3 (its option is spelled --O3, and defaults to --O0)
  * the portable schedule tiles twice, vectorizes, lowers vector.contract to
    outer products and hoists the C tile out of the k loop, so the inner loop
    is 4 vbroadcastss + 4 vfmadd231ps on zmm accumulators
  * f32 and bf16 are reported separately: without a bf16 ukernel, MLIR
    emulates bf16 as extend/compute/truncate, which is not what PACE does
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "out"
BENCH = ROOT / "bench"

SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

CPU_PIPELINE = [
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-loops",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]
VECTOR_PIPELINE = [
    "--convert-scf-to-cf",
    "--convert-vector-to-llvm=enable-x86vector",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]
# 2-D transfer_read/write need convert-vector-to-scf before vector-to-llvm.
# full-unroll keeps llvm.alloca out of the inner loop (LLVM 20 JIT segfaults
# otherwise on 8x8 transfers).
VECTOR_MEMREF_PIPELINE = [
    "--convert-linalg-to-loops",
    "--convert-vector-to-scf=full-unroll",
    "--canonicalize",
    "--cse",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-vector-to-llvm=enable-x86vector",
    "--convert-arith-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]
# Vectorized linalg (transform schedule already ran): bufferize, then lower
# the vector ops that vectorization produced.
#
# loop-invariant-subset-hoisting runs first and on tensors: it pulls the
# transfer_read/transfer_write pair for the C register tile out of the k loop
# so the loop carries a vector<4x16xf32> in registers instead of a
# load/fma/store round trip per k step (5.1 -> 13.1 GFLOP/s).  It has to
# happen before bufferization, where the tensor SSA chain that proves the
# subsets are invariant is gone.
VECTORIZED_CPU_PIPELINE = [
    "--loop-invariant-code-motion",
    "--loop-invariant-subset-hoisting",
    "--canonicalize",
    "--cse",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--canonicalize",
    "--cse",
    "--convert-vector-to-scf=full-unroll",
    "--convert-linalg-to-loops",
    "--canonicalize",
    "--cse",
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--lower-affine",
    "--convert-vector-to-llvm=enable-x86vector",
    "--convert-arith-to-llvm",
    "--finalize-memref-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]


def find_bin() -> Path:
    cands = []
    if os.environ.get("MLIR_BIN"):
        cands.append(Path(os.environ["MLIR_BIN"]))
    cands.append(Path("/opt/rocm/llvm/bin"))
    cands.extend(sorted(Path("/opt").glob("rocm-*/llvm/bin")))
    need = ("mlir-opt", "mlir-translate", "mlir-runner")
    for d in cands:
        if all((d / t).is_file() and os.access(d / t, os.X_OK) for t in need):
            return d
    sys.exit(
        "missing mlir-opt/mlir-translate/mlir-runner\n"
        "Run on the login node, or set MLIR_BIN=/path/to/llvm/bin"
    )


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = kw.pop("env", None) or os.environ.copy()
    env.update(SINGLE_THREAD_ENV)
    return subprocess.run(cmd, text=True, capture_output=True, env=env, **kw)


def instantiate(src: Path, subs: dict[str, str], tag: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    for key, val in subs.items():
        text = text.replace(key, val)
    dst = OUT / f"{src.name.replace('.mlir.in', '').replace('.mlir', '')}_{tag}.mlir"
    dst.write_text(text)
    return dst


def lower(opt: Path, src: Path, dst: Path, pipeline: list[str],
          transform: bool = False) -> None:
    cmd = [str(opt), str(src)]
    if transform:
        cmd += ["--transform-interpreter",
                "--test-transform-dialect-erase-schedule"]
    cmd += pipeline
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or f"mlir-opt failed: {src}")[:600])
    dst.write_text(p.stdout)


_RUNNER_FLAGS: dict[str, list[str]] = {}


def runner_flags(runner: Path) -> list[str]:
    """mlir-runner defaults to -O0. Ask for -O3 and link the runtime utils.

    The flag is spelled `--O3` in mlir-runner's own option table (it is not
    clang's `-O<n>`), so probe --help instead of guessing.
    """
    cached = _RUNNER_FLAGS.get(str(runner))
    if cached is not None:
        return cached
    flags: list[str] = []
    help_text = run([str(runner), "--help"]).stdout or ""
    if "--O3" in help_text:
        flags.append("--O3")
    elif "opt-level" in help_text:
        flags.append("--opt-level=3")
    lib = runner.parent.parent / "lib"
    libs = [lib / "libmlir_runner_utils.so", lib / "libmlir_c_runner_utils.so"]
    found = [str(p) for p in libs if p.is_file()]
    if found:
        flags.append(f"-shared-libs={','.join(found)}")
    _RUNNER_FLAGS[str(runner)] = flags
    return flags


def run_timed_ir(runner: Path, llvm_mlir: Path, timeout: int) -> list[float]:
    """Run a bench whose @main prints: seconds, then one or more checksums.

    Returns every number printed, so a row can police as much of the result as
    its kernel reports.

    @main is `void`: with -entry-point-result=i32 mlir-runner also prints the
    return value, and because it goes through llvm::outs() while printF64 goes
    through C stdio it lands *before* the timing on a pipe.
    """
    cmd = [str(runner), str(llvm_mlir), "-e", "main", "-entry-point-result=void"]
    cmd += runner_flags(runner)
    p = run(cmd, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "mlir-runner failed")[:600])
    nums = []
    for tok in (p.stdout or "").split():
        try:
            nums.append(float(tok))
        except ValueError:
            pass
    if len(nums) < 2:
        raise RuntimeError(f"no timing printed: {(p.stdout or p.stderr)[:300]}")
    return nums


def time_process(runner: Path, llvm_mlir: Path, reps: int, timeout: int) -> float:
    """Old-style whole-process timing (tiny teaching kernels only)."""
    cmd = [str(runner), str(llvm_mlir), "-e", "main", "-entry-point-result=i32"]
    cmd += runner_flags(runner)
    run(cmd, timeout=timeout)
    t0 = time.perf_counter()
    for _ in range(reps):
        p = run(cmd, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout)[:400])
    return (time.perf_counter() - t0) / reps


def gflops(m: int, n: int, k: int, reps: int, seconds: float) -> float:
    return 2.0 * m * n * k * reps / seconds / 1e9


def row(name, m, n, k, dtype, reps, seconds, notes) -> dict:
    return {
        "name": name, "M": m, "N": n, "K": k, "dtype": dtype,
        "flops_per_gemm": 2 * m * n * k, "iters": reps,
        "seconds": seconds if seconds == seconds else float("nan"),
        "gflops": gflops(m, n, k, reps, seconds) if seconds == seconds else float("nan"),
        "notes": notes,
    }


def fail_row(name, m, n, k, dtype, reps, err) -> dict:
    return {
        "name": name, "M": m, "N": n, "K": k, "dtype": dtype,
        "flops_per_gemm": 2 * m * n * k, "iters": reps,
        "seconds": float("nan"), "gflops": float("nan"),
        "notes": str(err)[:300].replace("\n", " "),
    }


# --------------------------------------------------------------------------
# production-size MLIR rows (1 core, in-IR timing, -O3)
# --------------------------------------------------------------------------
M, N, K = 128, 4096, 4096


def bench_prod_mlir(bin_dir: Path, dtype: str) -> list[dict]:
    opt, runner = bin_dir / "mlir-opt", bin_dir / "mlir-runner"
    timeout = int(os.environ.get("PROD_TIMEOUT", "1200"))
    naive_reps = int(os.environ.get("PROD_REPS_NAIVE", "1"))
    reps = int(os.environ.get("PROD_REPS", "3"))
    # expect=None means the row prints a checksum we do not police.
    specs = [
        ("matmul.mlir style (naive loops)", BENCH / "prod_naive.mlir.in",
         naive_reps, CPU_PIPELINE, False, "no tiling, no vectorization", None),
        ("vector_contract.mlir style (8x8)", BENCH / "prod_contract.mlir.in",
         reps, VECTOR_MEMREF_PIPELINE, False, "8x8 vector.contract tiles", None),
        ("portable (tile+tile+vectorize)", BENCH / "prod_portable.mlir.in",
         reps, VECTORIZED_CPU_PIPELINE, True, "[32,64,64]+[4,16,1]+vectorize",
         None),
    ]
    rows = []
    for label, src, r, pipeline, transform, note, expect in specs:
        tag = f"{dtype}_r{r}"
        name = f"MLIR {label}"
        try:
            inst = instantiate(src, {"$TY": dtype, "$REPS": str(r)}, tag)
            llvm_mlir = OUT / f"{inst.stem}.llvm.mlir"
            print(f"  lower {label} [{dtype}] …", flush=True)
            lower(opt, inst, llvm_mlir, pipeline, transform=transform)
            print(f"  run   {label} [{dtype}] reps={r} …", flush=True)
            nums = run_timed_ir(runner, llvm_mlir, timeout)
            secs = nums[0]
            if expect is not None:
                chk, bad = nums[1], (nums[2] if len(nums) > 2 else None)
                if chk != expect:
                    rows.append(fail_row(
                        name, M, N, K, dtype, r,
                        f"FAILED CHECK: C[0][0]={chk:.0f}, expected "
                        f"K*(1+reps)={expect}"))
                    continue
                if bad:
                    rows.append(fail_row(
                        name, M, N, K, dtype, r,
                        f"FAILED CHECK: {bad:.0f} of {M * N} C elements are "
                        f"not K*(1+reps)={expect}"))
                    continue
                note = f"{note}; all {M * N} C elems = {expect}"
            rows.append(row(name, M, N, K, dtype, r, secs, note))
        except Exception as e:  # noqa: BLE001
            rows.append(fail_row(name, M, N, K, dtype, r, e))
    return rows


# --------------------------------------------------------------------------
# library rows (PACE TPP, torch) — also 1 core
# --------------------------------------------------------------------------
PACE_CODE = r"""
import os, sys, time
os.environ.setdefault("LIBXSMM_TARGET", "cpx")
os.environ["PACE_INSPECT"] = "0"
os.environ.pop("LIBXSMM_VERBOSE", None)
sys.path.insert(0, os.environ["PACE_LEARN"])
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import pace  # noqa: F401
from pace.ops.enum import BackendType, DataType
from pace.ops.linear import Linear

m, n, k, reps = map(int, sys.argv[1:5])
layer = Linear(k, n, bias=False, dtype=DataType.BFLOAT16, backend_impl=BackendType.TPP)
layer.weight.copy_(torch.randn(n, k, dtype=torch.bfloat16))
layer.backend.preprocess(layer)
x = torch.randn(m, k, dtype=torch.bfloat16)
for _ in range(3):
    layer(x)
t0 = time.perf_counter()
for _ in range(reps):
    layer(x)
dt = (time.perf_counter() - t0) / reps
print(f"BENCH {dt:.6e}", flush=True)
"""

TORCH_CODE = r"""
import os, sys, time
import torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
m, n, k, reps = map(int, sys.argv[1:5])
dt_name = sys.argv[5]
td = torch.bfloat16 if dt_name == "bf16" else torch.float32
a = torch.randn(m, k, dtype=td)
b = torch.randn(k, n, dtype=td)
for _ in range(3):
    a @ b
t0 = time.perf_counter()
for _ in range(reps):
    a @ b
dt = (time.perf_counter() - t0) / reps
print(f"BENCH {dt:.6e}", flush=True)
"""


def _python() -> str:
    vpy = REPO / ".venv" / "bin" / "python"
    return str(vpy) if vpy.is_file() else sys.executable


def _bench_child(code: str, args: list[str], threads: int, timeout: int = 600) -> float:
    env = os.environ.copy()
    env["PACE_LEARN"] = str(REPO)
    env["PACE_INSPECT"] = "0"
    env.update(SINGLE_THREAD_ENV)
    env["OMP_NUM_THREADS"] = str(threads)
    p = subprocess.run([_python(), "-c", code] + args, text=True,
                       capture_output=True, cwd=str(REPO), env=env, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-400:])
    for line in (p.stdout or "").splitlines():
        if line.startswith("BENCH "):
            return float(line.split()[1])
    raise RuntimeError((p.stderr or p.stdout or "no BENCH line")[-400:])


def bench_libraries(threads_list: list[int]) -> list[dict]:
    rows = []
    for th in threads_list:
        tag = f"{th} core" if th == 1 else f"{th} cores"
        try:
            print(f"  PACE TPP bf16 ({tag}) …", flush=True)
            dt = _bench_child(PACE_CODE, [str(M), str(N), str(K), "5"], th)
            rows.append(row(f"PACE TPP bf16 ({tag})", M, N, K, "bf16", 1, dt,
                            "VNNI pack + libXSMM ukernel (vdpbf16ps)"))
        except Exception as e:  # noqa: BLE001
            rows.append(fail_row(f"PACE TPP bf16 ({tag})", M, N, K, "bf16", 1, e))
    for dt_name in ("f32", "bf16"):
        try:
            print(f"  torch {dt_name} (1 core) …", flush=True)
            dt = _bench_child(TORCH_CODE, [str(M), str(N), str(K), "5", dt_name], 1)
            rows.append(row(f"torch matmul {dt_name} (1 core)", M, N, K, dt_name, 1, dt,
                            "reference: vendor BLAS-style kernel"))
        except Exception as e:  # noqa: BLE001
            rows.append(fail_row(f"torch matmul {dt_name} (1 core)", M, N, K, dt_name, 1, e))
    return rows


# --------------------------------------------------------------------------
# tiny teaching kernels (unchanged sizes)
# --------------------------------------------------------------------------
def bench_tiny(bin_dir: Path) -> list[dict]:
    opt, runner = bin_dir / "mlir-opt", bin_dir / "mlir-runner"
    specs = [
        ("matmul_2x2", BENCH / "bench_matmul_2x2.mlir.in", 2, 2, 2,
         int(os.environ.get("ITERS_2X2", "400000")), CPU_PIPELINE, False, "naive loops"),
        ("vector_contract_8x8", BENCH / "bench_contract_8x8.mlir.in", 8, 8, 8,
         int(os.environ.get("ITERS_8X8", "200000")), VECTOR_PIPELINE, False, "vector"),
        ("portable_16x16", BENCH / "bench_portable_16x16.mlir.in", 16, 16, 16,
         int(os.environ.get("ITERS_16X16", "80000")), CPU_PIPELINE, True, "tiled"),
    ]
    rows = []
    for name, src, m, n, k, iters, pipeline, transform, note in specs:
        try:
            inst = instantiate(src, {"$ITERS": str(iters)}, f"i{iters}")
            llvm_mlir = OUT / f"{inst.stem}.llvm.mlir"
            lower(opt, inst, llvm_mlir, pipeline, transform=transform)
            dt = time_process(runner, llvm_mlir, reps=3, timeout=600)
            rows.append(row(name, m, n, k, "f32", iters, dt, note))
        except Exception as e:  # noqa: BLE001
            rows.append(fail_row(name, m, n, k, "f32", iters, e))
    return rows


def fmt(r: dict) -> str:
    gf = r["gflops"]
    sec = r["seconds"]
    gf_s = f"{gf:10.2f}" if gf == gf else "       n/a"
    # `seconds` is the total timed region; reps GEMMs happened inside it.
    sec_s = f"{sec / max(r['iters'], 1) * 1e3:10.2f}" if sec == sec else "       n/a"
    return (
        f"{r['name']:<38s} {r['M']:5d} {r['N']:5d} {r['K']:5d} "
        f"{r['dtype']:<5s} {r['flops_per_gemm']:12d} {r['iters']:6d} "
        f"{sec_s} {gf_s}  {r['notes']}"
    )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prod", action="store_true",
                    help="128x4096x4096 on one core, in-IR timing")
    ap.add_argument("--dtype", default="both", choices=["f32", "bf16", "both"])
    ap.add_argument("--threads", default="1",
                    help="comma list for the library rows, e.g. 1,8")
    args = ap.parse_args()

    bin_dir = find_bin()
    os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"mlir tools: {bin_dir}")
    print("GEMM FLOPs = 2*M*N*K\n")

    rows: list[dict] = []
    if args.prod:
        dtypes = ["f32", "bf16"] if args.dtype == "both" else [args.dtype]
        for d in dtypes:
            rows.extend(bench_prod_mlir(bin_dir, d))
        rows.extend(bench_libraries([int(t) for t in args.threads.split(",")]))
        outp = OUT / "flops_prod.txt"
        how = [
            f"Every row is M={M} N={N} K={K}, FLOPs/gemm = {2 * M * N * K}.",
            "All rows single core (OMP/MKL threads = 1) unless the name says otherwise.",
            "MLIR rows time only the GEMM (rtclock in @main); JIT and fills excluded.",
            "mlir-runner JITs at --O3; its default --O0 costs the portable row 1.7x.",
            "The naive row is identical at --O0 and --O3: it stalls on B's stride-4096",
            "  column walk, so JIT codegen quality cannot help it.",
            "Plain MLIR bf16 lowers arith on bf16 to extf/mulf/truncf, which is why",
            "  the bf16 portable row is no faster than the f32 one.",
            "The vdpbf16ps row (pack + 12x32 tile) needs LLVM 22 in Docker:",
            "  chain/bench_in_container.sh -> chain/out/prod12_flops.txt.",
        ]
    else:
        rows.extend(bench_tiny(bin_dir))
        rows.extend(bench_libraries([1]))
        outp = OUT / "flops.txt"
        how = [
            "Tiny teaching kernels. For the like-to-like production run:",
            "  ./bench_flops.sh --prod   ->  out/flops_prod.txt",
        ]

    header = (
        f"{'path':<38s} {'M':>5s} {'N':>5s} {'K':>5s} "
        f"{'ty':<5s} {'FLOPs/gemm':>12s} {'reps':>6s} "
        f"{'ms/gemm':>10s} {'GFLOP/s':>10s}  notes"
    )
    lines = ["GEMM FLOP comparison. GFLOP/s = 2*M*N*K*reps / seconds / 1e9", "",
             header, "-" * len(header)]
    lines += [fmt(r) for r in rows]
    lines += ["", "How to read it:"] + how + ["", f"host tools: {bin_dir}"]
    text = "\n".join(lines) + "\n"
    outp.write_text(text)
    print()
    print(text)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
