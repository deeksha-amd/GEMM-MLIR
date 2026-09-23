#!/usr/bin/env python3
"""Sweep register tiles / packed layouts / loop order for the bf16 ukernel.

Writes out/bf16_tuning.txt.  Every configuration is checksum-verified: A and B
are all bf16 1.0, so C[0][0] must be exactly K*(1+REPS).  A configuration that
does not match is reported FAILED CHECK and never considered the winner.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_flops as bf  # noqa: E402
from gen_bf16_ukernel import gen  # noqa: E402

M, N, K = bf.M, bf.N, bf.K
REPS = int(os.environ.get("TUNE_REPS", "10"))
TRIALS = int(os.environ.get("TUNE_TRIALS", "3"))
PIN = os.environ.get("TUNE_PIN", "8")

CONFIGS = [
    # (MR, NR, a_layout, b_layout, order)
    # -- the four tiles asked for, on the draft's flat (unpacked) B --------
    (4, 16, "flat", "flat", "jouter"),    # the original draft
    (8, 16, "flat", "flat", "jouter"),
    (4, 32, "flat", "flat", "jouter"),
    (6, 32, "flat", "flat", "jouter"),
    # -- isolate the B panel pack, then the A panel pack -------------------
    (4, 16, "flat", "panel", "jouter"),
    (4, 16, "panel", "panel", "jouter"),
    (8, 32, "flat", "panel", "jouter"),
    # -- the same four tiles on packed panels ------------------------------
    (4, 16, "panel", "panel", "jouter"),
    (8, 16, "panel", "panel", "jouter"),
    (4, 32, "panel", "panel", "jouter"),
    (6, 32, "flat", "panel", "jouter"),   # 128 % 6 != 0 -> flat A + tail tile
    (8, 32, "panel", "panel", "jouter"),
    # -- push past the plateau ---------------------------------------------
    (2, 64, "panel", "panel", "jouter"),
    (4, 64, "panel", "panel", "jouter"),
    (8, 64, "panel", "panel", "jouter"),
    (16, 16, "panel", "panel", "jouter"),
    (16, 32, "panel", "panel", "jouter"),
    (12, 32, "flat", "panel", "jouter"),
    # -- loop-order swap ----------------------------------------------------
    (8, 16, "panel", "panel", "iouter"),
    (4, 32, "panel", "panel", "iouter"),
    (8, 32, "panel", "panel", "iouter"),
    (4, 64, "panel", "panel", "iouter"),
]


PEAK_ITERS = int(os.environ.get("TUNE_PEAK_ITERS", "20000000"))
PEAK_PIPELINE = [
    "--convert-scf-to-cf",
    "--convert-vector-to-llvm=enable-x86vector",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-index-to-llvm",
    "--reconcile-unrealized-casts",
]


def measure_peak(bin_dir: Path):
    """How many vdpbf16ps/s this core retires with no memory traffic at all.

    This is the hard ceiling for the GEMM row: 16 independent accumulators and
    loop-invariant register operands, so nothing but issue rate is measured.
    """
    try:
        opt, runner = bin_dir / "mlir-opt", bin_dir / "mlir-runner"
        src = bf.OUT / "peak_dpbf16ps.mlir"
        src.write_text((Path(__file__).resolve().parent /
                        "peak_dpbf16ps.mlir.in").read_text()
                       .replace("$ITERS", str(PEAK_ITERS)))
        low = bf.OUT / "peak_dpbf16ps.llvm.mlir"
        bf.lower(opt, src, low, PEAK_PIPELINE)
        cmd = ["taskset", "-c", PIN, str(runner), str(low), "-e", "main",
               "-entry-point-result=void"] + bf.runner_flags(runner)
        secs = min(float(bf.run(cmd, timeout=600).stdout.split()[0])
                   for _ in range(3))
        n = PEAK_ITERS * 16  # 16 vdpbf16ps per loop iteration
        return n / secs / 1e9, n * 64 / secs / 1e9  # 64 flops per vdpbf16ps
    except Exception as e:  # noqa: BLE001
        print(f"  peak probe failed: {e}", flush=True)
        return None


def tag(cfg) -> str:
    mr, nr, al, bl, order = cfg
    return f"mr{mr}_nr{nr}_a{al}_b{bl}_{order}"


def main() -> None:
    bin_dir = bf.find_bin()
    opt, runner, translate = (bin_dir / "mlir-opt", bin_dir / "mlir-runner",
                              bin_dir / "mlir-translate")
    llc = bin_dir / "llc"
    bf.OUT.mkdir(parents=True, exist_ok=True)
    tune = bf.OUT / "tune"
    tune.mkdir(exist_ok=True)
    expect = K * (1 + REPS)

    results = []
    seen: set = set()
    for cfg in CONFIGS:
        if cfg in seen:
            continue
        seen.add(cfg)
        mr, nr, al, bl, order = cfg
        t = tag(cfg)
        src = tune / f"uk_{t}.mlir"
        low = tune / f"uk_{t}.llvm.mlir"
        try:
            src.write_text(gen(mr, nr, al, bl, order).replace("$REPS", str(REPS)))
            bf.lower(opt, src, low, bf.BF16_UKERNEL_PIPELINE)

            # assembly proof: the intrinsic must survive to a real vdpbf16ps
            ll, asm = tune / f"uk_{t}.ll", tune / f"uk_{t}.s"
            bf.run([str(translate), "--mlir-to-llvmir", str(low), "-o", str(ll)])
            bf.run([str(llc), "-O3", "-mcpu=native", "-mattr=+avx512bf16",
                    str(ll), "-o", str(asm)])
            text = asm.read_text()
            n_dp = sum(1 for l in text.splitlines() if "\tvdpbf16ps" in l)
            n_bc = sum(1 for l in text.splitlines()
                       if "\tvbroadcastss" in l or "\tvpbroadcastd" in l)

            cmd = ["taskset", "-c", PIN, str(runner), str(low), "-e", "main",
                   "-entry-point-result=void"] + bf.runner_flags(runner)
            # best of TRIALS: a stray migration or a boost-clock dip only ever
            # makes a run look slower, so the best trial is the honest one
            secs, chk, ok = float("inf"), float("nan"), True
            for _ in range(TRIALS):
                p = bf.run(cmd, timeout=1200)
                if p.returncode != 0:
                    raise RuntimeError((p.stderr or p.stdout)[:300])
                nums = [float(x) for x in p.stdout.split()]
                secs = min(secs, nums[0])
                chk = nums[1]
                # nums[2] = how many of the M*N C elements are wrong
                ok = ok and chk == expect and nums[2] == 0
            gf = bf.gflops(M, N, K, REPS, secs)
            results.append((t, gf, secs / REPS * 1e3, chk, ok, n_dp, n_bc, ""))
            print(f"  {t:<34s} {gf:8.2f} GF/s  chk={chk:.0f} "
                  f"{'ok' if ok else 'FAILED CHECK'}  dp={n_dp} bc={n_bc}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            results.append((t, float("nan"), float("nan"), float("nan"),
                            False, 0, 0, str(e)[:200].replace("\n", " ")))
            print(f"  {t:<34s} ERROR {str(e)[:160]}", flush=True)

    good = [r for r in results if r[4] and r[1] == r[1]]
    best = max(good, key=lambda r: r[1]) if good else None

    hdr = (f"{'config (MR x NR, layouts, order)':<34s} {'GFLOP/s':>9s} "
           f"{'ms/gemm':>9s} {'C[0][0]':>10s} {'check':>12s} "
           f"{'dpbf16ps':>9s} {'bcast':>6s}")
    lines = [
        f"bf16 VNNI + vdpbf16ps ukernel tuning, M={M} N={N} K={K}, "
        f"1 core (taskset -c {PIN}), reps={REPS}, best of {TRIALS} trials",
        f"checksum: A,B all bf16 1.0 => C[0][0] must be K*(1+reps) = {expect}",
        "dpbf16ps / bcast are static instruction counts in the llc -O3 output",
        "", hdr, "-" * len(hdr),
    ]
    for t, gf, ms, chk, ok, dp, bc, err in results:
        if gf != gf:
            lines.append(f"{t:<34s} {'n/a':>9s} {'n/a':>9s} {'n/a':>10s} "
                         f"{'ERROR':>12s} {0:>9d} {0:>6d}  {err}")
        else:
            lines.append(f"{t:<34s} {gf:9.2f} {ms:9.2f} {chk:10.0f} "
                         f"{'ok' if ok else 'FAILED CHECK':>12s} "
                         f"{dp:9d} {bc:6d}")
    peak = measure_peak(bin_dir)
    if peak:
        lines += ["", f"ISA ceiling (bench/peak_dpbf16ps.mlir.in): 16 independent",
                  "accumulators, register-only operands, zero memory traffic:",
                  f"  {peak[0]:.3f} G vdpbf16ps/s  =  {peak[1]:.1f} GFLOP/s bf16"]
    if best:
        lines += ["", f"WINNER: {best[0]}  {best[1]:.2f} GFLOP/s"
                  + (f"  ({100 * best[1] / peak[1]:.0f}% of ISA ceiling)"
                     if peak else "")]
    out = bf.OUT / "bf16_tuning.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
