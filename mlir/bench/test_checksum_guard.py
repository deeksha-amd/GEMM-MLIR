#!/usr/bin/env python3
"""Negative control for the bf16 ukernel's correctness check.

A GEMM benchmark that silently does less work looks *faster*, so the checksum
is the only thing standing between us and a fast-but-wrong number.  This
deliberately breaks the kernel and asserts the bench would reject it.

The mutation drops the last row-block from the main tile loop.  C[0][0] is in
the first block and stays correct, so this is exactly the class of bug that a
single-element checksum cannot see -- only the full-matrix scan catches it.

  python3 bench/test_checksum_guard.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench_flops as bf  # noqa: E402

REPS = 3


def main() -> int:
    src = (bf.BENCH / "prod_bf16_ukernel.mlir.in").read_text()
    src = src.replace("$REPS", str(REPS))
    nib = bf.M // 12  # main tile row-blocks for the frozen MR=12 kernel
    broken, n = re.subn(rf"%cMB = arith\.constant {nib} : index",
                        f"%cMB = arith.constant {nib - 1} : index", src)
    if n != 1:
        print("could not apply the mutation; has the register tile changed?")
        return 1

    bin_dir = bf.find_bin()
    p = bf.OUT / "negctl_bf16_ukernel.mlir"
    low = bf.OUT / "negctl_bf16_ukernel.llvm.mlir"
    p.write_text(broken)
    bf.lower(bin_dir / "mlir-opt", p, low, bf.BF16_UKERNEL_PIPELINE)
    secs, chk, bad = bf.run_timed_ir(bin_dir / "mlir-runner", low, 600)[:3]

    expect = bf.K * (1 + REPS)
    gf = bf.gflops(bf.M, bf.N, bf.K, REPS, secs)
    print(f"broken kernel reports {gf:.1f} GFLOP/s -- faster than the real one,")
    print(f"because it skips one 12-row block of the {bf.M} rows.")
    print(f"C[0][0] = {chk:.0f}, expected {expect}: "
          f"{'PASSES (!)' if chk == expect else 'caught'}")
    print(f"full-matrix scan: {bad:.0f} of {bf.M * bf.N} C elements wrong")

    ok = chk == expect and bad > 0
    print("\n" + ("PASS: the single-element checksum is fooled and the "
                  "full-matrix scan is what rejects the row."
                  if ok else "FAIL: negative control did not behave as designed"))
    for f in (p, low):
        f.unlink(missing_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
