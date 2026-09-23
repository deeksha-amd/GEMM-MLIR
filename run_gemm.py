#!/usr/bin/env python3
"""Run matmul (GEMM) on AMD EPYC using PACE kernels, and dump TPP layers.

PACE Linear is GEMM:

    Y = X @ W.T + b
    X: [M, K]   activation
    W: [N, K]   weight  (out_features x in_features)
    Y: [M, N]

TPPLinear layers (printed when PACE_INSPECT is not 0):
  1. Python pack   — 2D W -> 5D VNNI blocks
  2. C++ tiles     — Y and the BrgemmTPP tile schedule
  3. libXSMM JIT   — verbose log + dumped kernel files (search vdpbf16ps)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# libXSMM reads these at first kernel JIT. Set them before importing pace.
_ROOT = Path(__file__).resolve().parent
_DUMP_DIR = Path(os.environ.get("PACE_KERNEL_DUMP", str(_ROOT / "kernel_dump")))
_INSPECT = os.environ.get("PACE_INSPECT", "1") != "0"

os.environ.setdefault("LIBXSMM_TARGET", "cpx")
if _INSPECT:
    _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    # Negative VERBOSE dumps each JIT blob as a raw binary (named like the kernel).
    # Positive VERBOSE only prints a dispatch table at process exit — no files.
    os.environ.setdefault("LIBXSMM_VERBOSE", "-1")

import torch
import torch.nn.functional as F

import pace  # noqa: F401  — loads libpace_cpp.so
from pace.ops.enum import BackendType, DataType, OperatorType
from pace.ops.linear import Linear
from pace.ops.registry import backend_registry


def cpu_isa() -> None:
    flags = open("/proc/cpuinfo").read()
    print("CPU ISA:")
    for name in (
        "avx512f",
        "avx512_bf16",
        "avx512_vnni",
        "amx_tile",
        "amx_bf16",
    ):
        print(f"  {name:16s} {'yes' if name in flags else 'no'}")
    print(f"  OMP_NUM_THREADS  {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    print(f"  LIBXSMM_TARGET   {os.environ.get('LIBXSMM_TARGET', '<unset>')}")
    print(f"  LIBXSMM_VERBOSE  {os.environ.get('LIBXSMM_VERBOSE', '<unset>')}  (-1 dumps JIT binaries)")
    print()


def available_linear_backends() -> list[tuple[BackendType, DataType]]:
    return list(backend_registry.get_available_backends(OperatorType.LINEAR))


@contextmanager
def _capture_c_stderr(path: Path):
    """Capture C fprintf(stderr) from libXSMM, not only Python sys.stderr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    saved = os.dup(2)
    sys.stderr.flush()
    os.dup2(fd, 2)
    try:
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(fd)


def _print_tile_schedule(m: int, k: int, n: int, packed: torch.Tensor) -> None:
    """Decode C++ libxsmmlinear_kernel symbols from the packed 5D weight."""
    if packed.dim() != 5:
        print("  packed W is still 2D — TPP fast path did NOT engage")
        return
    nk, nc, k_inner, hk, vnni = packed.shape
    hc = k // nc
    bsb = 64
    rem = m % bsb
    print("  C++ tile schedule (libxsmmlinear_kernel):")
    print(f"    BS (tokens M)     = {m}")
    print(f"    BSb (token tile)  = {bsb}   remainder = {rem}")
    print(f"    C  (K in)         = {k}")
    print(f"    Nc x Hc (K split) = {nc} x {hc}")
    print(f"    Nk x Hk (N split) = {nk} x {hk}")
    print(f"    VNNI pack         = {vnni}  (2 = bf16 pairs)")
    print(f"    BrgemmTPP microkernel  M={bsb}  N={hk}  K={hc}  (tokens x out-block x k-block)")
    print(f"    loop scheme       = {os.environ.get('PACE_GEMM_LOOP_SCHEME', 'aCB')}")
    print(f"    Ncb (K tiles/BRGEMM) = {nc}  (all K unless PACE_LARGE_CACHE_OPT)")


def _list_dump_files(*dirs: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    skip_suffix = {".py", ".pyc", ".log", ".s", ".md", ".txt"}
    skip_names = {"run_inspect.log", "libxsmm_verbose.log"}
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_file() or p.name in skip_names or p.suffix in skip_suffix:
                continue
            if not p.name.startswith("libxsmm"):
                continue
            if p.stat().st_size < 64:
                continue
            if p.name.endswith(".s"):
                continue
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def _disassemble_jit(bin_path: Path) -> Path | None:
    """Raw libXSMM dumps are code bytes, not ELF. GNU objdump -b binary."""
    dump = shutil.which("objdump")
    if not dump:
        return None
    out_path = bin_path.with_suffix(bin_path.suffix + ".s")
    cmd = [
        dump,
        "-D",
        "-b",
        "binary",
        "-m",
        "i386",
        "-M",
        "x86-64",
        str(bin_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout or proc.stderr
    if not text:
        return None
    out_path.write_text(text)
    return out_path


def _read_jit_dumps(files: list[Path], log_path: Path) -> None:
    print("  captured stderr during GEMM:", log_path)
    if log_path.is_file() and log_path.stat().st_size:
        for ln in log_path.read_text(errors="replace").splitlines()[:30]:
            print(f"    {ln}")
    else:
        print("    (empty — expected: dumps are files, stats print at process exit)")

    print("  dumped kernel files (LIBXSMM_VERBOSE=-1):")
    if not files:
        print("    (none — need LIBXSMM_VERBOSE=-1, cwd writable)")
        return

    gemm = [p for p in files if p.suffix == ".mxm" and "br3" in p.name]
    other = [p for p in files if p not in gemm]
    print(f"    {len(gemm)} BRGEMM .mxm  +  {len(other)} helpers (tile-config / meltw)")
    for p in other[:8]:
        print(f"    helper  {p.name}  ({p.stat().st_size} B)")
    if len(other) > 8:
        print(f"    ... {len(other) - 8} more helpers")

    for p in gemm:
        print(f"    GEMM    {p.name}  ({p.stat().st_size} B)")
        asm_path = _disassemble_jit(p)
        if asm_path is None:
            print("      (install binutils objdump to disassemble)")
            continue
        asm = asm_path.read_text(errors="replace")
        print(f"      disassembly -> {asm_path.name}")
        dp = [ln for ln in asm.splitlines() if "vdpbf16ps" in ln.lower()]
        amx = [ln for ln in asm.splitlines() if "tdpbf16ps" in ln.lower()]
        if dp:
            print("      BF16 AVX-512 kernel (vdpbf16ps), not AMX:")
            for ln in dp[:8]:
                print(f"      {ln[:140]}")
        elif amx:
            print("      AMX tile kernel (unexpected on EPYC):")
            for ln in amx[:8]:
                print(f"      {ln[:140]}")
        else:
            print("      no vdpbf16ps in this blob (config/release stub?)")


def inspect_tpp_layers(m: int, k: int, n: int) -> None:
    """Print TPPLinear layers 1 (pack), 2 (Y/tiles), 3 (JIT dump)."""
    print("=" * 72)
    print(f"TPP layer dump  M={m} K={k} N={n}  Y = X @ W.T")
    print(f"kernel_dump dir: {_DUMP_DIR}")
    print("=" * 72)

    torch.manual_seed(0)
    td = torch.bfloat16
    layer = Linear(
        k, n, bias=False, dtype=DataType.BFLOAT16, backend_impl=BackendType.TPP
    )
    w = torch.randn(n, k, dtype=td)
    layer.weight.copy_(w)

    print("\n--- Layer 1: Python pack (TPPLinear.preprocess) ---")
    print(f"  backend           = {layer.backend}")
    print(f"  W before          = {tuple(layer.weight.shape)}  {layer.weight.dtype}")
    print(f"  W[0,:8]           = {layer.weight[0, :8].float().tolist()}")
    layer.backend.preprocess(layer)
    packed = layer.weight.data
    print(f"  W after           = {tuple(packed.shape)}  {packed.dtype}")
    if packed.dim() == 5:
        print("  layout            = [Nk, Nc, 32, Hk, 2]  last dim is VNNI")
        print(
            "  W[0,0,:2,:2,:]    =\n"
            f"    {packed[0, 0, :2, :2, :].float().cpu().numpy()}"
        )
    x = torch.randn(m, k, dtype=td)
    x3 = x
    orig = x.shape[:-1]
    if x3.dim() < 3:
        for _ in range(3 - x3.dim()):
            x3 = x3.unsqueeze(0)
    print(f"  X 2D (script)     = {tuple(x.shape)}")
    print(f"  X 3D (C++ kernel) = {tuple(x3.shape)}  orig_shape={tuple(orig)}")

    print("\n--- Layer 2: C++ tiles + Y (libxsmmlinear_plain) ---")
    _print_tile_schedule(m, k, n, packed)
    log_path = _DUMP_DIR / "libxsmm_verbose.log"
    cwd = Path.cwd()
    try:
        os.chdir(_DUMP_DIR)
        with _capture_c_stderr(log_path):
            y = layer(x)
    finally:
        os.chdir(cwd)

    ref = F.linear(x, w, None)
    max_err = (y.float() - ref.float()).abs().max().item()
    print(f"  Y shape           = {tuple(y.shape)}  {y.dtype}")
    print(f"  Y[0,:8]           = {y[0, :8].float().tolist()}")
    print(f"  ref[0,:8]         = {ref[0, :8].float().tolist()}")
    print(f"  max|Y-ref|        = {max_err:.4e}")

    print("\n--- Layer 3: libXSMM JIT (verbose log + dump files) ---")
    dumps = _list_dump_files(_DUMP_DIR)
    _read_jit_dumps(dumps, log_path)
    print()


def run_one(
    backend: BackendType,
    dtype: DataType,
    m: int,
    k: int,
    n: int,
    bias: bool,
    reps: int,
) -> None:
    torch.manual_seed(0)
    td = dtype.to_torch_dtype()

    layer = Linear(k, n, bias=bias, dtype=dtype, backend_impl=backend)
    w = torch.randn(n, k, dtype=td)
    layer.weight.copy_(w)
    b = None
    if bias:
        b = torch.randn(n, dtype=td)
        layer.bias.copy_(b)

    # TPP / AOCL pack weights into blocked GEMM layout.
    layer.backend.preprocess(layer)

    x = torch.randn(m, k, dtype=td)
    ref = F.linear(x, w, b)
    y = layer(x)

    # BF16 GEMM is not bit-exact vs PyTorch; use a loose atol.
    atol = 2e-2 if td == torch.bfloat16 else 1e-4
    max_err = (y.float() - ref.float()).abs().max().item()
    ok = torch.allclose(y.float(), ref.float(), rtol=1e-2, atol=atol)

    for _ in range(3):
        layer(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        layer(x)
    dt = (time.perf_counter() - t0) / reps
    gflops = (2.0 * m * n * k) / dt / 1e9

    print(
        f"  {backend.name:8s} {dtype.name:9s}  "
        f"Y={tuple(y.shape)}  max|err|={max_err:.4e}  "
        f"{'PASS' if ok else 'FAIL'}  "
        f"{dt*1e3:8.2f} ms  {gflops:8.1f} GFLOP/s  "
        f"backend={layer.backend}"
    )
    if not ok:
        raise SystemExit(f"mismatch on {backend}/{dtype}")


def main() -> None:
    cpu_isa()
    pairs = available_linear_backends()
    print("Registered Linear backends:")
    for b, d in pairs:
        print(f"  {b.name:8s}  {d.name}")
    print()

    if _INSPECT and (BackendType.TPP, DataType.BFLOAT16) in pairs:
        # Small: Y is readable. Large: packing matches the LLM GEMM.
        inspect_tpp_layers(m=32, k=128, n=64)
        inspect_tpp_layers(m=128, k=4096, n=4096)

    wanted = [
        (BackendType.TPP, DataType.BFLOAT16),
        (BackendType.AOCLDLP, DataType.BFLOAT16),
        (BackendType.JIT, DataType.BFLOAT16),
        (BackendType.NATIVE, DataType.BFLOAT16),
    ]
    to_run = [p for p in wanted if p in pairs]
    if not to_run:
        raise SystemExit("no Linear backends registered; is amd-pace installed?")

    print("Correctness  M=32 K=128 N=64  (Y = X @ W.T + b)")
    for backend, dtype in to_run:
        run_one(backend, dtype, m=32, k=128, n=64, bias=True, reps=20)
    print()

    m, k, n = 128, 4096, 4096
    print(f"Throughput  M={m} K={k} N={n}  BF16 GEMM")
    for backend, dtype in to_run:
        if dtype != DataType.BFLOAT16:
            continue
        run_one(backend, dtype, m=m, k=k, n=n, bias=False, reps=10)


if __name__ == "__main__":
    main()
