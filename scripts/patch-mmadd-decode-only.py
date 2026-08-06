#!/usr/bin/env python3
"""Patch tip libggml-sycl.so: MUL_MAT+ADD fuse only when x.ne[1]==1 (decode).

Root cause of quality break: any-batch fuse (prefill ne11>>1) takes GEMM/MMVQ
paths that drop or mis-apply residual addends → PPL 1e5–1e6. Decode ne11==1
keeps reorder-MMVQ epilogue with correct addends; wikitext-2 PPL matches
quality-safe (~12.6) and golden OK.

Binary site (treebeard-base-control tip, Jul 30 10:02):
  ggml_sycl_fuse_mul_mat_add @ 0x22952f:
    was: cmp $0x20, %rax; ja single_ADD_path   # ne[1]>32 → single still fuses
    now: cmp $0x1,  %rax; ja return_0          # ne[1]>1  → reject entire fuse

Usage:
  python3 scripts/patch-mmadd-decode-only.py path/to/libggml-sycl.so.0.17.0
"""
from __future__ import annotations
import sys
from pathlib import Path

OLD = bytes.fromhex("4883f8200f8732010000")  # cmp $0x20,%rax; ja +0x132
NEW = bytes.fromhex("4883f8010f8790ffffff")  # cmp $0x1,%rax;  ja -0x70 → reject

def patch(path: Path) -> None:
    data = bytearray(path.read_bytes())
    # Prefer known VA offset (identity-mapped in this ELF)
    off = 0x22952f
    if data[off : off + 10] == NEW:
        print(f"already patched: {path}")
        return
    if data[off : off + 10] == OLD:
        data[off : off + 10] = NEW
        path.write_bytes(data)
        print(f"patched VA 0x22952f: {path}")
        return
    # Fallback: unique signature search
    idxs = []
    i = 0
    while True:
        j = data.find(OLD, i)
        if j < 0:
            break
        idxs.append(j)
        i = j + 1
    if len(idxs) == 1:
        j = idxs[0]
        data[j : j + 10] = NEW
        path.write_bytes(data)
        print(f"patched sig@{hex(j)}: {path}")
        return
    raise SystemExit(
        f"refuse patch {path}: expected OLD at 0x22952f got {data[off:off+10].hex()}; "
        f"sig matches={len(idxs)}"
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    patch(Path(sys.argv[1]))
