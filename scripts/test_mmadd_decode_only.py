#!/usr/bin/env python3
"""Structural + binary-patch tests for decode-only MUL_MAT+ADD reclaim.

Drives the real shipped patcher (scripts/patch-mmadd-decode-only.py) and asserts
the quality-safe decode-only gate exists in the source fullsnippet used for
future source restores. Run from lx root:

  python3 scripts/test_mmadd_decode_only.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHER = ROOT / "scripts" / "patch-mmadd-decode-only.py"
SNIPPET = ROOT / "patches" / "0044-control-mul-mat-add-add-decode.fullsnippet.cpp"
TIP_LIB = Path(
    "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/"
    "build-base-control/bin/libggml-sycl.so.0.17.0"
)
CHAMP_LIB = Path(
    "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/"
    "build-mmadd-decode/bin/libggml-sycl.so.0.17.0"
)

OLD = bytes.fromhex("4883f8200f8732010000")
NEW = bytes.fromhex("4883f8010f8790ffffff")
OFF = 0x22952F


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def ok(msg: str) -> None:
    print(f"OK: {msg}")


def test_snippet_decode_only_gate() -> None:
    text = SNIPPET.read_text()
    if "ne[1] != 1" not in text and "ne11==1" not in text:
        fail("fullsnippet missing decode-only (ne11==1) gate")
    if "ENABLE_MUL_MAT_ADD_ANY_BATCH" not in text:
        fail("fullsnippet missing ANY_BATCH opt-in kill-switch")
    if "ggml_sycl_fuse_mul_mat_add" not in text:
        fail("fullsnippet missing fuse_mul_mat_add")
    ok("fullsnippet has decode-only gate + any-batch opt-in")


def test_patcher_on_copy() -> None:
    if not TIP_LIB.is_file():
        fail(f"missing tip lib {TIP_LIB}")
    if not PATCHER.is_file():
        fail(f"missing patcher {PATCHER}")
    with tempfile.TemporaryDirectory() as td:
        lib = Path(td) / "libggml-sycl.so.0.17.0"
        shutil.copy2(TIP_LIB, lib)
        raw = lib.read_bytes()
        if raw[OFF : OFF + 10] != OLD:
            fail(f"tip lib unexpected bytes at {hex(OFF)}: {raw[OFF:OFF+10].hex()}")
        r = subprocess.run(
            [sys.executable, str(PATCHER), str(lib)],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            fail(f"patcher failed: {r.stderr or r.stdout}")
        patched = lib.read_bytes()
        if patched[OFF : OFF + 10] != NEW:
            fail(f"after patch expected NEW got {patched[OFF:OFF+10].hex()}")
        # idempotent
        r2 = subprocess.run(
            [sys.executable, str(PATCHER), str(lib)],
            capture_output=True,
            text=True,
            check=False,
        )
        if r2.returncode != 0:
            fail(f"idempotent patcher failed: {r2.stderr or r2.stdout}")
        if b"already patched" not in (r2.stdout + r2.stderr).encode() and "already patched" not in (
            r2.stdout + r2.stderr
        ):
            # still ok if silent and bytes stay NEW
            if lib.read_bytes()[OFF : OFF + 10] != NEW:
                fail("idempotent re-patch corrupted bytes")
        ok("patcher applies OLD→NEW and is re-run safe")


def test_champion_lib_is_patched() -> None:
    if not CHAMP_LIB.is_file():
        fail(f"missing champion lib {CHAMP_LIB}")
    b = CHAMP_LIB.read_bytes()
    if b[OFF : OFF + 10] != NEW:
        fail(f"champion lib not patched at {hex(OFF)}: {b[OFF:OFF+10].hex()}")
    ok("champion build-mmadd-decode lib carries decode-only patch")


def test_env_defaults() -> None:
    env = (ROOT / "env.sh").read_text()
    if "build-mmadd-decode" not in env:
        fail("env.sh does not default LX_BIN to build-mmadd-decode")
    if 'DISABLE_MUL_MAT_ADD_FUSE:-0' not in env and 'DISABLE_MUL_MAT_ADD_FUSE:-0}"' not in env:
        # allow either form
        if 'MUL_MAT_ADD_FUSE:-0' not in env:
            fail("env.sh should default MUL_MAT_ADD enabled (decode-only champion)")
    if 'DISABLE_MOE_DUAL_DOWN:-1' not in env and "DISABLE_MOE_DUAL_DOWN:-1" not in env:
        fail("env.sh should keep dual_down killed")
    ok("env.sh defaults match quality-safe champion posture")


def main() -> None:
    test_snippet_decode_only_gate()
    test_patcher_on_copy()
    test_champion_lib_is_patched()
    test_env_defaults()
    print("ALL PASS")


if __name__ == "__main__":
    main()
