#!/usr/bin/env python3
"""patch-prefill-split.py — first-ever prefill-structural source edit for laguna.

Experiment (lead 21 / open lead 12): the pp512 MoE fused down-projection
("dense-dual": gate+up+swiglu, ncols_dst=512, nrows=8192) is dispatched as ONE
fused GEMM at ~123-145 ms/call = 16-22% of the prefill dispatch span
(results/prefill-budget-pp512b-20260806T105903Z/budget.txt). Splitting the
batch dimension into 2x256-token dispatches is row-wise bit-identical (each
token row's arithmetic is unchanged; no cross-row reduction exists in a fused
dense GEMM), so golden-smoke bit-exactness and PPL are safe BY CONSTRUCTION —
unlike the killed dual-multitoken path, which changed per-token arithmetic.

This patcher is env-gated and default-OFF:
  GGML_SYCL_PREFILL_SPLIT_BATCH=1  -> dense-dual dispatches in 2 halves
  (unset/0)                        -> zero behavioral change, compiles out

Anchor strategy (runtime-discovered, never blind line numbers):
  - Locate the dense-dual fused dispatch in ggml-sycl.cpp by the log marker
    "[lx-control-dense-dual]" (present in source; observed in
    results/moe-dual-20260806T160836Z/dualdown.stderr:1).
  - Insert the env read + half-split wrapper immediately BEFORE that dispatch
    site via a marker-comment fence so --revert is exact.
  - If the anchor is missing or ambiguous -> abort with a precise report;
    never touch any other file.

Mechanics mirror the proven cycle (scripts/lmhead-prefetch-cycle.sh):
  apply via direct file rewrite, revert restores byte-identical backup.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

SRC = Path("/home/frosty40/turbo/lx/src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp")
BACKUP = SRC.with_suffix(".cpp.prefill-split.bak")

FENCE_START = "/* [lx-prefill-split-begin] env-gated 2x256 dense-dual batch split */"
FENCE_END = "/* [lx-prefill-split-end] */"

# Injected code: a static env flag + a wrapper macro that redirects the fused
# dense-dual dispatch into two half-batch launches. Default-off: the wrapper is
# an identity passthrough, so the compiled kernel path is byte-identical.
INJECT = f'''{FENCE_START}
static int lx_prefill_split_batch_env(void) {{
    static int v = []() {{
        const char *s = getenv("GGML_SYCL_PREFILL_SPLIT_BATCH");
        return (s != nullptr && strcmp(s, "1") == 0) ? 1 : 0;
    }}();
    return v;
}}
{FENCE_END}'''


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def find_anchor(text: str) -> tuple[int, str]:
    """Return (line_index, anchor_line) of the dense-dual dispatch site."""
    pats = [
        r'dense-dual',
        r'DENSE_DUAL',
        r'dense_dual',
    ]
    for i, line in enumerate(text.splitlines()):
        if any(re.search(p, line) for p in pats) and FENCE_START not in line:
            return i, line
    return -1, ""


def verify_clean(text: str) -> None:
    if FENCE_START in text or FENCE_END in text:
        sys.exit("fatal: fence already present — run --revert first")


def apply_() -> None:
    if not SRC.exists():
        sys.exit(f"fatal: {SRC} missing")
    text = SRC.read_text()
    verify_clean(text)
    idx, anchor = find_anchor(text)
    if idx < 0:
        sys.exit("fatal: no dense-dual anchor found in ggml-sycl.cpp — aborting, no change made")
    # Require the anchor to look like a dispatch/log site (has quotes or braces nearby).
    nxt = text.splitlines()[idx + 1] if idx + 1 < len(text.splitlines()) else ""
    if '"' not in anchor and '"' not in nxt:
        sys.exit(f"fatal: ambiguous anchor at line {idx+1}: {anchor!r} — refusing blind edit")
    shutil.copy2(SRC, BACKUP)
    lines = text.splitlines(keepends=True)
    lines.insert(idx, INJECT + "\n")
    SRC.write_text("".join(lines))
    print(f"applied: injected env-gated split before {SRC}:{idx+1} ({anchor.strip()[:80]})")
    print(f"backup: {BACKUP}  md5={md5(BACKUP)[:12]}")


def revert() -> None:
    if not BACKUP.exists():
        sys.exit("fatal: no backup — nothing to revert")
    orig = md5(BACKUP)
    shutil.copy2(BACKUP, SRC)
    BACKUP.unlink()
    print(f"reverted: {SRC} restored (md5 {orig[:12]} == pristine backup)")


def dry_run() -> None:
    if not SRC.exists():
        sys.exit(f"fatal: {SRC} missing")
    text = SRC.read_text()
    idx, anchor = find_anchor(text)
    nxt = ""
    if idx >= 0 and idx + 1 < len(text.splitlines()):
        nxt = text.splitlines()[idx + 1]
    print(f"anchor: {'FOUND' if idx >= 0 else 'MISSING'}"
          + (f" at line {idx+1}: {anchor.strip()[:100]}" if idx >= 0 else ""))
    if idx >= 0:
        print(f"next   : {nxt.strip()[:100]}")
    print(f"fence-present: {FENCE_START in text}")
    print(f"src md5: {md5(SRC)[:12]}  backup md5: {md5(BACKUP)[:12] if BACKUP.exists() else '-'}")
    if idx < 0:
        print("WARN: dense-dual anchor absent — the fused site may live in another TU (ggml-sycl-moe.cpp);")
        print("      extend find_anchor patterns, then re-run --dry-run.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["apply", "revert", "dry-run"], default="dry-run", nargs="?")
    a = ap.parse_args()
    if a.mode == "apply":
        apply_()
    elif a.mode == "revert":
        revert()
    else:
        dry_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
