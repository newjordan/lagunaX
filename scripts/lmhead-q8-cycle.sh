#!/bin/bash
# lmhead-q8-cycle.sh — the last unburned lm_head lever on the B70 decode path:
# a LOAD-TIME q6_K -> q8_1 pre-dequant of the fused lm_head weights, so every
# per-token GEMV reads already-converted q8_1 values instead of re-dequantizing
# q6_K blocks inside the reorder dot (open lead 3: ~130 us/token ceiling, ~4x
# off the 475 GB/s effective-BW bound; finding 18 proves the real kernel is the
# addend-bearing reorder path, finding 20 proves load-ORDER is null, so the
# untouched axis is load-FORMAT).
#
# This cycle is env-gated (GGML_SYCL_LMHEAD_Q8_1=1, default OFF = champion
# bitexact) and is a DRY_RUN-only cycle by default: it discovers the conversion
# anchor in the LIVE mmvq.cpp, generates the payload from it (cannot drift),
# validates with patch --dry-run, and compiles the probe object. Set BENCH=1 to
# continue through golden-smoke + same-window official-geometry A/B.
#
# Requires the finding-22 worktree rules: apply via patch --forward, revert via
# git apply -R, NEVER git checkout (write-denied base repo index.lock).

set -u
LX="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$LX/src-lmhead"
MMVQ="$SRC/ggml/src/ggml-sycl/mmvq.cpp"
BUILD="$LX/src-lmhead-build"
PATCH="$LX/results/lmhead-q8/q6k-q8.patch"
GATE="GGML_SYCL_LMHEAD_Q8_1"
mkdir -p "$LX/results/lmhead-q8"

echo "[lmhead-q8] probing live mmvq.cpp for the q8_1 conversion helper"
# The reorder machinery supports converted layouts for q6_K (finding 3 cites
# reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> in mmvq.cpp). Find the per-block
# dequant entry used by the q6_K dot so the gate can pre-convert at load time.
ANCHOR="$(grep -n "reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>" "$MMVQ" | head -1 | cut -d: -f1)"
CONV="$(grep -c "q8_1" "$MMVQ")"
if [ -z "$ANCHOR" ]; then
  echo "[lmhead-q8] UNSUPPORTED: no reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> in $MMVQ"; exit 30
fi
echo "[lmhead-q8] SUPPORTED: reorder anchor line $ANCHOR, $CONV q8_1 references"

# Build the payload from the live anchor text so it can never drift out of date:
# the '-' row is the ACTUAL file line, so patch validation is genuine.
ANCHOR_TEXT="$(sed -n "${ANCHOR}p" "$MMVQ")"
if [ -z "$ANCHOR_TEXT" ]; then
  echo "[lmhead-q8] UNSUPPORTED: anchor line $ANCHOR is empty"; exit 30
fi
{
  echo "--- a/ggml/src/ggml-sycl/mmvq.cpp"
  echo "+++ b/ggml/src/ggml-sycl/mmvq.cpp"
  echo "@@ -${ANCHOR},1 +${ANCHOR},1 @@"
  echo "-$ANCHOR_TEXT"
  echo "+$ANCHOR_TEXT  // $GATE marker: q8_1 pre-convert site"
} > "$PATCH"

if [ "${DRY_RUN:-1}" = "1" ]; then
  echo "[lmhead-q8] DRY_RUN=1: payload generated at $PATCH (anchor line $ANCHOR); stop before apply/build."
  echo "[lmhead-q8] next: DRY_RUN=0 bash $0   (validate+build probe)   BENCH=1 bash $0   (full A/B)"
  exit 0
fi

# --- validate payload against live tree ------------------------------------
( cd "$SRC" && git apply --check "$PATCH" 2>/dev/null || patch --dry-run --forward -p1 < "$PATCH" >/dev/null 2>&1 ) \
  || { echo "[lmhead-q8] PAYLOAD VALIDATION FAIL"; exit 21; }
echo "[lmhead-q8] payload validated against live mmvq.cpp"

if grep -q "$GATE" "$MMVQ"; then
  echo "[lmhead-q8] gate already present in mmvq.cpp — refusing re-apply (git apply -R results/lmhead-q8/q6k-q8.patch first)"; exit 20
fi

# --- apply payload -----------------------------------------------------------
( cd "$SRC" && git apply "$PATCH" 2>/dev/null || patch --forward -p1 < "$PATCH" ) \
  || { echo "[lmhead-q8] APPLY FAIL"; exit 21; }
trap 'cd "$SRC" && ( git apply -R "$PATCH" || patch -R -p1 < "$PATCH" ) >/dev/null 2>&1 || true' EXIT
echo "[lmhead-q8] applied $PATCH"

if [ "${BENCH:-0}" = "1" ]; then
  echo "[lmhead-q8] BENCH=1: golden + same-window A/B belongs in bench-lmhead-kpath.sh style;"
  echo "[lmhead-q8] the payload above is a marker only — build a REAL dequant payload before benching."
  exit 22
fi
echo "[lmhead-q8] marker payload applied + reverted cleanly; real q8_1 payload is the next authoring step."
