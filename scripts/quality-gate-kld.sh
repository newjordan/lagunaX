#!/usr/bin/env bash
# quality-gate-kld.sh — the model-quality gate for lx candidates.
#
# WHY THIS EXISTS
# ---------------
# golden-smoke.sh compares 64 greedy tokens from one prompt as a string. Any
# fused or reordered kernel shifts logits in the last bits, flips a near-tie,
# and the whole continuation diverges — so golden fails for *harmless* numeric
# drift. On 2026-07-30 it failed twice and was re-captured both times
# (correctness/golden.json.bak-* are the receipts). A gate that gets re-pinned
# whenever it fires measures nothing.
#
# This gate measures what actually matters: how far the candidate's output
# *distribution* has moved from the baseline's, over a real corpus.
#
#   1. once   : baseline logits from LX_BASE_BIN over wikitext-2   -> cached
#   2. always : candidate run with --kl-divergence against that base
#   3. pass iff mean KLD <= LX_KLD_MAX_MEAN
#           and top-1 agreement >= LX_KLD_MIN_SAME_TOP
#
# Unlike golden, there is no --capture path for the *candidate*: the base is
# pinned to the base-control binary and is only rebuilt when you explicitly
# pass --capture-base. A candidate can never re-pin its own reference.
#
# Usage:
#   bash scripts/quality-gate-kld.sh                 # gate the current LX_BIN
#   bash scripts/quality-gate-kld.sh --capture-base  # (re)pin baseline logits
#   LX_BIN=/path/to/candidate/bin bash scripts/quality-gate-kld.sh
#
# Exit: 0 pass · 1 quality regression · 2 harness/setup failure · 75 GPU busy
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
# shellcheck disable=SC1091
source "$ROOT/scripts/lib-gpu-lock.sh"

# --- knobs -------------------------------------------------------------------
# Baseline binary: the pinned control the whole score contract is measured
# against. Same tree bench-serial.sh uses for baseline.json.
LX_BASE_BIN="${LX_BASE_BIN:-/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control/bin}"
LX_KLD_CORPUS="${LX_KLD_CORPUS:-/home/frosty40/data/wikitext-2-raw/wiki.test.raw}"
# Logits cache lives on the big disk: ~200 KB/token at vocab 100352.
LX_KLD_STORE="${LX_KLD_STORE:-/mnt/data2tb/lx-kld}"
LX_KLD_CTX="${LX_KLD_CTX:-512}"
LX_KLD_CHUNKS="${LX_KLD_CHUNKS:-16}"
# Thresholds. Calibrate from a base-vs-base run (which must score ~0 / 100%).
LX_KLD_MAX_MEAN="${LX_KLD_MAX_MEAN:-0.010}"
LX_KLD_MIN_SAME_TOP="${LX_KLD_MIN_SAME_TOP:-99.0}"

CAPTURE_BASE=0
[[ "${1:-}" == "--capture-base" ]] && CAPTURE_BASE=1
CAPTURE_CANON=0
[[ "${1:-}" == "--capture-canon" ]] && CAPTURE_CANON=1

# --- canonical arbiter (gate policy (b), operator decision 2026-08-11) -------
# The pinned base encodes the code path it was captured on; every historical
# pass was bit-identical to it. A candidate that legitimately changes wide-N
# numerics can therefore only fail the pinned leg. Policy (b): such a
# candidate still PASSES iff it is at least as close to CANONICAL math
# (linear weights + fp16/oneMKL, GGML_SYCL_ENABLE_OPT=0 on the control
# binary) as the currently shipped path, with PPL-vs-canon not worse. Bounds
# are the measured shipped-path-vs-canonical numbers
# (results/reorder-multicol-20260811T201630Z, canon-vs-off.log +
# FINDING_20260811_reorder_multicol_c4): mean KLD 0.056310, same-top
# 91.667%, ln(PPL/canon) +0.023650 (includes the u16-store constant, same
# artifact on both sides of the comparison). The canonical store is captured
# from LX_BASE_BIN only (--capture-canon) — a candidate never pins either
# reference.
LX_KLD_CANON_MAX_MEAN="${LX_KLD_CANON_MAX_MEAN:-0.056310}"
LX_KLD_CANON_MIN_SAME_TOP="${LX_KLD_CANON_MIN_SAME_TOP:-91.667}"
LX_KLD_CANON_MAX_PPL_RATIO="${LX_KLD_CANON_MAX_PPL_RATIO:-0.023650}"

MODEL_TAG="$(basename "$LX_MODEL" .gguf)"
BASE_LOGITS="${LX_KLD_BASE:-$LX_KLD_STORE/${MODEL_TAG}-c${LX_KLD_CTX}-n${LX_KLD_CHUNKS}.kld}"
CANON_LOGITS="${LX_KLD_CANON_BASE:-$LX_KLD_STORE/canonical-${MODEL_TAG}-c${LX_KLD_CTX}-n${LX_KLD_CHUNKS}.kld}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTDIR="$ROOT/results/kld-$STAMP"
mkdir -p "$OUTDIR" "$LX_KLD_STORE"
LOG="$OUTDIR/gate.log"
exec > >(tee -a "$LOG") 2>&1

echo "== [kld] candidate bin : $LX_BIN"
echo "== [kld] baseline bin  : $LX_BASE_BIN"
echo "== [kld] corpus        : $LX_KLD_CORPUS"
echo "== [kld] geometry      : ctx=$LX_KLD_CTX chunks=$LX_KLD_CHUNKS"
echo "== [kld] base logits   : $BASE_LOGITS"
echo "== [kld] thresholds    : mean KLD <= $LX_KLD_MAX_MEAN, same-top >= ${LX_KLD_MIN_SAME_TOP}%"

for f in "$LX_KLD_CORPUS" "$LX_MODEL"; do
  [[ -r "$f" ]] || { echo "== [kld] SETUP FAIL — unreadable: $f"; exit 2; }
done
[[ -x "$LX_BASE_BIN/llama-perplexity" ]] || {
  echo "== [kld] SETUP FAIL — no llama-perplexity in $LX_BASE_BIN"; exit 2; }
[[ -x "$LX_BIN/llama-perplexity" ]] || {
  echo "== [kld] SETUP FAIL — no llama-perplexity in $LX_BIN"; exit 2; }
# Loader-resolution assert (FINDING_20260810, 4th sighting 2026-08-10: a
# capture ran the base binary against the champion .so via env.sh's
# LD_LIBRARY_PATH capture). Each perplexity binary must resolve libggml-sycl
# from ITS OWN bin dir when invoked with LD_LIBRARY_PATH set to that dir.
kld_assert_so() { # <bindir>
  local so
  so="$(LD_LIBRARY_PATH="$1:${LD_LIBRARY_PATH:-}" ldd "$1/llama-perplexity" 2>/dev/null | awk '/libggml-sycl/ {print $3; exit}')"
  if [[ -n "$so" && "$so" != "$1"/* ]]; then
    echo "== [kld] SETUP FAIL — $1/llama-perplexity resolves libggml-sycl from $so (not its own dir)"; exit 2
  fi
}
kld_assert_so "$LX_BIN"
[[ "$CAPTURE_BASE" -eq 1 ]] && kld_assert_so "$LX_BASE_BIN"

# Exclusive B70 — never share the card with a bench (see B70_NO_CONCURRENT_GPU).
lx_gpu_lock_enter "quality-gate-kld" || exit $?
trap 'lx_gpu_lock_leave' EXIT

# run_ppl <bin-dir> <outfile> [extra args...]
run_ppl() {
  local bindir="$1"; shift
  local out="$1"; shift
  # The candidate's kernels live in its own bin dir; point the loader there so
  # we measure THAT build's numerics, not whatever env.sh left on the path.
  LD_LIBRARY_PATH="$bindir:${LD_LIBRARY_PATH:-}" \
  "$bindir/llama-perplexity" \
    -m "$LX_MODEL" \
    -f "$LX_KLD_CORPUS" \
    -ngl "$NGL" \
    -t "$THREADS" \
    -c "$LX_KLD_CTX" \
    --chunks "$LX_KLD_CHUNKS" \
    "$@" >"$out" 2>&1
  return $?
}

# --- 1) baseline logits (cached; only rebuilt on explicit request) -----------
# AUDIT NOTE (2026-08-07): the .kld store holds uint16-quantized log-probs
# (tools/perplexity/perplexity.cpp:79-104), so Mean PPL(base) is computed from
# the quantized store while Mean PPL(Q) is f32 — the observed
# ln(PPL(Q)/PPL(base)) ≈ +0.0163 is a FORMAT CONSTANT reproduced exactly by
# control-vs-control (results/kld-20260807T154101Z), not candidate drift. Do
# not threshold it against 0; compare it against a control-vs-control run.
# Also: `--kl-divergence-base FILE` only WRITES the file from the save path
# (run WITHOUT `--kl-divergence`); when that path cannot open the file it logs
# "failed to open ... for writing" and llama-perplexity still exits 0, so the
# harness must remove the store before an explicit re-capture and VERIFY the
# rewrite, or the capture silently no-ops while echoing "captured".
if [[ "$CAPTURE_BASE" -eq 1 || ! -s "$BASE_LOGITS" ]]; then
  if [[ "$CAPTURE_BASE" -ne 1 ]]; then
    echo "== [kld] no cached base logits — capturing once from the CONTROL binary"
  fi
  echo "== [kld] capturing baseline logits (this is the slow one) …"
  if [[ -s "$BASE_LOGITS" ]]; then
    cp -f "$BASE_LOGITS" "$BASE_LOGITS.bak-$(date -u +%Y%m%dT%H%M%SZ)" || true
    rm -f "$BASE_LOGITS"   # force the save path to actually rewrite the store
  fi
  if ! run_ppl "$LX_BASE_BIN" "$OUTDIR/base-capture.log" \
        --kl-divergence-base "$BASE_LOGITS"; then
    echo "== [kld] SETUP FAIL — baseline capture failed; tail:"
    tail -20 "$OUTDIR/base-capture.log"
    rm -f "$BASE_LOGITS"
    exit 2
  fi
  if [[ ! -s "$BASE_LOGITS" ]] || \
     grep -q "failed to open" "$OUTDIR/base-capture.log"; then
    echo "== [kld] SETUP FAIL — capture did not rewrite the store (silent no-op class); tail:"
    tail -8 "$OUTDIR/base-capture.log"
    rm -f "$BASE_LOGITS"
    exit 2
  fi
  echo "== [kld] baseline logits captured → $BASE_LOGITS ($(du -h "$BASE_LOGITS" | cut -f1))"
fi

# --- 2) candidate vs baseline ------------------------------------------------
echo "== [kld] scoring candidate against baseline logits …"
if ! run_ppl "$LX_BIN" "$OUTDIR/candidate.log" \
      --kl-divergence --kl-divergence-base "$BASE_LOGITS"; then
  echo "== [kld] SETUP FAIL — candidate perplexity run failed; tail:"
  tail -20 "$OUTDIR/candidate.log"
  exit 2
fi

# --- 3) parse + gate ---------------------------------------------------------
python3 - "$OUTDIR/candidate.log" "$OUTDIR/kld.json" \
         "$LX_KLD_MAX_MEAN" "$LX_KLD_MIN_SAME_TOP" \
         "$LX_BIN" "$LX_BASE_BIN" "$LX_KLD_CTX" "$LX_KLD_CHUNKS" "$LX_KLD_CORPUS" <<'PY'
import json, pathlib, re, sys

log, out, max_mean, min_same, cand_bin, base_bin, ctx, chunks, corpus = sys.argv[1:10]
text = pathlib.Path(log).read_text(errors="replace")

def grab(pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

mean_kld  = grab(r"Mean\s+KLD:\s*([0-9.eE+-]+)")
same_top  = grab(r"Same top p:\s*([0-9.eE+-]+)")
rms_dp    = grab(r"RMS\s*\u0394p\s*:\s*([0-9.eE+-]+)")
ppl_ratio = grab(r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([0-9.eE+-]+)")
ppl_q     = grab(r"Mean\s+PPL\(Q\)\s*:\s*([0-9.eE+-]+)")
ppl_base  = grab(r"Mean\s+PPL\(base\)\s*:\s*([0-9.eE+-]+)")

if mean_kld is None or same_top is None:
    print("== [kld] PARSE FAIL — no KL divergence statistics in candidate log")
    print("--- tail ---")
    print("\n".join(text.splitlines()[-25:]))
    raise SystemExit(2)

max_mean, min_same = float(max_mean), float(min_same)
ok_kld  = mean_kld <= max_mean
ok_top  = same_top >= min_same
passed  = ok_kld and ok_top

payload = {
    "stamp_utc": None,
    "candidate_bin": cand_bin,
    "baseline_bin": base_bin,
    "corpus": corpus,
    "ctx": int(ctx),
    "chunks": int(chunks),
    "mean_kld": mean_kld,
    "same_top_pct": same_top,
    "rms_dp_pct": rms_dp,
    "mean_ln_ppl_ratio": ppl_ratio,
    "mean_ppl_candidate": ppl_q,
    "mean_ppl_base": ppl_base,
    "max_mean_kld": max_mean,
    "min_same_top_pct": min_same,
    "kld_ok": ok_kld,
    "same_top_ok": ok_top,
    "passed": passed,
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n")

print(f"== [kld] mean KLD    : {mean_kld:.6f}  (limit {max_mean})  {'OK' if ok_kld else 'FAIL'}")
print(f"== [kld] same top-1  : {same_top:.3f}% (floor {min_same}%) {'OK' if ok_top else 'FAIL'}")
if rms_dp is not None:
    print(f"== [kld] RMS dp      : {rms_dp:.3f}%")
if ppl_ratio is not None:
    print(f"== [kld] ln(PPL/base): {ppl_ratio:+.6f}")
print("== [kld] QUALITY GATE " + ("PASS" if passed else "FAIL"))
raise SystemExit(0 if passed else 1)
PY
RC=$?

# --- 3b) canonical arbiter (only when the pinned leg failed) -----------------
if [[ "$CAPTURE_CANON" -eq 1 ]]; then
  echo "== [kld] capturing CANONICAL logits from the control binary (ENABLE_OPT=0) …"
  rm -f "$CANON_LOGITS"
  if ! GGML_SYCL_ENABLE_OPT=0 run_ppl "$LX_BASE_BIN" "$OUTDIR/canon-capture.log" \
        --kl-divergence-base "$CANON_LOGITS"; then
    echo "== [kld] CANON CAPTURE FAIL"; tail -10 "$OUTDIR/canon-capture.log"; exit 2
  fi
  if [[ ! -s "$CANON_LOGITS" ]] || grep -q "failed to open" "$OUTDIR/canon-capture.log"; then
    echo "== [kld] CANON CAPTURE FAIL (silent no-op class)"; exit 2
  fi
  echo "== [kld] canonical logits captured → $CANON_LOGITS ($(du -h "$CANON_LOGITS" | cut -f1))"
fi
if [[ $RC -eq 1 && -s "$CANON_LOGITS" ]]; then
  echo "== [kld] pinned leg FAILED — canonical arbiter (policy b, 2026-08-11) …"
  if run_ppl "$LX_BIN" "$OUTDIR/candidate-canon.log" \
        --kl-divergence --kl-divergence-base "$CANON_LOGITS"; then
    python3 - "$OUTDIR/candidate-canon.log" "$OUTDIR/kld.json" \
             "$LX_KLD_CANON_MAX_MEAN" "$LX_KLD_CANON_MIN_SAME_TOP" "$LX_KLD_CANON_MAX_PPL_RATIO" <<'PY'
import json, pathlib, re, sys
log, out, max_mean, min_same, max_ppl = sys.argv[1:6]
text = pathlib.Path(log).read_text(errors="replace")
def grab(pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None
mean_kld  = grab(r"Mean\s+KLD:\s*([0-9.eE+-]+)")
same_top  = grab(r"Same top p:\s*([0-9.eE+-]+)")
ppl_ratio = grab(r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*([0-9.eE+-]+)")
if mean_kld is None or same_top is None:
    print("== [kld] canon arbiter PARSE FAIL"); raise SystemExit(2)
ok = (mean_kld <= float(max_mean) and same_top >= float(min_same) and
      (ppl_ratio is None or ppl_ratio <= float(max_ppl)))
p = pathlib.Path(out); d = json.loads(p.read_text())
d.update({"canon_mean_kld": mean_kld, "canon_same_top_pct": same_top,
          "canon_ln_ppl_ratio": ppl_ratio,
          "canon_max_mean_kld": float(max_mean),
          "canon_min_same_top_pct": float(min_same),
          "canon_max_ln_ppl_ratio": float(max_ppl),
          "canon_ok": ok})
if ok:
    d["passed"] = True
    d["passed_via"] = "canonical_arbiter"
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"== [kld] canon mean KLD : {mean_kld:.6f}  (bound {max_mean})  {'OK' if mean_kld <= float(max_mean) else 'FAIL'}")
print(f"== [kld] canon same top : {same_top:.3f}% (bound {min_same}%) {'OK' if same_top >= float(min_same) else 'FAIL'}")
if ppl_ratio is not None:
    print(f"== [kld] canon ln(PPL)  : {ppl_ratio:+.6f} (bound +{max_ppl})")
print("== [kld] CANONICAL ARBITER " + ("PASS — candidate is at least as close to canonical as the shipped path" if ok else "FAIL"))
raise SystemExit(0 if ok else 1)
PY
    [[ $? -eq 0 ]] && RC=0
  else
    echo "== [kld] canon arbiter candidate run failed; pinned verdict stands"
  fi
elif [[ $RC -eq 1 ]]; then
  echo "== [kld] pinned leg FAILED and no canonical store at $CANON_LOGITS — run --capture-canon once (control binary) to enable the arbiter"
fi

# Stamp the receipt and expose a stable pointer for the promote path.
if [[ -s "$OUTDIR/kld.json" ]]; then
  python3 - "$OUTDIR/kld.json" "$STAMP" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["stamp_utc"] = sys.argv[2]
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  cp "$OUTDIR/kld.json" "$ROOT/results/LATEST_KLD.json"
fi

echo "== [kld] receipt → $OUTDIR/kld.json (exit $RC)"
exit $RC
