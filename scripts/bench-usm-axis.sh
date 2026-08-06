#!/usr/bin/env bash
# bench-usm-axis.sh — Level Zero driver/allocator policy axis for the B70 serial score.
#
# Why this axis is NEW and distinct from every prior direction:
#   - results/knob-ab-ledger.md proves the GGML_SYCL_* runtime plane is exhausted:
#     of 21 interleaved A/B arms, the only 10 rows beating the ±0.68% drift bound are
#     all regressions (-1.7% .. -22.0%). Every remaining lever is either a source edit
#     (lm_head pretranspose, ceiling +5.12% tg per lmhead-probe-ledger.md) or a
#     measurement artifact.
#   - results/usm-placement-policy-audit-20260807.json records artifact_mentions: 0 for
#     the Level Zero USM residency/allocator controls (SYCL_PI_LEVEL_ZERO_USE_USM_ALLOCATOR,
#     SYCL_PI_LEVEL_ZERO_USM_RESIDENT_DEVICE, ...) — an untouched control axis, read by the
#     oneAPI Level Zero plugin at allocation time, NOT by ggml.
#   - ABSOLUTE_LIMIT.md:25-26 concludes decode is memory-bound; if the bench allocations
#     are host-visible shared USM, device-resident allocation removes per-kernel host-side
#     residency arbitration. Allocation policy cannot change numerics, so any win is
#     quality-invisible by construction (golden smoke still gated below).
#
# Design: a thin env-diff wrapper over the OFFICIAL pipeline (scripts/bench-serial.sh).
# env.sh never unsets these names, so exporting them before the official bench propagates.
# Each arm benches through the same lock/thermal/golden/cooldown path as the champion.
#
# Usage: scripts/bench-usm-axis.sh [--dry-run] [CTRL|USM_DEVICE_RESIDENT|USM_ALLOCATOR|ALL]

set -uo pipefail
cd "$(dirname "$0")/.."

BENCH_SERIAL=scripts/bench-serial.sh
LATEST=results/LATEST_SCORE.json
LEDGER=results/usm-ab-ledger.md
MODE="${1:-ALL}"

[ -f "$BENCH_SERIAL" ] || { echo "FATAL: $BENCH_SERIAL missing" >&2; exit 2; }
[ -f "$LATEST" ] || { echo "FATAL: $LATEST missing" >&2; exit 2; }

resolve_artifacts() {
  # Reuse the champion binary + model the official board points at, so the A/B is
  # binary-identical and only the env differs (same-window delta vs the drift bound).
  local bin model
  bin="$(jq -r '.candidate_meta.binary // .candidate_path' "$LATEST" 2>/dev/null | sed 's#^\./##')"
  model="$(jq -r '.candidate_meta.model // empty' "$LATEST" 2>/dev/null)"
  if [ -z "$model" ] || [ -z "$bin" ] || [ ! -x "$bin" ]; then
    echo "FATAL: cannot resolve champion binary ($bin) or model ($model) from $LATEST" >&2
    exit 2
  fi
  echo "$bin|$model"
}

declare -A ARM_ENV=(
  [CTRL]=""
  [USM_RESIDENT]="SYCL_PI_LEVEL_ZERO_USM_RESIDENT=1"
  [USM_ALLOCATOR_OFF]="SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR=1"
  [SINGLE_THREAD]="SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE=1"
  [POLL_0]="LX_POLL=0"
)
declare -A ARM_NOTE=(
  [CTRL]="same-window control (no extra env) — bounds ambient drift"
  [USM_RESIDENT]="pin ALL USM device-resident (USM_RESIDENT=1, adapter-read)"
  [USM_ALLOCATOR_OFF]="bypass USM allocator (DISABLE_USM_ALLOCATOR=1)"
  [SINGLE_THREAD]="adapter host single-thread mode (SINGLE_THREAD_MODE=1)"
  [POLL_0]="host submission busy-poll (LX_POLL=0 vs default 50)"
)

run_arm() {
  local arm="$1" stamp bin model
  bin="${ARTIFACTS%%|*}"; model="${ARTIFACTS##*|}"
  if [ "$arm" = "CTRL" ]; then
    unset SYCL_PI_LEVEL_ZERO_USM_RESIDENT SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE LX_POLL
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    echo "== CTRL arm: $stamp" 
    bash "$BENCH_SERIAL" || { echo "CTRL arm FAILED (rc $?)" >&2; return 1; }
  else
    local kv="${ARM_ENV[$arm]}"
    export "${kv%%=*}=${kv#*=}"
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    echo "== $arm arm: $stamp  ($kv)"
    local rc=0
  for try in 1 2 3 4 5 6 7 8 9 10; do
    bash "$BENCH_SERIAL" --note "usm-$arm" && rc=0 && break
    rc=$?
    if [[ "$rc" == "75" ]]; then
      echo "  [usm-$arm] GPU lock busy (rc 75) — retry $try/10 in 90 s" >&2
      sleep 90
    else
      echo "[usm-$arm] bench failed rc=$rc" >&2
      break
    fi
  done
  [[ "$rc" == "0" ]] || { echo "usm-$arm arm FAILED (rc $rc)" >&2; return 1; }
  fi
  # The official bench mints its own timestamped dir; resolve the newest receipt.
  local mjson tg pp
  mjson="$(ls -t results/2026*/metrics.json 2>/dev/null | head -1)"
  [ -f "$mjson" ] || { echo "no metrics.json produced by $BENCH_SERIAL" >&2; return 1; }
  tg="$(jq -r '.tg128 // .tg_avg_ts // .decode_tok_s // empty' "$mjson" 2>/dev/null)"
  pp="$(jq -r '.pp512 // .pp_avg_ts // .prefill_tok_s // empty' "$mjson" 2>/dev/null)"
  echo "| $stamp | $arm | tg=$tg | pp=$pp | ${ARM_NOTE[$arm]} |" >> "$LEDGER"
  echo "logged: $stamp $arm"
}

ARTIFACTS="$(resolve_artifacts)"
echo "champion binary: ${ARTIFACTS%%|*}"
echo "model:           ${ARTIFACTS##*|}"

if [ "${MODE}" = "--dry-run" ] || [ "${MODE}" = "dry-run" ]; then
  echo "dry-run: plumbing OK (env axes defined: ${!ARM_ENV[*]})"
  exit 0
fi

[ -f "$LEDGER" ] || {
  echo "# USM axis A/B ledger — Level Zero driver/allocator policy (vs same-window CTRL)" > "$LEDGER"
  echo "# Drift bound: ±0.68% tg (between-run card-state, see knob-ab-ledger.md)." >> "$LEDGER"
  echo "| stamp | arm | tg_tok_s | pp_tok_s | note |" >> "$LEDGER"
}

case "$MODE" in
  CTRL|USM_RESIDENT|USM_ALLOCATOR_OFF|SINGLE_THREAD|POLL_0) run_arm "$MODE" ;;
  ALL|--all)
    run_arm CTRL
    run_arm USM_RESIDENT
    run_arm USM_ALLOCATOR_OFF
    run_arm SINGLE_THREAD
    run_arm POLL_0
    ;;
  *) echo "unknown arm: $MODE (CTRL|USM_RESIDENT|USM_ALLOCATOR_OFF|SINGLE_THREAD|POLL_0|ALL)" >&2; exit 2 ;;
esac
