#!/usr/bin/env bash
# thermal-gate.sh — B70 thermal precondition for official benches.
#
# Problem (measured): the same binary+flags scores 1.1908 right after a
# 49-min proof-suite vs 1.2181 after >=5 min idle — a ~2.3% score swing
# from card thermal/power state. Absolute temp thresholds don't work here:
# this datacenter box idles at ~70-80C across its 21 hwmon sensors, so a
# "cold" target would never be reached.
#
# Solution: gate on STABILITY, not absolutes. Wait until
#   (a) at least THERMAL_MIN_IDLE_S (default 300) has elapsed, AND
#   (b) the max sensor temp has stopped falling meaningfully
#       (drop over the trailing window < THERMAL_STABLE_DROP_C, default 2).
# No hwmon -> fall back to a plain min-idle wait (never hard-blocks).
# THERMAL_TIMEOUT_S (default 900) caps the wait; on timeout we proceed with
# a warning unless THERMAL_STRICT=1 (then abort).
#
# Exit codes: 0 = benchable (or proceeding warm); 2 = hwmon gone mid-wait.
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
THERMAL_MIN_IDLE_S="${THERMAL_MIN_IDLE_S:-300}"
THERMAL_STABLE_DROP_C="${THERMAL_STABLE_DROP_C:-2}"
THERMAL_TIMEOUT_S="${THERMAL_TIMEOUT_S:-900}"
THERMAL_STRICT="${THERMAL_STRICT:-0}"
POLL_S=15

# ---- hwmon discovery --------------------------------------------------------
HWMON="${THERMAL_HWMON:-}"
if [ -z "$HWMON" ]; then
  for d in /sys/class/drm/card*/device/hwmon/hwmon*; do
    [ -d "$d" ] || continue
    HWMON="$d"
    break
  done
fi

max_temp_mc() { # millidegrees Celsius of hottest sensor; empty if unreadable
  local f v max=0
  for f in "$HWMON"/temp*_input; do
    [ -r "$f" ] || continue
    v="$(cat "$f" 2>/dev/null || true)"
    [ -n "$v" ] && [ "$v" -gt "$max" ] 2>/dev/null && max="$v"
  done
  [ "$max" -gt 0 ] && echo "$max"
}

now_s() { date +%s; }

if [ -n "$HWMON" ] && [ -d "$HWMON" ]; then
  if t="$(max_temp_mc)" && [ -n "$t" ]; then
    echo "thermal-gate: hwmon=$HWMON max_temp=$((t/1000)).$((t%1000/100))C" >&2
  else
    echo "thermal-gate: hwmon present but temps unreadable -> idle-wait fallback" >&2
    HWMON=""
  fi
else
  echo "thermal-gate: no hwmon -> idle-wait fallback (THERMAL_MIN_IDLE_S=${THERMAL_MIN_IDLE_S}s)" >&2
  HWMON=""
fi

start="$(now_s)"
first_max="$(max_temp_mc || true)"
last_max="$first_max"
deadline=$(( start + THERMAL_TIMEOUT_S ))

while :; do
  now="$(now_s)"
  cur="$(max_temp_mc || true)"
  [ -n "$cur" ] && last_max="$cur"

  # idle floor estimate: the min temp seen so far (falling curve bottoms out)
  if [ -n "$first_max" ] && [ -n "$cur" ]; then
    [ "$cur" -lt "$first_max" ] && first_max="$cur"
  fi

  elapsed=$(( now - start ))

  if [ "$elapsed" -ge "$THERMAL_MIN_IDLE_S" ]; then
    stable=1
    if [ -n "$HWMON" ] && [ -n "$first_max" ] && [ -n "$cur" ]; then
      # hottest now vs coolest seen: how much is the card still cooling?
      drop_c=$(( (cur - first_max) / 1000 ))
      # negative drop => reheating; treat as unstable
      if [ "$drop_c" -lt 0 ] || [ "$drop_c" -ge "$THERMAL_STABLE_DROP_C" ]; then
        stable=0
      fi
    fi
    if [ "$stable" -eq 1 ]; then
      echo "thermal-gate: OK after ${elapsed}s idle (cur=${cur:+$((cur/1000))C})" >&2
      exit 0
    fi
    echo "thermal-gate: ${elapsed}s idle, still cooling (cur=${cur:+$((cur/1000))C}, floor=${first_max:+$((first_max/1000))C})" >&2
  else
    echo "thermal-gate: ${elapsed}s/${THERMAL_MIN_IDLE_S}s idle, waiting (cur=${cur:+$((cur/1000))C})" >&2
  fi

  if [ "$now" -ge "$deadline" ]; then
    if [ "$THERMAL_STRICT" = "1" ]; then
      echo "thermal-gate: TIMEOUT after ${THERMAL_TIMEOUT_S}s, THERMAL_STRICT=1 -> abort" >&2
      exit 1
    fi
    echo "thermal-gate: TIMEOUT after ${THERMAL_TIMEOUT_S}s — proceeding WARM (set THERMAL_STRICT=1 to abort)" >&2
    exit 0
  fi

  sleep "$POLL_S"
done
