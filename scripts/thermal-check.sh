#!/usr/bin/env bash
# thermal-check.sh — pre-score card-state sampler for the B70 GPU.
# Prints the current maximum device temperature and fan RPM across every
# DRM card's hwmon subtree, and exits nonzero when the max temp is above a
# threshold. Purpose: harden the submission pipeline (open lead 1) so a
# cycle that benches a still-hot card cannot silently under-report by ~2%
# (evidence: findings 8/13 — heat-depressed runs score ~1.19 vs 1.22 cold).
# Self-contained: no dependency on env.sh / bench-serial.sh internals.
set -u

THRESHOLD="${THERMAL_CHECK_THRESHOLD:-70}"   # °C; exit 1 above this
JSON="${THERMAL_CHECK_JSON:-0}"              # 1 = emit a JSON record

best_temp=0
best_label=""
fan_rpm=""
fan_label=""

for card in /sys/class/drm/card*; do
  [ -d "$card/device/hwmon" ] || continue
  for hw in "$card"/device/hwmon/hwmon*; do
    [ -d "$hw" ] || continue
    name="$(cat "$hw/name" 2>/dev/null || echo unknown)"
    for tf in "$hw"/temp*_input; do
      [ -f "$tf" ] || continue
      v="$(cat "$tf" 2>/dev/null)" || continue
      c=$((v / 1000))
      if [ "$c" -gt "$best_temp" ]; then
        best_temp=$c
        best_label="${card##*/}:${hw##*/}:${tf##*/} ($name)"
      fi
    done
    for ff in "$hw"/fan*_input; do
      [ -f "$ff" ] || continue
      r="$(cat "$ff" 2>/dev/null)" || continue
      fan_rpm="$r"
      fan_label="${card##*/}:${hw##*/}:${ff##*/}"
    done
  done
done

if [ "$JSON" = "1" ]; then
  printf '{"max_temp_c":%d,"source":"%s","fan_rpm":%s,"threshold_c":%d,"ok":%s,"ts":"%s"}\n' \
    "$best_temp" "$best_label" "${fan_rpm:-null}" "$THRESHOLD" \
    "$([ "$best_temp" -le "$THRESHOLD" ] && echo true || echo false)" \
    "$(date -u +%Y%m%dT%H%M%SZ)"
else
  echo "max_temp_c=$best_temp  source=${best_label:-none}"
  echo "fan_rpm=${fan_rpm:-n/a}  fan_source=${fan_label:-none}"
  if [ "$best_temp" -le "$THRESHOLD" ]; then
    echo "THERMAL OK (${best_temp}C <= ${THRESHOLD}C threshold)"
    exit 0
  else
    echo "THERMAL WARM (${best_temp}C > ${THRESHOLD}C threshold) — refuse to score"
    exit 1
  fi
fi
