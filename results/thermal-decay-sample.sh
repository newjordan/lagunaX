#!/usr/bin/env bash
# Sample B70 hwmon temps from now every 60s for DECAY_MIN minutes (default 15).
set -u
OUT="results/thermal-decay-$(date -u +%Y%m%dT%H%M%SZ).log"
DECAY_MIN="${DECAY_MIN:-15}"
END=$(( $(date +%s) + DECAY_MIN*60 ))
while [ "$(date +%s)" -lt "$END" ]; do
  T=""
  for f in /sys/class/drm/card0/device/hwmon/hwmon6/temp*_input; do
    v=$(cat "$f" 2>/dev/null); [ -n "$v" ] && [ "$v" -gt "${T:-0}" ] && T=$v
  done
  F=""; [ -r /sys/class/drm/card0/device/hwmon/hwmon6/fan1_input ] && F=$(cat /sys/class/drm/card0/device/hwmon/hwmon6/fan1_input)
  L="$(pgrep -fc 'llama-(server|bench|cli)' 2>/dev/null || echo 0)"
  echo "$(date -u +%Y%m%dT%H%M%SZ) max_temp=$((T/1000))C fan=${F}rpm llama_procs=$L" >> "$OUT"
  sleep 60
done
echo "decay-sampler done -> $OUT"
