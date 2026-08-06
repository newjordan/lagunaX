#!/usr/bin/env bash
# attrib-suffix.sh — attribute the layer-timer per-suffix dispatch histogram to
# concrete tensor names from the nm[] dump. Closes the "-39 anomaly": the lmhead
# probe histogram counts 917 suffix-39 dispatches while only 131 l_out-39 calls
# appear in the ffn_out bucket row — the dump says WHICH tensors actually carry
# the -39 suffix and whether a second fused group hides in the last block.
#
# usage: bash scripts/attrib-suffix.sh <bench.stderr>
set -u

STDERR="${1:-}"
if [[ -z "$STDERR" || ! -f "$STDERR" ]]; then
  echo "usage: $0 <bench.stderr>" >&2
  exit 2
fi

echo "== 1) per-suffix dispatch histogram (lmhead-probe FINAL line, ALL dispatches) =="
hist=$(grep -o 'per-suffix hits:.*' "$STDERR" | head -1 | sed 's/per-suffix hits: //')
if [[ -z "$hist" ]]; then
  echo "(no lmhead-probe FINAL histogram found in $STDERR)"
else
  echo "$hist" | tr ' ' '\n' | grep -v '^$' | awk -F: '{ n=$2+0; if (n>0) printf "  suffix %2d: %8d dispatches\n", $1, n }'
fi
echo

echo "== 2) nm[] name dump: per-suffix tensor-name census (dump capped at first 200 lines) =="
# name[count] for each distinct tensor name; then per-suffix aggregates.
grep -oE '\[layer-timer\] nm\[[0-9]+\] [^ ]+' "$STDERR" \
  | sed -E 's/.*nm\[[0-9]+\] //' \
  | awk '
      {
        name = $0;
        # suffix = trailing number after last '-'; else classify as "nosuffix"
        if (match(name, /-[0-9]+$/)) {
          suf = substr(name, RSTART + 1);
        } else {
          suf = "nosuffix";
        }
        cnt[name]++;
        sufname[suf" "name]++;
        sufcnt[suf]++;
      }
      END {
        n = 0;
        for (s in sufcnt) n++;
        if (n == 0) { print "  (no nm[] lines found)"; exit; }
        print "  --- per suffix: distinct names / total dispatches (within the 200-dump window) ---";
        for (s in sufcnt) {
          printf "  suffix %-9s: %5d dispatches\n", s, sufcnt[s];
        }
        print "";
        print "  --- tensors carrying the LAST-BLOCK suffix (max numeric suffix present) ---";
        max = -1;
        for (s in sufcnt) if (s != "nosuffix" && s + 0 > max) max = s + 0;
        for (n in sufname) {
          split(n, a, " ");
          if (a[1] == max) printf "  suffix %-9s: %-28s %5d\n", a[1], a[2], sufname[n];
        }
      }' | sort
echo

echo "== 3) bucket rows (us / calls / us-call) =="
grep -E '^\[layer-timer\] bucket ' "$STDERR" | sed -E 's/^\[layer-timer\] //'
echo

echo "== 4) cross-check: suffix-39 dump count vs histogram 39:N (dump is capped at 200; >200 dispatches of any suffix will undercount) =="
dump39=$(grep -oE '\[layer-timer\] nm\[[0-9]+\] [^ ]+' "$STDERR" | grep -cE '\-[0-9]+$' || true)
# count names ending in -39 within the dump
dump39n=$(grep -oE '\[layer-timer\] nm\[[0-9]+\] [^ ]+-39' "$STDERR" | wc -l)
hist39=$(echo "$hist" | tr ' ' '\n' | grep -E '^39:' | cut -d: -f2)
echo "  suffix-39 names in 200-line dump: $dump39n ; histogram suffix-39 total: ${hist39:-n/a}"
