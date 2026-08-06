# Mount Doom — 2-day push goal

**Window:** 2026-07-31 → **2026-08-02 14:00 UTC** (~48h)  
**Campaign dir:** `results/quest-2day-20260731/`  
**Status:** PAUSED afternoon 2026-07-31 — see `notes/HANDOFF_20260731_afternoon.md`

## Primary goal

Beat the post-brownout **quality-safe champion** on serial formal score while
keeping golden + quality floors:

| bar | score | tg128 | pp512 | notes |
|-----|------:|------:|------:|-------|
| pin | 1.000 | 107.35 | 1139 | never re-pin |
| tip-freeze QS | 1.209 | 136.4 | 1187 | do not go below |
| **start champion** | **1.227** | **139.3** | **1183** | `20260731T141436Z` |
| **GOAL** | **≥ 1.250** | **≥ 143** | floors OK | golden OK, PPL sane |
| stretch | ≥ 1.300 | ≥ 148 | — | only if quality holds |

```text
score = decode_speedup^0.75 * prefill_speedup^0.25
floors: both speedups ≥ 0.95
hard gate: greedy golden match
quality: no 1e5+ PPL (dual_down / any-batch mm-add stay gated unless fixed)
```

## Non-goals (48h)

- Multi-slot / np capacity numbers
- Re-pinning baseline or golden to invent wins
- Shipping dual_down or any-batch mm-add without PPL proof
- Concurrent GPU jobs (wedges xe)

## Work tracks (priority)

| # | track | why | done when |
|---|-------|-----|-----------|
| **T1** | Tip-source restore (rms/rope/softplus/add_add/FA + mm-add) | rebuildable tip; drop binary patch | source `build-*` golden OK + score ≥ start |
| **T2** | Decode MoE kernel wins under quality-safe tip | MoE tiny-N still dominates ktrace | formal score ↑, golden OK |
| **T3** | Safe micro-wave (flags/env, ub/b, residual knobs) | noise vs real under tip | board of arms; keep only winners |
| **T4** | dual_down decode-only isolation | big prefill if fixable | PPL OK + golden OK or kill stays |

## Loop (always on)

```bash
# continuous rebench + board (ship flags from env.sh)
bash scripts/quest-2day-launch.sh
tail -f results/quest-2day-20260731/quest.log
```

Cycle body (GPU exclusive lock):

1. golden smoke every 4th cycle (fail → alert, no champion update)
2. formal `bench-serial` on `$LX_BIN` (champion binary)
3. score vs pin; update CHAMPION if better **score** (then tg, then pp)
4. every 3rd: short power profile; every 6th: onednn ktrace
5. sleep `QUEST_SLEEP_S` (default **1200s / 20 min**) so agents can take the card

## Agent push loop

Between quest cycles, agents work **T1→T4**:

1. edit → rebuild (never overwrite `build-mmadd-decode` until source beats it)
2. golden → formal under lock
3. ship note + board update on win
4. if score ≥ 1.250: freeze binary under `results/goal-hit-<stamp>/`

## Hard rules

1. One Level-Zero client — `notes/B70_NO_CONCURRENT_GPU.md`
2. Champion binary until source golden matches tip oracle:  
   `LX_BIN=.../build-mmadd-decode/bin`
3. dual_down + dual_multitoken stay **OFF** in `env.sh` until fixed
4. mm-add any-batch stays OFF (decode-only only)
5. No baseline re-pin

## How to check progress

```bash
cat results/quest-2day-20260731/CHAMPION.json
cat results/quest-2day-20260731/BOARD.json | head
cat results/LATEST_SCORE.json
tail -50 results/quest-2day-20260731/quest.log
```
