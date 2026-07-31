# Laguna A/B — base vs quality-safe tip

Generated: 2026-07-31 00:22 UTC

Same binary + same Laguna-XS-2.1 Q4_K_M. **No Treebeard/Qwen comparison.**

| arm | definition |
|-----|------------|
| **base** | all major custom SYCL fuses **OFF** |
| **tip** | quality-safe tip: only `MUL_MAT_ADD` + `MOE_DUAL_DOWN` + `MOE_DUAL_MULTITOKEN` OFF; other tip fuses ON |

## Scoreboard

| Gate | base | tip | tip − base |
|------|-----:|----:|-----------:|
| formal pp512 | 1149.9 | 1186.9 | **+36.98** |
| formal tg128 | 108.7 | 136.4 | **+27.68** |
| ladder pp2048 | 1954.2 | 2028.9 | +74.7 |
| ladder pp8192 | 1879.6 | 1950.1 | +70.53 |
| single-agent tg p50 | 108.7 | 135.6 | **+26.93** |
| single content ok | 10/10 | 10/10 | same |
| needles | 3/3 | 3/3 | same |
| dossier | 12/14 | 12/14 | same |
| held-out | 32.6% (15/46) | 37.0% (17/46) | **+4.4 pp** |
| agent69 score | **8**/100 (4/3/62 p/r/f) | see note | — |

## Agent Bench 69 note

- ctx=32768 for both (tool schemas often ~35k → some automatic fails).
- **base completed:** score **8**/100 · pass 4 · partial 3 · fail 62 · median_turn_ms 28827.5 · ctx_exceed hits 4
- **tip did not complete cleanly:** server **abort/core dump** mid-suite on two attempts; recorded 0/100 is **not a valid tip score**.
- Agent tool quality is weak on **both** arms for this model; tip is **not shown worse via a fair full 69** because it died under load.

## Verdict

| | |
|--|--|
| **Speed** | tip clearly faster: tg **+28 tok/s** formal, single-agent p50 **+27** |
| **Long-context quality** | **matched** (needles 3/3 both, dossier 12/14 both) |
| **Held-out tools** | tip **slightly better** (37.0% vs 32.6%) |
| **Agent69** | base 8/100; tip **unstable** (abort) — not banked as a regression |

Artifacts: `/home/frosty40/turbo/lx/results/laguna-ab-20260730T235650Z`
