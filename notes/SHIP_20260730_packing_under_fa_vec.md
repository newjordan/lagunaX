# Research — packing / dual-down re-probe under FA VEC tip (2026-07-30)

## Status: **no tip change** (tip rebench confirms noise band)

| arm | tg64 / formal | notes |
|-----|---------------|-------|
| **tip FA VEC** formal | tg128 **135.0** / +59.61% | `20260730T134432Z` |
| tip rebench | tg128 **134.83** / **+59.59%** | `20260730T135411Z` |
| dual sgs=1 (tip) | tg64 135.23 | default |
| dual sgs=2 | tg64 135.17 | flat |
| dual sgs=4 | tg64 134.69 | slight under |
| dense dual sgs=1 | tg64 135.31 | default |
| dense dual sgs=2 | tg64 135.52 | noise +0.2 |
| dense dual sgs=4/8 | ≤135.31 | no win |
| dense dual+down residual ON | tg64 133.89 | under tip |
| down sgs=4 / 16 | ~135.2–135.4 | flat vs sgs=8 |

## Tip rebench

| metric | stamp | value |
|--------|-------|------:|
| pp512 | 20260730T135411Z | 3729.7 |
| tg128 | 20260730T135411Z | ~135.0 |
| score | | **+59.59%** |

Golden OK under re-captured VEC oracle.

## Conclusion

Under FA VEC tip, packing / dense dual-down levers remain **closed** (flat or under).
No default change.

## Next

1. New theory under +59.6% tip (not packing thrash).
2. Prefill residual2 / GEMM-post remain closed.
3. Optional: attn gate GEMV+softplus (low expected ROI).
