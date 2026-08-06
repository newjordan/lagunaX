# Absolute hardware limit — Laguna XS 2.1 on Arc B70

**Mission:** max serial tok/s on this GPU.  
mlx.fast score is a yardstick only.

**Ops:** one Level-Zero client at a time on this card. Concurrent bench/PPL wedges
xe (`xe_validation_lock` → reboot). See `notes/B70_NO_CONCURRENT_GPU.md`.

## Current ceiling (serial, 2026-07-29)

| Config | pp512 | tg128 | Notes |
|--------|------:|------:|-------|
| **control + Q4_K_M (ship)** | **~1135–1147** | **~107.5–107.9** | flag/env plateau |
| control + Q5_K_M | **1185** | 100.3 | +prefill / −decode (more bytes) |
| package + Q4_K_M | 819 | 105.0 | package loses solo |

**Multi-slot (other campaign, other metric):** ~513 mean / 522 max @ np64 — not serial.

## Bottleneck class (measured)

| Phase | avg W | % of 230 W | tok/s | Class |
|-------|------:|-----------:|------:|-------|
| decode (tg) | **86** | **37%** | 107.6 | **memory-bandwidth** (MoE expert weight reads) |
| prefill (pp4096) | **87** | **38%** | 2066 | also well under cap; MoE small-GEMM / BW limited |

Decode is **not** power-bound. More EU/compute knobs will not unlock big wins.  
Expect gains from: fewer bytes/token, better MoE decode kernels, or multi-token methods (spec/MTP).

## Wave1: runtime knobs EXHAUSTED

All env/flag arms within **±0.4% noise**. Losers only:
- FA off → tg **72** (−33%)
- SYCL graph on → pp −2.6%

Artifacts: `results/abs-serial-w1-20260729T201511Z/`

## What can still move the absolute limit

Ordered by leverage for **serial**:

1. **MoE decode kernels** (GEMV / expert gather / fused dequant+mul) — primary BW path  
2. **Smaller quant that stays correct** (IQ3/IQ2/TQ etc.) — fewer bytes/tok if quality holds  
3. **Speculative decoding / draft** — if a Laguna draft exists  
4. **Prefill-only path** — longer windows already ~2k t/s @ pp4096; pp512 is overhead-heavy  
5. **Not** more multi-slot packing (done @ 513; different product number)

## Ship (serial absolute)

```bash
source /home/frosty40/turbo/lx/env.sh
# binary: base-control
# model:  Q4_K_M
-ngl 99 -fa on -ub 4096 -b 8192 -ctk f16 -ctv f16 -t 16
# GGML_SYCL graph: keep OFF
```

~**108 tg128 · ~1140 pp512** on this hardware with this stack.
