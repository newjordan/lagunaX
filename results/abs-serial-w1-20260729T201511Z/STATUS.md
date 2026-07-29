# Serial absolute-limit — wave1 COMPLETE

**Date:** 2026-07-29  
**Binary:** treebeard base-control  
**Model:** Laguna-XS-2.1-Q4_K_M  
**Baseline:** pp512=1139.2 · tg128=107.35

## Verdict

**Runtime knobs are exhausted within noise (~±0.4%).**  
Serial on this binary+quant is already at the **flag/env ceiling**.

| Peak axis | Arm | Value |
|-----------|-----|------:|
| Best score | t32_ub4k | **+0.39%** (noise) |
| Best tg128 | dnn_on_ub4k | **107.86** |
| Best pp512 | ub8k_b8k | **1146.9** |

Absolute serial band on control+Q4: **~107.3–107.9 tg · ~1130–1147 pp**.

## What does NOT move the needle (serial)

| Lever | Result |
|-------|--------|
| `-ub` 512→8192 | flat (pp512 fits in one ubatch already) |
| `-b` 2k→8k | flat for solo |
| oneDNN on/off | noise |
| fusion off | slight pp dip, tg flat |
| topk MoE off | noise on Laguna path here |
| force MMQ | flat |
| SYCL graph ON | **pp −2.6%** — keep graph off |
| FA off | **tg −33%** (72) — never |
| threads 8/16/32 | noise |

## Ship (serial, control, Q4)

```text
-ngl 99 -fa on -ub 4096 -b 8192 -ctk f16 -ctv f16 -t 16
# graph OFF (default / DISABLE_GRAPH=1)
# oneDNN: either (noise)
```

## Where the absolute limit actually lives next

Flag space is done. Remaining headroom is **not** another env sweep:

1. **Bottleneck class** — power profile under decode (memory vs compute vs launch)
2. **Kernel path** — MoE expert GEMV/GEMM, Q4 dequant, attention on B70 SYCL
3. **Binary** — package vs control solo (package loses multi-slot win ≠ solo)
4. **Quant** — Q5/Q8/IQ if bytes/token or dequant path helps (or hurts bandwidth)
5. **Speculative / MTP** — if draft model exists for Laguna
6. **Multi-slot** — already plateaued at ~513 agg @ np64 (separate claim)

Board: `BOARD.json`
