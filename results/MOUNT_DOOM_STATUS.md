# Mount Doom — live board (2026-08-02, harness-unblocked + multitoken root-caused)

## Score: PLATEAUED at ~1.21–1.23 (champion unchanged)

| | pin | **ship champion** | goal | stretch |
|--|--:|--:|--:|--:|
| pp512 | 1139 | **~1142–1183** (variance) | — | — |
| tg128 | 107.35 | **~138.7–139.3** | — | — |
| score | 1.000 | **~1.21–1.227** | ≥1.250 | ≥1.300 |
| golden | — | OK | OK | OK |

- Champion binary: `treebeard-base-control-latest/build-mmadd-decode/bin` (old 06:38 source + decode-only mm-add binary patch)
- Champion anchor this session (harness-fixed): `results/20260802T014532Z/` → **1.2128**
- Baseline pinned `baseline/baseline.json` — **do not re-pin**. Golden checked-in — **do not re-pin**.

## What changed understanding this session (evidence in `notes/SHIP_20260802_harness_unblocked_multitoken_mirage.md`)

1. **Scoring harness was BROKEN — FIXED.** `bench-serial.sh` passed `--threads-batch`
   (no such option) and `-fa -1` (invalid); every bench died at `-- pp512 --`. This is
   why no candidate could be scored. Fix: drop threads-batch; make `-fa` conditional.
2. **multitoken is now QUALITY-SAFE** (PPL 12.87 @ -ub 2048, golden OK) on the current
   12:43 source — contradicts `HANDOFF_20260807` which called it "dead".
3. **The +63% (score 1.637, pp512=3734) was FAST GARBAGE.** MMQ/oneDNN on in-place
   reordered (SoA) quant weights computes wrong results with minimal work. The
   reorder-chunk fix (ggml-sycl.cpp:4462-4485) makes multitoken correct but ~baseline
   speed. Proved: per-expert GEMM (M≈16) is ~baseline whether MMVQ-on-reordered or
   MMQ-on-linear (OPT=0). **There is no real 3x prefill headroom in the per-expert path.**
4. **dual_down A/B is neutral** (within ±1.5% noise) on current source.

## Real levers left (honest, ordered)
1. **Grouped/batched MoE down-GEMM** (prefill) — replace 256 per-expert down launches
   with one batched GEMM. Potential ~+15-20% pp → score ~1.26. New kernel, medium risk.
2. **Smaller quant (Q3_K_M / IQ3)** — fewer bytes/token → faster decode (weight 0.75).
   Not on disk; must quantize. Golden is Q4-pinned → needs golden re-capture. Quality risk.
3. mm-add prefill epilogue (GEMM-correct) — small.

## Ops
One GPU client. `notes/B70_NO_CONCURRENT_GPU.md`. Lock: `results/.b70-gpu.lock`.
GPU is FREE (only a CPU-side embeddings server runs).

## DO NOT (re-confirmed by measurement)
- Ship multitoken/dual_down as a speed win — quality-safe but neutral.
- Treat 1.637/+63% as recoverable headroom — it was garbage.
- `GGML_SYCL_ENABLE_OPT=0` — kills decode (tg 99 < floor).
- Re-add `--threads-batch` or unconditional `-fa -1` to bench-serial.sh.
- Re-pin baseline/golden; FA-off; graph-on; concurrent GPU jobs.
