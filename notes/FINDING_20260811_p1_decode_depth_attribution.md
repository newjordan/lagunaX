# FINDING 2026-08-11 — P1: decode-at-depth is FA-kernel-bound (~42% of peak BW on the full-attention layers); q8_0 KV is a dead lever; Laguna is interleaved-SWA (only ~10 deep layers)

Receipts: `results/p1-decode-depth-20260811T214831Z/` (depth-sweep.md,
depth-sweep-q8kv.md, energy.tsv, lx-chrono.bin, drivers). Binary: the C4
candidate (`lx-reorder-multicol`, knob ON — decode path identical to champion;
knob only accelerates the -d prefill priming). Loop charter iteration 1.

## Architecture (GGUF metadata, hand-parsed — settles the KV arithmetic)

`sliding_window = 512`, per-layer `head_count = [48,64,64,64, 48,64,…]`
(period 4): **~10 full-attention layers** (48 Q heads, rope 500K/yarn×32) and
**~30 SWA-512 layers** (64 Q heads, rope 10K). KV uniform: 8 KV heads × 128 =
4 KiB/token/layer (f16 K+V). Structural proof SWA-KV allocation is live: full
40-layer KV at 131072 would need ~21 GiB (doesn't fit beside the 18.9 GiB
model); 10 full + 30×512 ≈ 5 GiB — and serve-laguna runs 131072 on this card.
Max-context decode therefore only pays deep-KV reads on ~10 layers.

## Depth sweep (tg128, f16 KV, ub2048, fa on, r=2)

| depth | t/s | ms/token | Δms per 4096 depth |
|---|---|---|---|
| 0 | 152.52 ± 0.23 | 6.556 | — |
| 4096 | 122.54 ± 11.30 | 8.161 | +1.605 (first step: window fill + fixed) |
| 8192 | 114.24 ± 9.26 | 8.753 | +0.592 |
| 16384 | 98.38 ± 6.48 | 10.165 | +0.706/4K |
| 24576 | 86.54 ± 5.31 | 11.555 | +0.695/4K |

Asymptotic slope **0.171 µs per token-of-depth**. Peak-BW cost for the ~10
full layers: 40 KiB/token-of-depth ÷ 567 GB/s = **0.072 µs** ⇒ the at-depth
FA read runs at **~42% of peak bandwidth**. Device power during the sweep:
139 W of the 230 W cap (60%) — not power/compute-saturated. ⇒ The wall is
the FA kernel's memory-access efficiency at long KV, not bytes, not power.

## q8_0 KV probe — DEAD, do not retry

| depth | f16 KV t/s | q8_0 KV t/s |
|---|---|---|
| 0 | 152.52 | 111.58 (−27%) |
| 8192 | 114.24 | 42.45 (−63%) |
| 24576 | 86.54 | 26.48 (−69%) |

Halving KV bytes should *win* if BW-bound; instead q8_0 KV collapses decode at
every depth — the quantized-KV FA path on this stack is catastrophically worse
(kernel fallback class). Dead lever on this build; add to the do-not-try table.

## Prize arithmetic (why this matters for max context)

If the FA at-depth linear term reached peak BW: d24576 token 11.56 → ~9.3 ms
(**+24% decode at 24.5K**). At d≈100K the linear term dominates: measured-slope
token ≈ 24.7 ms (40 t/s) vs BW-ideal ≈ 14.8 ms (67 t/s) — **~+65% at 100K**.
The lever grows with context, which is the stated priority.

Note the constraint from FINDING_20260811_fattn_fullchain_backport_no_win:
master's FA chain was **−12%** at-depth decode on this box — importing kernels
is receipted-worse; the work is making OUR TILE-FA path BW-efficient at long
KV (GQA 48Q/8KV, head_dim 128), or overlapping the 10 deep layers' reads.

Chrono ledger captured (lx-chrono.bin) but per-name attribution is sync-point
polluted (known artifact) — not load-bearing here; the sweep+power+probe carry
the conclusion.

## Next (iteration 2)

Read the TILE-FA decode path for the at-depth shape (which kernel, workgroup
geometry, KV access pattern per 8-KV-head/128-dim GQA at ne1=1); find why
effective BW is 42%; candidate mutations behind env knobs, gated per AGENTS.md
+ canonical triangulation.

## Iteration 2 addendum — mechanism located; env levers measured

Recon (fattn.cpp / fattn-vec.hpp / fattn-common.hpp, C4 tree): decode (ne1=1)
runs `flash_attn_ext_vec` D=128 WG=256 warp16 — NOT tile. Mechanism of the 42%
efficiency: `ncols2` is hardcoded 1 in the vec launch (fattn-vec.hpp:623,630)
⇒ one workgroup per Q head ⇒ the shared KV head is re-read gqa_ratio (6)×:
at 24.5K depth, 604 MB fetched per token where 100.7 MB is unique (L2 absorbs
part — that is why the slope is 2.4× ideal rather than 6×). Split-K exists but
`parallel_blocks` is capped at max_wg_per_cu=4 (fattn-common.hpp:1048-1092;
a commented-out uncap sits at :1070) ⇒ 192 WGs at 48 heads, each serially
walking 6144 KV rows at 24.5K. No mask early-exit at decode (KV_max null,
gate at fattn-common.hpp:1013). K-pass SIMD16 issues two disjoint 128 B
segments per load (lanes split across rows 8 apart) — secondary suspect.

Env-only probes at d {0, 8192, 24576} (fa-tile.md, fa-nt128.md):
  baseline           152.52 / 114.24 / 86.54
  FORCE_TILE=1       147.82 /  97.49 / 63.61   DEAD (tile kernel loses more
                                               than GQA batching saves)
  DECODE_NTHREADS=128 151.45 / 115.92 / 88.63  +2.4% at depth, −0.7% d0 —
                                               marginal, floor-grazing
Ranked mutations (iteration 3+, all env-gated default-OFF on the C4 branch):
  M1 parallel_blocks env override (3-line; more split-K, shorter serial walks)
  M2 vec-kernel ncols2=2 for gqa%2==0 (halves KV re-read 6→3×; the principled
     fix; kernel surgery in fattn-vec.hpp + launch_fattn ncols2 plumbing)
  M3 K-load lane mapping → one contiguous 256 B segment per SIMD16
Prize bound unchanged: +24% decode at 24.5K, more at 100K+.

## Iteration 3 — M1 LANDED: split-K width override, +10% at-depth decode

Commit `d61bdf435` (lx-reorder-multicol branch, stacks on C4), .so
`94015650…`. `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=<N>`, default 0 = stock
(pb=0 leg re-measured bit-consistent with the pre-mutation binary: 86.51 vs
86.54). Receipts: `results/p1-m1-pb-20260811T220715Z/`.

| pb | tg128 d0 | d8192 | d24576 |
|---|---|---|---|
| 0 (stock) | 152.67 | 114.38 | 86.51 |
| 8 | 152.55 | 117.00 | 92.53 |
| **16** | **152.79** | **118.29** | **95.60 (+10.5%)** |
| 32 | 152.69 | 115.90 | 94.73 |
| 96 | 152.81 | 116.29 | 85.12 (over-split: combine tax) |

Real-text 23K (llama-cli, ship env + C4 knob): decode **81.8 → 89.7 t/s
(+9.7%)**, prefill unchanged (1542.8 vs 1537.7). Gates: golden-smoke PASS in
both knob states (short-kv geometry is unchanged by construction:
ntiles_KQ=1). 23K greedy A/B: one near-tie flip ~40 tokens in, both
continuations coherent — reduction-order class (stock already fp-combines 4
split-K partials; pb=16 combines 16; same mechanism, different grouping).
**Instrument gap recorded: the pinned KLD gate is prefill-only and cannot see
decode-FA numerics; no decode-logit-distance instrument exists yet.** M1 ships
env-gated default-OFF like C4.

Next: M2 — vec-kernel ncols2=2 GQA batching (halve the 6× KV re-read).
Post-M1 estimate: ~+9-12% more at 24.5K.

## Iteration 4 — M2 (vec GQA head-batching) FALSIFIED and reverted

Implemented ncols2=2 head-pair batching for the vec decode kernel (2 Q heads
sharing a KV head per workgroup, one K/V read for both; engage marker
confirmed firing). Receipts: `results/p1-m2-gqa-20260811T*/` (m2-gqa.patch
preserves the full diff; .so c7f8b947). Measured, tg128:

| config | d0 | d24576 |
|---|---|---|
| stock | 152.11 | 86.53 |
| gqa2 | 144.41 (−5%, FLOOR VIOLATION) | 88.27 |
| gqa2+pb16 | 144.82 | 91.45 |
| gqa2+pb32 | — | 90.64 |
| gqa2+pb64 | — | 88.87 |
| **M1-only pb16 (reference)** | **152.79** | **95.60** |

GQA batching loses to per-head workgroups at every split width. Mechanism:
the 6× KV re-read was already largely served from L2 (known: stock slope is
2.4× BW-ideal, not 6×), so halving DRAM re-reads bought little, while
halving z-parallelism (48 → 24 head-pair groups) and doubling per-WG state
(2× Q registers, KQ scratch, softmax bookkeeping) cost real occupancy — the
d0 −5% is the per-WG overhead alone. Reverted from the tree (patch kept in
the receipt dir); branch stays at d61bdf435 (C4+M1).

P1 standing after iteration 4: **M1 (+10% at-depth) is the P1 win**; M3
(K-load lane contiguity: one 256 B row per SIMD16 instead of 2×128 B
segments) is the remaining bounded FA idea. The L2-absorption evidence also
lowers M3's expected value — the loads may already coalesce adequately at
the L2 interface. Next iteration: M3 only if cheap; otherwise pivot to P2
(expert-tile GEMM evidence unit at serving geometry).

## Deep-depth close-out (2026-08-12, RL-workload priority)

First measurement of this model's decode at RL-loop depths, full stack,
`results/p1-deepdepth-20260812T*/`. pb sweep {16,32,64} at d {49152, 98304,
122880}: **pb=16 is the flat optimum at every depth** (wider splits pay
combine overhead faster than they add parallelism) — no depth-scaling patch
needed; the shipped knob is already right. Stock-vs-shipped A/B (tg64):

| depth | stock pb=4 | pb=16 | M1 win |
|---|---|---|---|
| 49152 | 64.16 | 69.63 | +8.5% |
| 98304 | 42.20 | 47.29 | +12.1% |
| 122880 | 36.02 | **40.80** | **+13.3%** |

The split-K win grows with depth as the serial-walk model predicted. Decode
at the full 131K window: **40.8 t/s** (was ~36 stock, never measured before
today). Remaining headroom to BW-ideal at 122K is still ~1.6x — M3 and the
future FA rework own it.
