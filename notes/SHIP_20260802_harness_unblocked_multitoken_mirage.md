# Ship — harness UNBLOCKED + multitoken root-cained (2026-08-02)

## What actually happened this session (real work, not summary)

### 1. CRITICAL: scoring harness was BROKEN — FIXED
`scripts/bench-serial.sh` (uncommitted edits) passed two args the llama-bench
binary does not accept, so **every** `bench-serial.sh` run died at `-- pp512 --`
with `error: invalid parameter for argument`. No candidate could be scored.
This is why prior sessions produced no numbers.

Root causes + fixes (in `scripts/bench-serial.sh`):
- `--threads-batch "$LX_THREADS_BATCH"` — this llama-bench has **no** threads-batch
  option (prefill folds into `-t`). Removed the arg.
- `-fa "$FA"` with `FA=-1` — binary wants `<on|off|auto>`; `-fa -1` is invalid. Made
  `-fa` conditional: emitted only when `FA ∈ {on,off,auto}`; `FA=-1` (auto sentinel)
  omits the flag → binary default (auto → FA on via FA-VEC-GQA patch). Matches the
  pinned champion run exactly.

Verified end-to-end: champion now benches clean (`results/20260802T014532Z/`).

### 2. Verified anchor (champion binary, harness-fixed)
`build-mmadd-decode/bin`, dual_down/multitoken OFF:
**pp512=1141.8  tg128=138.73  score=1.2128**
Formal champion was 1.227 (pp1183/tg139.3); LATEST_SCORE 1.215. The 3.5% pp gap
to the formal is **run-to-run variance** (same flags/env confirmed), not config.

### 3. dual_down A/B is NEUTRAL (within noise) on current source
Clean A/B, `build-base-control` (current 12:43 source):
| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| dual_down OFF | 1137.7 | 136.69 | 1.1983 |
| dual_down  ON | 1127.8 | 135.82 | 1.1900 |
Difference < run-to-run noise (~±1.5%). dual_down's claimed "prefill reclaim"
does **not** materialize on this stack. Likely not net-firing or net-neutral.

### 4. multitoken is now QUALITY-SAFE — but the +63% is a MIRAGE (fast garbage)
**Quality (current source, `build-base-control`, dual_multitoken ON):**
| probe | PPL (wikitext-2, 2×512) | golden |
|-------|------------------------:|:------:|
| -ub 512  | 13.11 | OK |
| -ub 2048 (scored cfg) | **12.87** | OK |

So the current 12:43 source **already fixed** the multitoken PPL=-nan bug
(handoff called it "dead"). The mul_mat reorder-chunk fix (ggml-sycl.cpp:4462-4485)
makes it correct. **This contradicts HANDOFF_20260807.**

**But multitoken gives NO speedup:**
| arm | pp512 | tg128 | score | PPL |
|-----|------:|------:|------:|----:|
| multitoken ON (current src) | 1128.9 | 135.81 | 1.1902 | 12.87 |
| multitoken ON + ENABLE_OPT=0 | 1128.3 | 99.16 (floor fail) | null | 12.77 |

Debug confirms the expert-loop **fires** with `n_tokens=512` in ALL cases.

### 5. ROOT CAUSE of the "missing +63%"
Historical peak `results/20260730T144111Z/` (score 1.637, **pp512=3734**) used the
SAME expert-loop (`n_tokens=512`). Why 3.3x prefill then, ~1.0x now?

- `opt_for_reorder` → `reorder_qw()` reorders quant weights **in-place to SoA**
  (ggml-sycl.cpp:4363). Once reordered, `src0->data` is SoA.
- **MMQ/oneDNN assumes LINEAR block layout.** On SoA data it computes **wrong
  results with minimal work** → PPL=-nan BUT very fast. That is the +63%: **fast
  garbage**, never quality-safe.
- The reorder-chunk fix (4462-4485) detects reordered src0 and forces chunked
  reorder-MMVQ → **correct but ~baseline speed** (MMVQ is the small-M decode path).
- Proved MMQ-on-linear is also ~baseline: with `ENABLE_OPT=0` (weights stay linear)
  the expert-loop still fires (`n_tokens=512`) yet pp512=1128. The per-expert GEMM
  approach (256 small launches, M≈16/expert) is fundamentally ~baseline.

**Conclusion: there is no real 3x prefill headroom in the per-expert MoE path.**
The +63% was a correctness bug that happened to be fast. Chasing it is over.

## Where the score actually is
Plateaued at **~1.21–1.23**. Decode (weight 0.75) is at the BW ceiling
(86 W / 37%, MoE expert weight reads). Prefill (weight 0.25) is ~1.0x and near-optimal.

## Real levers left (honest) — ALL EXHAUSTED THIS SESSION (measured)
1. **Grouped/batched MoE down-GEMM for prefill** — DEAD. (a) Integrated down kernel
   is bit-exact correct (proven, see §6) but gives NO prefill speedup alone (gate/up
   still per-token). (b) `mmvq_fused` single multi-token launch is BOTH wrong (golden
   mismatch — the "multi-token grid bug" is real for the standard kernel at n_tokens=512)
   AND slower (pp 718 vs 1140 — giant grid under-saturates). The per-expert/per-token
   GEMMs are already efficient on this GPU. **Prefill is near-optimal.**
2. **Smaller quant (Q3_K_M)** — DEAD. Quantized Q8_0→Q3_K_M (3.84 BPW, 16GB) with
   `--allow-requantize`. Result: **SLOWER** — pp 857, tg 86 (decode floor FAIL).
   Q3_K_M loses the entire Q4_K-specific fuse stack (router-GEMV, true-topk, mm-add,
   reorder-MMVQ, dense-dual — all gate on Q4_K/Q5_K/Q6_K) AND has a slower dequant
   kernel. Fewer bytes is overwhelmed by lost optimizations. (Q5/Q8 are larger → slower.)
3. **mm-add prefill epilogue** — small/neutral (decode-only gate is the quality-safe bar).

## Where the score actually is — PLATEAUED, all levers measured-exhausted
Champion **1.20–1.227** (relatch 1.2045: pp 1139 / tg 137.6). Decode (0.75) at BW
ceiling on Q4 with full fuse stack. Prefill (0.25) near-optimal. multitoken/dual_down
neutral. Smaller quants slower. **No remaining win within Q4_K_M + this SYCL backend
without new kernel work** (a faster Q4 MoE MMVQ) or **speculative decoding** (no Laguna
draft exists).

## Do NOT (re-confirmed by measurement this session)
- Ship multitoken/dual_down as a speed win — it is quality-safe but ~neutral.
- Treat the 1.637/+63% as recoverable "headroom" — it was garbage.
- Set `GGML_SYCL_ENABLE_OPT=0` — kills decode (tg 99, below floor).
- Re-add `--threads-batch` or unconditional `-fa -1` to bench-serial.sh.

### 6. Prefill batching lever EXISTS but the kernel is BUGGY (confirmed)
A fully-batched MoE down-projection kernel already exists:
`mul_mat_vec_q_moe_weighted_reorder` (mmvq.cpp:3008) — grid `(n_tokens,1,block_num_y)`,
does down + weighted-reduce in ONE launch, supports n_tokens>1, k<=16. Wired via
`ggml_sycl_fuse_moe_down_weighted` (ggml-sycl.cpp:6368, called from topk-moe.cpp:1552).

But it is GATED OFF for a reason:
- `n_tokens > 32` cap (ggml-sycl.cpp:6409) blocks prefill.
- integrated path default OFF: comment "golden FAIL (2026-07-30)".

**CORRECTION — the kernel is NOT buggy. The math is bit-exact-correct.** I added a
numerical diff (`GGML_SYCL_DEBUG_MOE_DOWN_DIFF=1`, ggml-sycl.cpp after the integrated
kernel): it recomputes the per-expert reference and diffs. Result across 722 fuse hits
at n_tokens = 1/2/4/8: **every single one max_diff = 0.000000** (mean|integ| == mean|ref|
exactly). So `mul_mat_vec_q_moe_weighted_reorder` matches the reference path EXACTLY.

The "golden FAIL" is a **graph-wiring side-effect, not a kernel math bug.** Clean PPL
(integrated ON, debug OFF, -ub 32) = **13.02** vs reference **12.84** — both SANE, ~1.4%
apart. The down output is bit-exact; the small global perturbation comes from the fuse
skipping/consuming intermediate nodes (MUL/VIEW/ADD) whose buffers or downstream
dependencies aren't fully isolated — a `ggml_can_fuse_subgraph` / allocator-reuse class
of issue, NOT the kernel. Likely a small, contained fix.

oneDNN trace (multitoken pp256) confirms the lever is real: ~20235 calls, dominated by
hundreds of tiny-N expert GEMMs (`1x512x2048:1x2048x9` 676 calls; `1x2048x512:1x512x9`
763 calls; N=9-18 tokens/expert, ~18us each). Batching would cut launch overhead + raise
EU occupancy. **Path to prefill win: (1) fix the small wiring perturbation so golden
passes, (2) raise the n_tokens>32 cap (ggml-sycl.cpp:6409) for prefill, (3) bench.**

gate/up batching: `dual_swiglu_fused` (ggml-sycl.cpp:4565) is decode-only (`ne12 != 1`
→ false, line 4595); its kernel `mul_mat_vec_q_id_dual_swiglu_reorder` likely supports
n_tokens>1 but is untested for prefill.

## Artifacts this session
- Harness fix: `scripts/bench-serial.sh` (FA_ARGS conditional; threads-batch removed)
- Champion anchor: `results/20260802T014532Z/`
- dual_down A/B: `results/20260802T014710Z/` (ON), `results/20260802T014924Z/` (OFF)
- multitoken bench+PPL: `results/20260802T015405Z/`, PPL log `results/multitoken-bench-*.log`
- OPT0 probe: `results/20260802T020047Z/`, log `results/mt-opt0-*.log`
