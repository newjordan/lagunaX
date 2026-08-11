# PLAN 2026-08-11 — attack dossier: real-text serving prefill (the ~6x expert-loop headroom)

Read-only reconnaissance. No GPU runs, no builds, no kernel edits were performed
producing this document. Every code citation is `file:line` in the **live serving
tree** `/home/frosty40/turbo/worktrees/lx-champion-tier12` (= `scripts/serve-laguna.sh`
`REPO`, tag `lx-champion-1.3105-20260810`, hub `c7d3bfe6d`).

Target: `llama-server`/`llama-cli` real-text prefill 308.8–310.8 t/s vs llama-bench
synthetic 1895.9 t/s at identical geometry (FINDING_20260810_serving_prefill_routing_skew).

---

## 0. Three corrections to the brief, up front

These change the plan, so they lead.

**(a) `gate-up-concat` cannot be "re-measured on real text" as it stands — it is
unreachable code on this model.** `GGML_SYCL_LX_GATE_UP_CONCAT` is read at
`ggml-sycl.cpp:7246` and used only at `ggml-sycl.cpp:7398-7443`, i.e. *inside*
`ggml_sycl_mul_mat_id_dual_down_multitoken_expert_loop` (7178). That function is
reached only from the graph fuse `ggml_sycl_fuse_moe_dual_swiglu` (6117), whose
`types_ok` predicate (6222-6234) requires
`down_w->type ∈ {Q4_K,Q5_K,Q6_K} && down_w->type == gate_w->type`. Laguna's
`ffn_down_exps` is **IQ4_NL** and `ffn_gate/up_exps` are **Q6_K**
(`results/diag-pp512-20260809T030915Z/gguf-tensor-types.txt`: q6_k 379, iq4_nl 60,
f32 239; iq4_nl is exactly the 39+21 down projections). `types_ok` can never pass.
The same gate is re-asserted inside the expert-loop function itself (7191-7196).
On top of that, `scripts/serve-laguna.sh:57` exports `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1`,
which kills the fuse a second time (6156-6165). The 2026-08-10 receipt
`results/20260810T132032Z` (`note: "lx-gate-up-concat: merged gate+up GEMM per
expert in pp512 expert loop"`, pp512 1130.9 vs 1174 champion) therefore measured a
**pure control** — and it was additionally run on
`treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench`, the exact binary
of the LX_BIN env-order contamination window (FINDING_20260810_lxbin_env_order_contamination).
This was already concluded once in
FINDING_20260810_realtext_prefill_device_bound_master_4x ("gate-up-concat revival:
DEAD on this model regardless of env"). Section 5 gives the *correct* replacement
action: re-implement the concat in the path that actually runs.

**(b) "69.9% of wall in ffn_moe_gate" is a sync-point attribution, not a GEMM
attribution.** The chrono ledger stamps *host* spans. `ffn_moe_gate` is the first
of the three MUL_MAT_ID ops in each MoE layer, so it is the op that pays the ids
D2H + drain (`ggml-sycl.cpp:6688-6697`; the `[lx-ids-once]` memo at 6666-6680 hands
the memoized sort to `ffn_moe_up`/`ffn_moe_down` for free). Its bucket therefore
absorbs *all* device backlog queued by everything before it. That is exactly why
gate = 60.5 ms and up = 11.1 ms **at identical shape, type and call count** —
a 5.4x asymmetry that arithmetic cannot produce. Corroboration:
`WAIT_TOTAL_NS` on the *guarded* wait is only 0.19 s of 76.6 s because the
unguarded `stream->memcpy` at 6691 has already drained the queue, and gdb sampling
found 504/518 samples idle with the busy signature in `sched_yield` inside UR
`appendKernelLaunch`. Conclusion: **no measurement in the campaign has yet
localised real-text prefill device cost to a specific op class.** Section 3 is
about fixing that, for free, before mutating anything.

**(c) `lx/mmid-device-batched-sgemm` contains zero batched-sgemm work — nothing to
salvage.** In `/home/frosty40/turbo/worktrees/lx-dev-mmid-batch`:
`git log` = `c7d3bfe6d` (champion preserve) → `3337c0f11` (backport oneMKL XMX FA +
oneDNN SDPA) → `f8638cfa9` (arc770 FLASH_ATTN_EXT fix). `git reflog` shows only
`reset: moving to HEAD` then those two cherry-picks — the branch was created and
immediately repurposed for the FA backport attempt; no sgemm commit ever existed.
Working tree is one dirty file (`ggml/src/ggml-sycl/fattn-vec.hpp`, 1 line). No
stash relevant (`stash@{0}` is an unrelated `agent/treebeard-single-wavefront`
server-context WIP). **The only batched-GEMM code that exists is the
`GGML_SYCL_LX_GEMM_BATCH` block that is already merged into the champion tree
(`ggml-sycl.cpp:6796-6923`), default OFF, and receipted as hanging the B70 —
including after the `[lx-gb-fix-20260810]` OOB fix (FINDING_20260810_lxbin
corrected-measurements section: "reproducible hang (rc=124 at 240 s), the 'strided
gemm_batch dead on B70' verdict stands even on the fixed shape").** Salvage value:
zero. Delete-or-leave, do not revive.

---

## 1. Path map — what actually executes at serving prefill

Serving env (`scripts/serve-laguna.sh:52-59`): `DISABLE_DNN=1`, `DISABLE_GRAPH=1`,
`FUSE_NORM_ROPE=1`, `DISABLE_MUL_MAT_ADD_FUSE=0`, `DISABLE_MOE_DUAL_DOWN=1`,
`DISABLE_MOE_DUAL_MULTITOKEN=1`, `DISABLE_QKV_SHARED_QUANT=1`.
(Note: `GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN` **does not exist** anywhere in this
tree — `grep -rn MOE_DUAL_MULTITOKEN ggml/src/ggml-sycl/` returns nothing. It is an
inert line inherited from an older tree. Harmless, but it is not doing what the
comment claims.)

Prefill (`ne12 = n_tokens = 2048`) walks:

| # | site | what it costs |
|---|---|---|
| 1 | `ggml-sycl.cpp:6631` `ggml_sycl_mul_mat_id` | entry, once per MUL_MAT_ID node (gate, up, down) per layer |
| 2 | `6647` → `6438` `..._mmvq_fused` | **bails immediately**: cap `ne12 ≤ 64` (6450) and `ne12>1` needs `ENABLE_MMID_FUSED_BATCH` (6452), unset. Device-routed path is decode-only. |
| 3 | `6666-6680` `[lx-ids-once]` memo | gate misses, up/down hit. Saves 2 of 3 syncs per layer. |
| 4 | **`6691` `stream->memcpy(ids_host, ids_dev, …)`** | **the unguarded D2H.** Blocking `memcpy` on an in-order queue = full drain, and it is *outside* `lx_chrono_wait_guard`, so `WAIT_TOTAL_NS` under-reports it to ~0. This is the "drain that hides". |
| 5 | `6696` `stream->wait()` | guarded; sees ~nothing left to drain (that is the artefact in (b)). |
| 6 | `6751` `mmid_counting_sort_rows` | host counting sort → `expert_row_counts/offsets`, `mmid_row_mapping` |
| 7 | `6769` `memcpy(dev_row_mapping, …)` + `6779` `k_copy_src1_to_contiguous` | H2D map + 1 gather kernel; packs 16384 routed rows |
| 8 | `6789-6795` | `GGML_SYCL_LX_DIAG_EXPERT_CAP` reader (break at `6930`) — measurement hook, live in this binary |
| 9 | `6796-6923` | `GGML_SYCL_LX_GEMM_BATCH` block, default OFF, hangs the device (§6) |
| 10 | **`6924-6984` the per-expert loop** | `for i02 in 0..n_as(256)`, skip empty, then `6982 ggml_sycl_mul_mat(ctx,&src0_row,&src1_row,&dst_row)` on the expert slice `src0_original + i02*nb02` |
| 11 | `6953-6983` | `GGML_SYCL_LX_MMVQ_PREFILL` alternative (Q6_K only, 8-col chunks) — default OFF, receipted −17% |
| 12 | `6990-7003` `k_copy_dst_from_contiguous` | 1 scatter kernel back to `dst` |

Inside step 10, `ggml_sycl_mul_mat` (`4660`) resolves to
`ggml_sycl_op_mul_mat<no_quantize_q8_1>(… ggml_sycl_op_mul_mat_sycl)` (`4789`),
and `ggml_sycl_op_mul_mat_sycl` (`2601`) does, **per expert, per projection,
per dispatch**:

| cost | site | size (Laguna: n_embd 2048, n_ff_exp 512, 256 experts, k=8) |
|---|---|---|
| pool alloc + `to_fp16_sycl(src0 slice)` | `2796-2807` (`ne = row_diff*ne00`) | **row-count-independent**: reads 840 KiB (Q6_K) or 576 KiB (IQ4_NL), writes **2.00 MiB** fp16 — the whole expert slice, even for a 1-row expert |
| pool alloc + `to_fp16_sycl(src1)` | `2811-2821` | `src1_ncols*2048*2` B |
| pool alloc `dst_f16` + `dpct::gemm` (oneMKL, DNN off) | `2836-2848` | M=512(or 2048), N=rows, K=2048(or 512) |
| `to_fp32_sycl(dst)` | `2849-2852` | `row_diff*src1_ncols*4` B |

⇒ **4 device launches + 3 pool allocations + 1 oneMKL submission per (expert ×
projection)**, of which the largest single term (the 2 MiB fp16 weight write) is
completely independent of how many tokens that expert was routed. That is the
Zipf tax, mechanically.

Other structural facts worth having:
- `MMVQ_MAX_BATCH_SIZE = 8` (`common.hpp:177`); `can_use_mul_mat_vec_q` caps at it
  (`4657`) → no native quantized GEMM above 8 columns exists (FINDING_20260809).
- `ggml_sycl_supports_reorder_mmvq` (`3981`) includes **Q6_K**, excludes IQ4_NL.
- If the Q6_K expert banks ever get `optimized_feature.reorder` (set lazily by
  `opt_for_reorder_id`, `4628`, called from the **decode-only** fused paths at
  `6470` and `4882-4883`), then the guard at **`4718-4745`** force-routes *any*
  column count to chunked 8-col reorder-MMVQ instead of oneMKL. This is a real
  state-dependent path flip between instruments — see §4/C, where the existing
  receipts falsify it as the 6x cause.
- `reorder_qw_q6_k_moe` (`4302`) reorders **within each expert's `nb02` stride**
  (`base = data_device + e*expert_bytes`), so per-expert slice arithmetic stays
  valid under reorder; and `ggml_get_to_fp16_sycl` already returns
  `dequantize_row_q6_K_sycl_reorder` when the flag is set (`convert.cpp:689`).
  I.e. the fp16/oneMKL path is *already* reorder-safe; the blanket guard at 4718
  is broader than it needs to be. (Its stated justification, PPL −nan on
  oneDNN/MMQ multi-row, is about *those* kernels, not about the fp16 converter.)

---

## 2. Mechanism model — where the ~6x goes

Geometry: 39 MoE layers, ub 2048 → a 23K prompt is 12 ubatches; 2048 tok × k=8 =
16384 routed rows per ubatch per layer.

Measured anchors:
- real text: 308.8 t/s ⇒ 74.2 s / 23K ⇒ **6.2 s per ubatch** ⇒ **158 ms per layer per ubatch**
- synthetic `-p 16384` ub2048: 1895.9 t/s ⇒ 8.64 s ⇒ **1.08 s per ubatch** ⇒ **27.7 ms per layer per ubatch**
- ratio **5.7x**, same binary, same `-ub/-b/-c`, same depth profile

Per-expert call cost, back-solved (E = distinct non-empty experts per layer per ubatch):
```
per-call cost = (per-layer-per-ubatch wall) / (3 × E)
real  : 158 ms / (3 × E_real)      → 206 µs if E_real = 256
synth : 27.7 ms / (3 × E_syn)      → 288 µs if E_syn  =  32
                                    →  36 µs if E_syn  = 256
```
Reference floors from the campaign: oneDNN N-tile floor 17.6–18.9 µs/call
(FRONTIER_20260802_gate_up_gemm_concatenation), `create:cache_hit` 1.18 µs/call
(FRONTIER_20260731_onednn_cache_hit_tax), oneMKL tiny-slice GEMM exec 6.78 µs/call
and per-expert host submission 10–27 µs at pp512
(FINDING_20260809_pp512_wall_decomposition, FINDING_20260810_pp512_hostbound).

Two models fit the data, and **they are not yet distinguished by any receipt**:

**M1 — fixed-cost × active-expert-count.** Per-call cost is roughly N-independent
(~200–290 µs at ub2048 on both instruments) and the wall is
`E × 3 × 39 × c_fixed` per ubatch. Then the 5.7x gap *is* `E_real/E_syn ≈ 8`:
uniform-random tokens (`llama-bench.cpp` `std::rand()%n_vocab`) are far
out-of-distribution, the router's entropy collapses, and synthetic prefill hits a
small expert set with very large N; real text spreads across all 256 with small N.
Consistent with: `LX_QUANT`-class call counts at pp512 (`op_mul_mat_sycl` ≈
15,317/pass ≈ 39×3×126 + dense ⇒ E ≈ 126 at T=512) and with the fp16-weight-write
term (2.00 MiB/expert/projection) being row-count-independent.
**M1 predicts** real-text prefill should improve ~4x going ub 512 → 2048 (E grows
slower than tokens). Measured: 300.5 → 308.8, **+2.8%**. So M1 only survives if
E(ub) grows almost *linearly* with ub in this range.

**M2 — per-row inefficiency.** Cost is dominated by device execution that scales
with total rows, and real text is 6x worse *per row* because its rows are spread
thin across many experts, so every GEMM is a small-N (N≈64) shape running at
~0.65 TFLOPS vs ~3.7 TFLOPS for the fat synthetic shapes. **M2 predicts exactly the
observed ub-invariance** (cost ∝ rows ∝ tokens ⇒ per-token constant), and it also
explains why `-c 131072` vs `-c 32768` and warm-vs-cold are both dead levers.

Sanity bounds that both models must respect (Laguna numbers, B70 peak device BW
**567 GB/s**, copy engine 1362 GB/s — FINDING_20260802_b70_bw_headroom):
- expert-weight traffic per ubatch, all 256 experts:
  `39 × 256 × [gate 0.84+2.0+2.0 | up 0.84+2.0+2.0 | down 0.56+2.0+2.0] MiB ≈ 48 GiB`
  ⇒ **85 ms/ubatch at peak BW = 1.4% of the 6.2 s wall.** Bandwidth is not the wall.
- expert-GEMM arithmetic per ubatch:
  `39 × 2048 × 8 × 3 × 2 × 512 × 2048 ≈ 4.0 TFLOP` ⇒ **200 ms at 20 TFLOPS.**
  Flops are not the wall either.
- device launches per ubatch from the expert loop: `39 × E × 3 × 4`. At E=256 that
  is **119,808 launches/ubatch**; at the 4–6 µs device-visible per-launch floor
  measured in FINDING_20260809_residual_launch_ledger, **0.48–0.72 s/ubatch = 8–12%**.

**⇒ 85–90% of the real-text prefill wall is currently unattributed.** It is not
weight bandwidth, not expert FLOPs, not launch-count floor, not attention (the
full fattn chain backport moved prefill 308.8 → 307.3, FINDING_20260811), not host
dispatch (host 97% idle on real text), not context allocation, not warmup, not
ubatch. The residual is *small-N GEMM execution efficiency* × *expert count* —
which is precisely the M1/M2 axis. **Measure it before mutating.**

---

## 3. Step 0 (mandatory, free, env-only, zero build): attribute the wall

Everything below runs on the **shipped champion binary** — all these hooks are
compiled in and default-inert. Same instrument as
`scripts/validate-fattn-backport.sh:37-48` (llama-cli, `-c 32768 -b 4096 -ub 2048
-n 128 --temp 0 -no-cnv --jinja --chat-template-file …` — the jinja flags are
REQUIRED, the embedded template does not parse). Use `scripts/lib-gpu-lock.sh`.

**0.1 De-confound instrument vs prompt content (the single most important run).**
Every row of the routing-skew table pairs *llama-bench + synthetic* against
*llama-cli/server + real text*. Instrument and content are perfectly correlated;
no leg breaks the tie. Build `correctness/synthetic-23k.txt` (uniform-random
token ids detokenized, or simply a shuffled-vocabulary word salad of the same
token length) and run **llama-cli on it**, same flags as the wikitext leg.
- if synthetic-via-llama-cli ≈ 1900 t/s → routing is confirmed, M1/M2 both live,
  proceed to 0.2.
- if synthetic-via-llama-cli ≈ 310 t/s → **the 6x is the instrument, not the
  routing**, the entire premise of this dossier changes, and the next question is
  what llama-bench does differently (it does not use `common_init_from_params`
  warmup; `common/common.cpp:1421-1457`).

**0.2 Get the (tensor, expert-row-count) histogram — free, exact.**
`GGML_SYCL_DIAG_NAME_QUANT=1` (`ggml-sycl.cpp:2709-2740`) prints at exit
`LX_QUANT_NAMES total=… distinct=…` then one `LX_QUANT_NAME <dst> ty=<type>
ncols=<N> count=<c>` line per (tensor, N). Behaviour-identical run. Do it on the
wikitext leg **and** the synthetic leg. This yields, directly:
E per layer, the rows-per-expert distribution, and the call count — i.e. it
decides M1 vs M2 in one pair of runs. This is the number the whole attack rests on
and nobody has it.

**0.3 Size the expert loop on real text.**
`GGML_SYCL_LX_DIAG_EXPERT_CAP ∈ {8,16,32,64,128,256}` (`6789-6795`, break `6930`;
measurement-only, dst goes stale — quality output is garbage, timing is valid).
Gives the wall-vs-active-expert-count curve on real text, the analogue of the
pp512 curve (cap 0 → 1168 t/s, 64 → 3947, 128 → 2929, 256 → 1170) that established
"per-expert loop ≈ 300–345 ms of a 434 ms pp512 pass".

**0.4 Split GEMM exec from everything else on real text.**
`GGML_SYCL_DIAG_SKIP_QUANT=1` (`2691-2708`) zeroes dst for quantized-src0 calls
only, leaving the F32 router GEMM intact so routing does not collapse (the
methodology guard from FINDING_20260809). Wall delta = the quantized-GEMM
exec+conversion share. Pair with `DIAG_SKIP_TINY_N` / `DIAG_SKIP_LARGE_N`
(`2630-2660`, prints `LX_DIAG_COUNTS tiny=… large=… skip=…`) to split by N.

**0.5 Attention control.** Same real-text leg with `-fa off` and with
`GGML_SYCL_DISABLE_DNN=0`. The fattn backport already says attention is not the
wall; this is a 2-run confirmation on the shipped binary so the claim is anchored
to *this* build.

**0.6 Re-time the hidden drain.** `GGML_SYCL_LX_CHRONO=1 LX_CHRONO_NAMES=1` plus a
`lx_chrono_wait_guard` around the *memcpy* at `6691` — this one needs a 2-line
source change, so it belongs to the first build, not to step 0. Until then, treat
`WAIT_TOTAL_NS` as a lower bound only.

Exit criterion for step 0: a table of real-text per-ubatch wall attributed to
{expert GEMM exec, expert fp16 conversions, ids sync/drain, attention, other},
plus E and the N-histogram. Then, and only then, pick from §4.

---

## 4. Ranked candidate mutations

Ranking convention follows `PLAN_20260807_next_candidates.md`: mechanism →
expected win with arithmetic → risk → kill-switch → measurement protocol.
Every candidate's gate sequence is the AGENTS.md mandatory one, **with the
real-text instrument added as a first-class gate**:

```
mutate kernel source -> build
  -> scripts/golden-smoke.sh                       (LX_BIN explicit)
  -> LX_BIN=<bin> bash scripts/quality-gate-kld.sh (mean_kld / same_top_pct; ignore mean_ln_ppl_ratio 0.016271, it is a u16-store constant)
  -> REAL TEXT A/B: llama-cli -f correctness/wikitext-23k.txt   <-- REQUIRED, per FINDING_20260810_serving_prefill_routing_skew implication 3
  -> bash scripts/bench-serial.sh --note "<what changed>"       (guard the board: decode must not regress)
```
Reference points to beat, same instrument: **real 23K prefill 308.8 t/s / decode
92.4 t/s; tg128 d0 152.5; pp512 1174.** Export `LX_BIN` **before** `source env.sh`
(FINDING_20260810_lxbin_env_order_contamination) and assert the resolved
`libggml-sycl.so.0` sha, as `serve-laguna.sh:63-68` does.

---

### C1 — Expert-blocked quantized GEMM: one kernel launch per MUL_MAT_ID dispatch
**Rank 1. This is the only candidate that can plausibly deliver multiples.**

*Mechanism.* Replace the 256-iteration host loop at `6924-6984` (and everything it
drags in per iteration at `2796-2852`) with a **single** `parallel_for` whose grid
is `(expert-block × row-tile × col-tile)`. Each workgroup: looks up its expert from
the device-side `expert_row_counts/offsets` (already computable on device —
`mmid_device_count_experts` / `mmid_device_exclusive_scan` /
`mmid_device_fill_mapping` exist at `7286-7315` and are used by the expert-loop
fuse), loads its weight tile **straight from the quantized bank**, dequantizes in
registers/SLM, and accumulates against the packed activation rows
(`src1_contiguous`, already produced at `6779`). No fp16 shadow, no per-expert
pool allocs, no per-expert oneMKL submission, no dst fp32 round-trip.
This is exactly item 1 of FINDING_20260809's "What this funds" — *native batched
(ncols>8) iq4_nl/q6_k GEMM for pp slices* — and it is the structural hole that
`gemm_batch` was trying and failing to fill.

*Expected win, arithmetic.* Deletes, per ubatch:
`39 × E × 3` oneMKL submissions, `39 × E × 3 × 4` device launches (**119,808 at
E=256**), `39 × E × 3` × 2.00 MiB of fp16 weight writes + the matching reads
(**≈ 31 GiB/ubatch of pure round-trip traffic**, i.e. 55 ms/ubatch at 567 GB/s that
simply stops existing), and `39 × E × 3 × 3` pool allocations. What remains is the
irreducible 4.0 TFLOP + ~14 GiB of quantized weight reads per ubatch: a **~200 ms
floor** vs the 6.2 s measured. Even at 25% of that ideal the ubatch lands ≈ 0.8 s
⇒ **prefill ≈ 2400 t/s (7.7x)**; at 10% of ideal ⇒ **≈ 1000 t/s (3.2x)**. The
downside case is bounded by the mmvq experience: if the hand-written dequant loop
lands at the same efficiency as the 8-col chunked MMVQ, expect a *loss* (see C1's
risk), which is why step 0.2's N-histogram gates the tile shape choice.

*Risk.* HIGH effort (a real kernel, two quant formats: Q6_K reorder-SoA aware for
gate/up, IQ4_NL linear for down), MEDIUM numerics risk (accumulation order differs
from oneMKL fp16 → KLD, not bit-exactness, is the arbiter — decode is already
KLD-clean on q8_1 activations so the precedent exists). Do **not** reuse the
per-token GEMV shape: `mul_mat_vec_q_id` style has zero weight reuse across tokens
(16384 token-slots × 840 KiB/expert-slice ⇒ 13.4 GiB *per layer*), which is the
mechanical reason `LX_MMVQ_PREFILL` lost 17%. The tile must iterate *tokens* inside
the workgroup, weights outermost.

*Kill-switch.* `GGML_SYCL_LX_EXPERT_TILE_GEMM=1` to enable (default OFF), i.e. the
mutation ships inert; plus `GGML_SYCL_DISABLE_LX_EXPERT_TILE_GEMM=1` as the
hard-off for the serving env once it flips default-on.

*Protocol.* Step 0 first (tile shape from the N-histogram). Then golden-smoke →
KLD → real-text A/B (must beat 308.8) → `bench-serial.sh` (pp512 and **tg128 ≥
152.5**: prefill must never be bought with decode) → depth check at d4096/d16384
against 129.5/102.7.

---

### C2 — Delete the per-expert fp16 weight round-trip: hoist to one active-expert conversion per dispatch
**Rank 2. Bounded, cheap, and it isolates one term of C1 so C1 can be scoped.**

*Mechanism.* Today `to_fp16_sycl` runs once per (expert × projection) at
`2796-2807`, converting the **entire** expert slice regardless of row count.
Replace with a single pre-loop launch that converts only the **non-empty** experts'
slices into one contiguous fp16 scratch, then have the loop point
`src0_row.data` into that scratch and skip the per-call conversion (a flag on the
op, or a dedicated variant of `ggml_sycl_op_mul_mat_sycl`). Byte volume is
unchanged; **launch count drops from `E×3` conversions to `3`**, and the
allocation churn (3 pool allocs per call) collapses.

*Expected win, arithmetic.* Removes `39 × E × 3 = 29,952` conversion launches per
ubatch at E=256. At the 4–6 µs device-visible per-launch floor: **0.12–0.18 s per
ubatch = 2–3%** of wall on the launch term alone; the pool-alloc and
submission-setup savings ride along. **Expected 2–8%, i.e. ~315–335 t/s.** Modest —
this is a scoping probe, not a win by itself.

*Risk.* MEDIUM-LOW. The distinguishing evidence: FINDING_20260810_pp512_hostbound §5
measured **full-tensor** pre-conversion at **−24%** (all 256 experts, 537 MB
fp16/family/layer, serializing the device on giant conversion kernels). This
candidate is the *active-only* variant — same bytes as today, fewer launches — so
the −24% receipt does **not** transfer, but it is a warning that the conversions
are already device-overlapped and the headroom is the launch count, not the bytes.
If step 0.2 shows E ≈ 256, the active-only and full-tensor variants converge and
this candidate inherits the −24% risk directly; **if E is small, this is nearly
free.** Numerics: bit-identical (same converter, same inputs).

*Kill-switch.* `GGML_SYCL_LX_EXPERT_CONV_HOIST=1` (default OFF).

*Protocol.* As above. Additional required leg: real-text at ub 512 **and** 2048 —
if the win is ub-invariant it is a launch-count win; if it grows with ub it was a
conversion-overlap artefact.

---

### C3 — Collapse the three per-layer MoE dispatches into one expert loop (make the multi-token expert-loop reachable for mixed-quant down)
**Rank 3. Also the prerequisite that makes gate-up-concat testable at all.**

*Mechanism.* Relax `types_ok` at `6222-6234` and the mirror at `7191-7196` so the
fuse accepts `gate/up ∈ {Q4_K,Q5_K,Q6_K}` with **any quantized `down`** (Laguna:
IQ4_NL). The expert-loop function `7178-7628` then becomes live for this model and
delivers, per layer: **one** ids resolution instead of three (device sort at
`7280-7315`, one drain at `7350-7357` rather than the `6691` memcpy-drain per
node), **one** `src1` pack instead of three (`7332-7348` vs three copies of
`6779`), no intermediate `glu` write, no second counting sort, and the weighted
reduce fused (`7513+`). It also unlocks `GGML_SYCL_LX_GATE_UP_CONCAT` (§5).

*Expected win, arithmetic.* Removes 2 of 3 per-layer drains — but note the
`[lx-ids-once]` memo (`6666-6680`) already removes those two, so the *sync* saving
is ~0 on this build. The real saving is 2 of 3 `k_copy_src1_to_contiguous` gathers
(16384 rows × 2048 F32 = 128 MiB each ⇒ **2 × 128 MiB × 39 = 10 GiB/ubatch = 18 ms
at peak BW**), 2 of 3 `k_copy_dst_from_contiguous` scatters, and the glu round
trip. **Expected 3–8%.** Its value is mostly *structural*: it is the vehicle for
C4/§5 and it converts three loops over 256 experts into one, halving the
opportunity cost of every later per-expert optimisation.

*Risk.* HIGH, and the reason is on record: `serve-laguna.sh:57` disables this fuse
family because `DISABLE_MOE_DUAL_DOWN` guards "a known-broken path (NaN logprobs)
that is DEFAULT-ENABLED in source". Before touching `types_ok`, read
`notes/SHIP_20260731_dual_down_expert_loop_ppl.md` and
`notes/SHIP_20260731_dual_down_mul_mat_reorder_fix.md` and establish whether the
NaN family is (a) the reorder-SoA-into-oneDNN hazard the `4718-4745` guard exists
for, or (b) something in the fused reduce. If (a), this candidate is safe on the
DNN-off serving env and the guard already covers it. **If you cannot establish
which, do not run this candidate.** The MOE_DUAL_DOWN NaN family is a listed
do-not-try (§6) and this candidate deliberately re-opens it — it needs its own
written justification and a KLD receipt before anything else.

*Kill-switch.* `GGML_SYCL_LX_MIXED_DOWN_ELOOP=1` to opt **in** (leave
`DISABLE_MOE_DUAL_DOWN=1` semantics untouched, so serving stays on the current
path unless the new knob is set explicitly).

*Protocol.* golden-smoke → KLD (**this one must be spotless: mean_kld ≈ −0.0,
same_top 100%**) → real-text A/B → bench-serial → and a long-generation NaN watch
(the failure mode is NaN logprobs, which a 128-token `--temp 0` leg may not
surface; run ≥1024 tokens and grep for `nan`/`-inf`).

---

### C4 — Guarded fall-through: let reordered Q6_K expert slices use the fp16/oneMKL path
**Rank 4. Small, surgical, removes a state-dependent performance cliff.**

*Mechanism.* The block at `4718-4745` force-routes **any** column count to 8-col
chunked reorder-MMVQ whenever `src0` carries the reorder flag. Its justification is
that reordered SoA weights must not reach oneDNN/MMQ multi-row GEMM. But the
fp16/oneMKL path at `2796-2807` calls `ggml_get_to_fp16_sycl`, which **already**
returns `dequantize_row_q6_K_sycl_reorder` for reordered Q6_K (`convert.cpp:689`),
and `reorder_qw_q6_k_moe` (`4302`) keeps each expert self-contained within its
`nb02` stride so slice arithmetic holds. Narrow the guard to
`src1->ne[1] <= MMVQ_MAX_BATCH_SIZE` (decode / spec-verify), letting multi-column
prefill take fp16+oneMKL as it does pre-reorder.

*Expected win, arithmetic.* By the existing receipts, **~0 on today's serving
path**: `llama-server` request #1 (pre-reorder, 310.8 t/s) vs request #2 (warm,
post-first-decode, therefore post-reorder, 311.5 t/s) differ by 0.2%. So this
candidate's *throughput* value is a rounding error today. Its value is
**determinism**: it removes a documented 8-col-chunking cliff whose activation
depends on whether a decode has happened, which is precisely the class of
instrument-dependence that produced the confound in §3.1. Take it as a hygiene fix
bundled with C1/C2, not as a standalone win. **Expected 0–3%.**

*Risk.* MEDIUM — it re-permits a path the guard was written to forbid; the
`-nan` history is real. Guard it, measure PPL, and keep the old behaviour one env
flip away.

*Kill-switch.* `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` (default OFF).

*Protocol.* Must be A/B'd in **both** reorder states: (i) llama-cli with the prompt
first (pre-reorder), (ii) a second prefill after a generation (post-reorder), plus
`GGML_SYCL_ENABLE_OPT=0` as the reorder-free control. KLD mandatory.

---

### C5 — Out-of-order compute queue for the expert loop
**Rank 5. Cheap to try, plausible mechanism, but the evidence base is thin and it
is a correctness minefield.**

*Mechanism.* `ggml-sycl.cpp:1042` pushes
`dpct::get_current_device().default_queue()`, which is the **in-order** queue
(`dpct/helper.hpp:723,786-788`); the out-of-order queue exists and is dead code
(FRONTIER_20260806_inorder_queue_serialization). Under in-order semantics the
expert loop's `conv → conv → gemm → conv → conv → conv → gemm → …` chain cannot
overlap even though consecutive experts are fully independent (disjoint weight
slices, disjoint row ranges of `src1_contiguous`/`dst_contiguous`).

*Expected win, arithmetic.* If the ~120K launches/ubatch pay a 4–6 µs
serialization gap each, the recoverable budget is **0.48–0.72 s/ubatch = 8–12%**
(prefill ≈ 335–350 t/s). Overlap of the 2 MiB conversions with the GEMMs could add
more, but FINDING_20260810_pp512_hostbound §5 already found the conversions
"device-overlapped, pipelined", suggesting some of this is banked.

*Risk.* HIGH on correctness: dpct submits with raw USM pointers and **no implicit
dependency tracking**, so the ordering guarantees the current code relies on
(gather → GEMMs → scatter; `dev_row_mapping` H2D before the gather kernel;
`counts_ready_ev` semantics at `7355`) all evaporate. A partial version — keep the
global in-order queue, but submit only the per-expert GEMM triples on a secondary
out-of-order queue joined by an explicit barrier before the scatter — is the sane
form. Also note the tree already contains a hard-won comment at `7351-7356` that
"the async D2H counts event alone is not reliable on this L0 stack" — the driver
has form here.

*Kill-switch.* `GGML_SYCL_LX_OOO_EXPERT_QUEUE=1` (default OFF).

*Protocol.* golden-smoke is the first tripwire (nondeterministic ordering shows up
as golden mismatch), then run golden-smoke **10x** — an ordering bug that appears
once in ten runs is still a ship-blocker. Then KLD, then real text.

---

## 5. gate-up-concat: the correct re-measurement

The brief asks for a real-text re-measure because the rejection is documented
unsafe. It *is* unsafe — but not for the reason recorded. **Re-running
`GGML_SYCL_LX_GATE_UP_CONCAT=1` on real text on this model would produce another
pure control** (§0(a): `types_ok` at `6222-6234` can never pass with IQ4_NL down;
and `serve-laguna.sh:57` disables the fuse anyway). Do not spend a GPU slot on it.

The correct actions, in order:

1. **Prove the null first, cheaply.** One real-text run with
   `GGML_SYCL_LX_GATE_UP_CONCAT=1` **and** `GGML_SYCL_DISABLE_MOE_DUAL_DOWN=0`,
   watching stderr for the one-shot line
   `[lx-control-moe-dual] fuse hit (dual+down multi-token)` (`6238-6244`).
   **Predicted: the line never prints.** That is the receipt that retires the
   2026-08-10 verdict as "measured a control", and it costs one run.
2. **Re-implement the concat where the code actually runs** — the generic
   per-expert loop at `6924-6984`. Two forms:
   - **(2a) load-time static concat** — the FRONTIER_20260802 design. Pre-stack
     `ffn_gate_exps` and `ffn_up_exps` into one `[2*n_ff, n_embd, n_expert]`
     tensor at model load, emit a single MUL_MAT_ID, split in the SwiGLU kernel.
     **Zero runtime copy cost.** Halves gate/up submissions: per ubatch
     `39 × E` calls deleted, each carrying 1 oneMKL submission + 4 launches +
     2.00 MiB fp16 write. At E=256 that is **9,984 calls and 20 GiB/ubatch of
     round-trip traffic removed** ⇒ by FRONTIER's own tile arithmetic
     (2 × 17.61 µs → 18.89 µs, 46% on the gate/up family, which is 64% of matmul
     execs) a **~30% ceiling on the matmul-exec term**. Bit-exact by construction:
     `[W_gate; W_up]ᵀ x = [gate(x); up(x)]`.
   - **(2b) runtime packed copy** — what `7398-7443` actually implements today:
     two `stream->memcpy` of a full expert weight slice **per expert per
     dispatch** before the GEMM. That adds `2 × 840 KiB` read+write per expert per
     dispatch = **39 × 256 × 3.3 MiB ≈ 32 GiB/ubatch of new traffic** to save one
     submission. **This form is arithmetically doomed** and should not be
     resurrected — if the fuse had ever fired, it would have lost for this reason.
   ⇒ **Implement (2a). Never (2b).** Note this makes gate-up-concat a *model
   loading* change (`llama-model.cpp` / graph build + a SYCL split), not a
   SYCL-only change, so it is a larger unit than C1's kernel; sequence it after C1
   unless step 0.2 shows E is small (in which case per-call deletion is the whole
   game and (2a) jumps to rank 2).
3. **Measurement protocol** identical to §4's gate sequence, with the added
   requirement that the A/B legs are **real text at ub 2048 and ub 512** and that
   the fuse-hit stderr line is captured in the receipt directory. A candidate that
   cannot show its own fuse-hit line is not a candidate — that is the whole lesson
   of the 2026-08-10 receipt.

---

## 6. Do NOT try — already receipted dead

| dead end | receipt | why it is dead |
|---|---|---|
| `GGML_SYCL_LX_GEMM_BATCH=1` (strided `dpct::gemm_batch`) | FINDING_20260810_pp512_hostbound §4; FINDING_20260810_lxbin corrected-measurements | **Hangs the B70** (busy-spin, rc=124 at 240 s, host in S state). Confirmed still hanging *after* the `[lx-gb-fix-20260810]` OOB fix at `6796-6923`. Grouped `oneapi::mkl::blas::row_major::gemm_batch` aborts with `UR_RESULT_ERROR_OUT_OF_RESOURCES` (L0 error 40) on first dispatch. The only working GEMM surface on this device is per-call single `dpct::gemm`. |
| `GGML_SYCL_LX_MMVQ_PREFILL=1` | FINDING_20260810_lxbin | **−17%** (974.7 vs 1172.6 pp512). Multi-col q8_1 MMVQ in 8-col chunks loses to oneMKL fp16 per-expert for ncols 2–8. Mechanically: no weight reuse across chunks. |
| `GGML_SYCL_ENABLE_MMQ=1` | FINDING_20260809 §"MMQ is broken" | pp512 >120 s no sample; pp64 timed out at 75 s. Compiled MMQ kernels unusable on this path as shipped. Also IQ4_NL is excluded from `ggml_sycl_supports_mmq` anyway. |
| any-batch `mul_mat`+`add` fuse (`GGML_SYCL_ENABLE_MUL_MAT_ADD_ANY_BATCH=1`) | `SHIP_20260731_mmadd_decode_only.md`, code comment `4922-4924` | PPL explosion. Decode-only (`ne11==1`) is the quality-safe boundary. |
| `MOE_DUAL_DOWN` family default-on | `serve-laguna.sh:57`, `SHIP_20260731_dual_down_expert_loop_ppl.md` | NaN logprobs. Default-ENABLED in source, disabled in the serving env on purpose. C3 re-opens this deliberately and owes a written justification + spotless KLD before it runs. |
| full-tensor fp16 pre-conversion / fp16 weight shadow | FINDING_20260810_pp512_hostbound §5 | **−24%**. Conversions are device-overlapped; hoisting them whole serializes the device on 537 MB/family/layer kernels. Full shadow is also VRAM-impossible (FRONTIER_20260801_moe_weight_reorder_fullbank). |
| MMVQ ncols cap 8→32 as a wall lever | FINDING_20260809 "Rejected-with-new-evidence" | Only touches the 66.6 ms tiny term (15% of pp512); the large-slice wall needs a real batched kernel. |
| SYCL graph capture for prefill | FRONTIER_20260731_onednn_cache_hit_tax; `ABSOLUTE_LIMIT.md:33` | pp −2.6%; graph is env-killed. The decode-only case was never measured and remains open — but it is a *decode* lead, not a prefill one. |
| attention / fattn work as the prefill lever | FINDING_20260811_fattn_fullchain_backport_no_win | Full chain `7e1e28cae..dd1ea5243`, KLD-clean, real-text prefill **308.8 → 307.3** (nil) and depth-decode **92.4 → 81.4 (−12%)** DNN-off, −41% DNN-on. Attention is not the wall. |
| ubatch / context / warmup tuning | FINDING_20260810_serving_prefill_routing_skew table | ub 512→2048 = +2.8%; c 131072→32768 = −0.7%; warm request #2 = +0.2%; server-vs-cli = +0.6%. All dead. |
| `lx/mmid-device-batched-sgemm` salvage | §0(c), `git log`/`reflog` in `lx-dev-mmid-batch` | Branch contains only the FA backport; no batched-sgemm commit ever existed. |

---

## 7. Sequencing

1. **Step 0** (§3) — env-only, shipped binary, ~8 runs, no build. Deliverable: the
   attribution table + the E/N histogram + the de-confound verdict.
2. **C2** (conversion hoist) as the first build — smallest diff that touches the
   suspected term, and its result calibrates C1's scope.
3. **C1** (expert-blocked quantized GEMM) — the multiples candidate. Tile shape
   chosen from step 0.2.
4. **§5.2a** (load-time gate/up concat) — independent of C1's kernel, composes
   with it (a concatenated bank halves C1's dispatch count too).
5. **C4** hygiene, bundled with whichever build is in flight.
6. **C3** / **C5** only with their own written risk justification.

Non-negotiables carried from `PLAN_20260807_next_candidates.md` and AGENTS.md:
one variable per measurement; explicit `LX_BIN` exported **before** `source env.sh`;
never re-capture `correctness/golden.json`; never `LX_ALLOW_UNGATED`; save every
patch under `results/<stamp>/` with the `libggml-sycl.so` sha; **never buy prefill
with decode** — tg128 d0 ≥ 152.5 and real-text decode ≥ 92.4 are floors, not
preferences.

---

## Step 0 results (measured)

Run 2026-08-11. Receipts: `results/expert-loop-step0-20260811T191653Z/`
(`summary.txt`, per-leg `NN-*.err|.out`, `hist-*.txt`, `analyze-hist.py`,
`driver-*.sh`, `run-leg.sh`, `synthetic-23k.txt`, `gensynth.py`, `dumpvocab.py`).
Binary: the shipped champion `/home/frosty40/turbo/worktrees/lx-champion-tier12/build/bin`,
`libggml-sycl.so.0.17.0` sha256 `ae6407a41512…`. Env: the `serve-laguna.sh:48-56`
ship block, unchanged. Instrument: `llama-cli -f <text> -c 32768 -b 4096 -ub 2048
-n 128 --temp 0 -no-cnv --jinja --chat-template-file …`, one GPU process at a
time under `scripts/lib-gpu-lock.sh`. No source modified, nothing built,
nothing committed.

**Harness note (blocking, applies to every future step-0-class run).** This
fork's `llama-cli` returns to its `> ` turn loop after generation and spins
forever on stdin EOF (`tools/cli/cli-context.cpp:478-513`: empty buffer →
`continue`). `</dev/null` does **not** end it, and the `timeout` SIGTERM/SIGKILL
suppresses every `atexit()` diag printer — i.e. `LX_QUANT_NAME`, `LX_DIAG_COUNTS`,
`LX_IDS_ONCE` are unobtainable that way. Feeding `/exit` on stdin
(`< <(printf '/exit\n')`) makes `cli_context::run()` break and return 0, a normal
exit, and the printers fire. Every leg below is `rc=0` except 0.5/`-fa off`.

**Geometry (measured, not assumed).** `correctness/wikitext-23k.txt` is
**25,338 tokens** = 12 × ub2048 + 1 × 762 → **13 ubatches**. The model has
**40 layers, 38 of them MoE** (`ffn_moe_*-1 … -38`).

### 0.1 De-confound: instrument vs prompt content — **the 6x is NOT routing skew**

`results/.../synthetic-23k.txt` was built by sampling 23,200 uniform-random ids
from the model's own 100,352-entry vocabulary (`dumpvocab.py` reads
`tokenizer.ggml.tokens` straight out of the GGUF; `gensynth.py` byte-level-BPE
decodes them and concatenates) — the llama-bench `std::rand()%n_vocab` analogue,
fed through the *same* llama-cli instrument.

| leg | prompt | prefill t/s | decode t/s |
|---|---|---|---|
| `01-wikitext-base` | real text | **306.6** | 80.6 |
| `03-wikitext-hist` | real text (+`DIAG_NAME_QUANT=1`) | 307.5 | 81.3 |
| `02-synthetic-base` | random-vocab salad | **308.4** | 82.2 |
| `04-synthetic-hist` | random-vocab salad (+diag) | 308.4 | 82.0 |
| `11-wikitext-nowarmup` | real text, `--no-warmup` | 306.9 | 81.7 |

**Synthetic-via-llama-cli = 308.4 t/s, not ~1900.** Per §3.0.1's own decision
rule this is the "the 6x is the instrument, not the routing" branch: the entire
routing-skew premise of FINDING_20260810_serving_prefill_routing_skew, and with
it §2's M1/M2 axis, is falsified. `DIAG_NAME_QUANT=1` costs +0.3% — the diag is
free, as claimed. (`--no-warmup` is inert in this fork: the `tokens=2` warmup
fuse markers are still present in `11-*.err`.)

### 0.2 The histogram pair — **the shipped path never touches `ggml_sycl_op_mul_mat_sycl`**

`GGML_SYCL_DIAG_NAME_QUANT=1` on the ship env, **both prompts**:
`LX_QUANT_NAMES` **is never printed** — the map is empty, i.e. **zero
quantized-src0 calls reach `ggml_sycl_op_mul_mat_sycl` in the entire run**
(the printer is registered lazily inside that `if`, `ggml-sycl.cpp:2709-2740`).
The size-probe counters from 0.4 bound it exactly: over all 25,338 prompt tokens
the function is entered **608 times total** (114 with `ncols≤32`, 494 with
`ncols>32`; 494 = 38 MoE layers × 13 ubatches = the **F32 router GEMM**, one per
MoE layer per ubatch — and those are F32, not quantized).

⇒ **§1's path map (rows 10-12) and §2's whole cost model describe code that does
not execute on the serving path.** There is no fp16 shadow, no `dpct::gemm`
submission, no 2 MiB-per-expert conversion, no 119,808 launches/ubatch — none of
it, because none of those calls happen.

Re-run with `GGML_SYCL_ENABLE_OPT=0` (leg `14`) and the same hook fills up:

```
LX_QUANT_NAMES total=262751 distinct=38725
moe_expert_gemm_calls=258630   dense_calls=4121
ffn_moe_gate 86210 calls   ffn_moe_up 86210   ffn_moe_down 86210
```

So the expert loop *does* run — it just dispatches somewhere else by default.

**Corrected weight types (§0(a) is wrong for this GGUF).** The histogram carries
`ty=`: `ffn_moe_gate`/`ffn_moe_up` are **q4_K in all 38 MoE layers**;
`ffn_moe_down` is **q4_K in 22 layers, q6_K in 16**. There is **no IQ4_NL in the
expert banks at all**. All three families are therefore in
`ggml_sycl_supports_reorder_mmvq` (`3981-3993`), and `types_ok` at `6222-6234`
is *not* structurally unreachable (it fails only where down≠gate type).

**E and the N-histogram (from the OPT=0 leg — the only state in which these
calls are visible).** Per layer per projection per ubatch:
`E = 185.2 mean (min 128.7, max 246.3)` distinct non-empty experts — and that is
a *lower* bound, because experts with `N ≤ 8` take `mul_mat_vec_q` and never
enter this hook (`min_N=9` in the table below is a selection effect, not a
property of the routing). True `E` is close to 256.

```
N-histogram over 258,630 MoE expert GEMM calls (N = rows routed to that expert)
   N range     calls  %calls         rows   %rows  cum%calls
   9-16        41589  16.08%       511233   2.24%     16.08%
  17-32        58209  22.51%      1390623   6.09%     38.59%
  33-64        64251  24.84%      2994096  13.12%     63.43%
  65-128       53232  20.58%      4811022  21.07%     84.01%
 129-256       26136  10.11%      4557759  19.97%     94.12%
 257-512        9579   3.70%      3368451  14.76%     97.82%
 513-1024       3864   1.49%      2765868  12.12%     99.32%
1025-2048       1770   0.68%      2429112  10.64%    100.00%
 min_N=9  max_N=2014  mean_N=88.3  median_N(by call)=44  median_N(by row)=162
```

**C1 tile sizing, direct read:** 63% of calls are `N ≤ 64` but they carry only
21% of the rows; 84% of calls are `N ≤ 128`; the top 2.2% of calls (`N > 512`)
carry 23% of the rows. A single fixed tile is wrong — the kernel wants a
**token-tile of 64 with a grid stride over N** (covers the mass of calls at full
occupancy, and the fat `N ∈ [512, 2014]` tail amortises the weight load over
8-32 tiles). Do **not** size for `N ≈ 8`.

### 0.3 `GGML_SYCL_LX_DIAG_EXPERT_CAP` sweep on real text (ship env)

| cap | prefill t/s | vs cap 0 |
|---|---|---|
| 0 (baseline) | 306.6 / 306.9 | — |
| 8 | **662.5** | 2.16x |
| 16 | 642.0 | 2.09x |
| 32 | 601.5 | 1.96x |
| 64 | 523.0 | 1.70x |
| 128 | 422.9 | 1.38x |
| 256 | 306.8 | 1.00x |

Monotone, no peak — the *opposite* shape to the pp512 curve quoted in §3.0.3
(0→1168, 64→3947, 128→2929, 256→1170), and cap 256 reproduces the baseline
exactly, confirming E ≤ 256. Wall attribution: prefill wall is
25,338/306.9 = **82.6 s**; at cap 8 it is 25,338/662.5 = **38.2 s**. The
per-expert GEMM loop is therefore **≈46 s = 56% of the real-text prefill wall**,
and the **44% residual is not the ids sync** (§0(b)'s suspect) — it is the dense
projections, which take the same degraded path (see 0.6).

### 0.4 GEMM-exec split — the oneMKL term is ~0 on the shipped path

| leg | prefill t/s | Δ | counter |
|---|---|---|---|
| `06 DIAG_SKIP_QUANT=1` | 306.8 | **0.0%** | no `LX_DIAG_QUANT_SKIP` line ⇒ **count = 0** |
| `07 DIAG_SKIP_TINY_N=32` | 306.9 | 0.0% | `tiny=114 large=494 skip=114` |
| `08 DIAG_SKIP_LARGE_N=32` | 311.9 | **+1.6%** | `tiny=114 large=494 skip=494` |

Deleting *every* GEMM that `ggml_sycl_op_mul_mat_sycl` executes buys **1.6%** —
and those 494 calls are the F32 router GEMM, the only thing left on that path.
The quantized-GEMM-exec + fp16-conversion term that C1 and C2 are designed to
attack is **0.0% of the shipped real-text prefill wall**.

### 0.5 Attention control

| leg | result |
|---|---|
| `09 -fa off` | **no completion inside the 600 s cap** (`rc=137`); baseline needs 100 s wall. Non-FA attention at 25K is ≥6x worse — no usable number, and no evidence attention is the lever. |
| `10 GGML_SYCL_DISABLE_DNN=0` | first attempt aborted at model load (`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`, `ggml-sycl.cpp:614`) because the SIGKILLed `-fa off` leg had not released VRAM — harness artefact, re-run as `10b` |
| `10b GGML_SYCL_DISABLE_DNN=0` | **not obtained** — 3 further attempts (90 s apart) all aborted at load with the same OOM; the card never recovered inside this session |

0.5 is the one leg of step 0 that did **not** produce a number. It is also the
least load-bearing: FINDING_20260811_fattn_fullchain_backport_no_win already
says attention is not the wall, and 0.4 now shows oneDNN could at most touch the
608 calls that reach `ggml_sycl_op_mul_mat_sycl` (1.6% of wall) — DNN-on cannot
be the lever while the reorder guard is in force. **Re-run `10b` on a clean card
before quoting anything about DNN-on for real text.**

**Second harness hazard (record it).** SIGKILLing a llama process on this B70
leaks its VRAM: the xe driver leaves a `kworker/u128 gt-ordered-wq` in `D` for
minutes and every subsequent load aborts at
`ggml_backend_sycl_buffer_set_tensor` (`ggml-sycl.cpp:614`) with
`UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` (L0 error 39). Legs `10` and the first
`bench-*` attempts died this way. Any harness that can SIGKILL a leg needs a
settle-and-retry before the next one — `driver-6.sh` in the receipt dir is the
pattern.

### 0.6 (new, decisive) The mechanism: the reorder flag, `ggml-sycl.cpp:4714-4744`

§3.0.1's branch instruction ("what does llama-bench do differently") has a
one-env answer.

| leg | env | prefill t/s | decode t/s |
|---|---|---|---|
| `01/03` wikitext | ship | 306.6 / 307.5 | 80.6 / 81.3 |
| `12` wikitext | ship + **`GGML_SYCL_ENABLE_OPT=0`** | **1610.8** | 69.5 |
| `14` wikitext | ship + `ENABLE_OPT=0` + hist | 1581.0 | 69.1 |
| `02` synthetic | ship | 308.4 | 82.2 |
| `13` synthetic | ship + **`GGML_SYCL_ENABLE_OPT=0`** | **1737.5** | 69.9 |

**`GGML_SYCL_ENABLE_OPT=0` is worth 5.25x on real-text prefill** (306.9 →
1610.8) — the entire "6x headroom" this dossier was written to chase, on the
shipped binary, with no build. It lands within 15% of llama-bench's synthetic
1895.9 t/s, which is the remaining, ordinary gap (llama-bench is a pure prefill
loop with no server, no template, no sampler). The llama-bench anchor legs
(`bench-opt1`/`bench-opt0`, `driver-4.sh`/`driver-6.sh`) could **not** be
collected — see the VRAM note below — and are the one open item from this
session.

Mechanism, and it is fully consistent with every counter above:

1. The warmup decode fires the decode-only fuses
   `ggml_sycl_mul_mat_id_dual_swiglu_fused` (`4837`, `ne12 != 1 → return false`
   at `4867`) and `ggml_sycl_mul_mat_id_mmvq_fused` (`6475`), each of which calls
   **`opt_for_reorder_id`** on the expert banks (`4882-4883`, `6475`) and
   permanently sets `extra->optimized_feature.reorder`. The one-shot marker
   `[lx-control-moe-dual] fuse hit (gate+up+swiglu)` is present in every ship-env
   `.err` and **absent** in `12-wikitext-opt0.err` — that is the flag flipping.
2. From then on, the blanket guard at **`ggml-sycl.cpp:4714-4744`** ("Force
   chunked reorder-MMVQ for any column count when src0 is already reordered")
   catches *every* quantized multi-column matmul — its `src1->ne[2]==1 &&
   ne[3]==1` precondition is satisfied verbatim by the expert loop's `src1_row`
   (`6710-6715`) — and shreds each `N`-row expert GEMM into `ceil(N/8)`
   8-column dp4a MMVQ launches with **no weight reuse between chunks**, instead
   of one fp16/XMX oneMKL GEMM. At `mean_N = 88` that is ~11x more expert-slice
   reads and no systolic path.
3. It is not only the experts: `[lx-control-qkv] fuse hit q=reordered[6144]
   … ncols=2048` in the ship-env logs shows the **dense attention projections at
   prefill** on the same reordered-MMVQ path. Under `ENABLE_OPT=0` those same
   nodes reappear as `Qcur/Kcur/Vcur ty=q4_K ncols=2048` oneMKL calls in the
   histogram. That is the missing 46% the cap sweep could not reach.

This is exactly the "state-dependent path flip between instruments" flagged in
§1 and then **wrongly dismissed** in C4. C4's falsification used
`llama-server` request #1 (310.8) vs request #2 (311.5) — but request #1 is
*already* post-warmup, i.e. both legs were on the reordered side of the cliff.
There is no gradient here to measure; it is a one-way latch thrown before the
first user token.

Cost side, and it is real: decode pays for the reorder — 81.5 → 69.5 t/s
(−14.7%) on the same real-text leg. `ENABLE_OPT=0` is therefore **not** a
shippable config (AGENTS.md: never buy prefill with decode). The fix is C4's
shape: narrow the `4714-4744` guard to `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE`
so decode keeps reorder-MMVQ and prefill takes fp16/oneMKL.


### Verdict

**The data supports none of C1, C2, §5.2a, or "unattributed idle". It supports
C4 — and C4 is not a 0-3% hygiene item, it is the whole 5.25x.**

- **C1 (expert-blocked quantized GEMM) — premise void as written.** Its target
  (`39 × E × 3` oneMKL submissions, `× 4` launches, 2 MiB fp16 weight
  round-trips, 3 pool allocs per call) is measured at **zero occurrences** on the
  shipped path: `DIAG_SKIP_QUANT=1` costs 0.0% and the name histogram is empty.
  C1 remains the right kernel to own the *post-C4* path (where 258,630 real
  oneMKL expert GEMMs do exist), but it cannot deliver the 6x, because the 6x is
  not there. Tile from 0.2: token-tile 64, grid-stride over N.
- **C2 (conversion hoist) — void, same reason.** Zero `to_fp16_sycl` expert
  conversions execute on the shipped path. Its predicted 2-8% is 2-8% of a term
  that is 0%.
- **§5.2a (load-time gate/up concat) — arithmetic unchanged but currently
  inert**, for the same reason; and §0(a)'s justification is separately wrong:
  the expert banks are **q4_K/q6_K, not IQ4_NL**, so `types_ok` is not
  structurally unreachable.
- **"Unattributed idle" — retired.** §2's "85-90% of the wall is unattributed"
  was an artefact of measuring a path that does not run. The wall is now
  attributed: **~56% per-expert loop + ~44% dense projections, both on the
  8-column reorder-MMVQ path, ~1.6% router GEMM, ~0% oneMKL expert GEMM.**
- **C4 — promoted to rank 1.** Not "expected 0-3%": **+425% prefill measured**
  (306.9 → 1610.8 t/s) by the crude env proxy, with the decode cost
  (−14.7%) that the narrowed guard exists to avoid. Sequence it before C1.

Also retired by these runs: §0(b)'s "no measurement has localised real-text
prefill device cost" (0.3 + 0.4 localise it), §2's M1/M2 dichotomy (0.1 kills
both), and §4's ranking (C4 → C1 → C2, not C1 → C2 → C4).

**Next step is a build, not another env leg:** narrow `ggml-sycl.cpp:4714-4744`
to `src1->ne[1] <= MMVQ_MAX_BATCH_SIZE` behind
`GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` (default OFF), then golden-smoke → KLD →
real-text A/B (must beat 306.9 prefill **and** hold 81.5 decode) →
`bench-serial.sh` (tg128 d0 ≥ 152.5). The `-nan` history that motivated the
blanket guard (`SHIP_20260731_dual_down_expert_loop_ppl.md`) is about
reorder-SoA weights reaching oneDNN/MMQ multi-row GEMM; the serving env is
`DISABLE_DNN=1` and MMQ is off, and `ggml_get_to_fp16_sycl` already returns
`dequantize_row_q6_K_sycl_reorder` for reordered banks (`convert.cpp:689`) —
so the fp16/oneMKL fall-through is the one case the guard did not need to cover.
KLD is the arbiter.
