# Research — dual_down expert-loop PPL break (2026-07-31 push)

## Status: **isolated; not shipped**

Quality-safe tip still kills `MOE_DUAL_DOWN`. Expert-loop is the prefill win **and** the PPL killer.

## Formal (tip binary `build-mmadd-decode`)

| arm | score | tg128 | pp512 | golden | wikitext-2 PPL 2×512 |
|-----|------:|------:|------:|:------:|---------------------:|
| quality-safe ship (mm-add decode-only) | ~1.220 | ~138.5 | ~1173 | OK | **12.45** |
| dual_down ON, multitoken OFF (expert-loop default) | **1.231** | 138.6 | **1215** | OK | **-nan** (neg logprob std) |
| dual_down ON + `DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1` | ~1.218 | ~138.4 | ~1170 | OK | **12.48** (OK) |
| dual_down ON + host sort only | — | — | — | — | -nan |
| dual_down ON + kill packed reduce | — | — | — | — | -nan |

Artifacts: `results/push-wave-20260731T1622/`

## Conclusion

1. **Expert-loop** (`ggml_sycl_mul_mat_id_dual_down_multitoken_expert_loop`) is the quality break on prefill.
2. Prefill **+~40–45 pp512** (~score 1.231) is **invalid** for quality-safe claims until PPL fixed.
3. Decode dual_down alone (expert-loop killed) is golden/PPL safe but **no scored win**.
4. Device-sort vs host-sort and packed-reduce kill do **not** fix PPL.

## Deeper bisect (2026-07-31 evening)

Expert-loop still **-nan** when:

| kill | still nan? |
|------|:----------:|
| mm-add OFF | yes |
| residual fuses OFF (rms/softplus/rope/add_add) | yes |
| true-topk OFF | yes |
| router GEMV/sigmoid OFF | yes |
| TOPK_MOE fuse OFF | yes |
| MMID device-sort OFF | yes |
| packed_reduce OFF | yes |
| DNN OFF (native GEMM) | yes |

Live hit log (PPL chunk):
```
[lx-control-moe-dual] multi-token dual+down EXPERT-LOOP n_tokens=2 k=8 n_experts=256 packed_reduce=1
```

Also: `MOE_DUAL_MULTITOKEN=ON` alone (dual_down OFF) → **PPL -nan** + golden OK; formal score **worse** (~1.212). So multitoken dual GEMM path is also quality-broken on current tip; only **decode dual MMVQ** is safe.

Hypothesis (open): expert-loop `ggml_sycl_mul_mat` on Q4_K expert slices after reorder-MMVQ path corrupts layout; or packed inv/weight indexing for [1,k,T] weights.

## Fix track

1. Restore expert-loop from `patches/0028` + packed reduce into source (tip binary only today).
2. Bitexact vs compose dual_down (eloop off) on a single MoE layer dump.
3. Guard: refuse expert-loop if weight `extra->optimized_feature.reorder` set; force non-reorder GEMM.
4. Until fixed: keep dual_down + dual_multitoken **OFF** in `env.sh`.

## Env (still quality-safe default)

```bash
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
# research only (safe but no win):
# GGML_SYCL_DISABLE_MOE_DUAL_DOWN=0
# GGML_SYCL_DISABLE_MOE_DUAL_DOWN_EXPERT_LOOP=1
```
