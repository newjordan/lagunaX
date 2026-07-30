# Research note — MoE dual SwiGLU multi-token (2026-07-30)

## Status: **research / opt-in only** — not scored tip

| Config | golden | note |
|--------|:------:|------|
| tip stack, dual multi-token **OFF** (default) | **OK** | formal tip ~tg118.8 / +7.94% (hybrid mode2) |
| dual multi-token **default ON** (ne12 1..32) | **FAIL** | greedy smoke mismatch |
| `GGML_SYCL_ENABLE_MOE_DUAL_MULTITOKEN=1` | research | same path as failed default; do not score |

## Goal

Apply MoE dual gate+up+SwiGLU to prefill (`ne12>1`) without D2H, aiming prefill lift on serial pin.

## What we built

1. Extended `ggml_sycl_mul_mat_id_dual_swiglu_fused` to accept `ne12` in **1..32**.
2. Multi-token path: **per-token** dual launches (same `mul_mat_vec_q_id_dual_swiglu_reorder` as decode), shared quantize of all token×slot rows, pointer offsets per token.
3. **Opt-in only** after default-ON golden FAIL.

Enable research:
```bash
export GGML_SYCL_ENABLE_MOE_DUAL_MULTITOKEN=1
```
Full dual kill (decode+research): `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

## Findings

- Per-token dual is still not bitexact on golden (same class of failure as multi-token mmid batch).
- Prefill correctness surface is hard; tip prefill stays ~pin (1130–1140).
- Decode dual path unchanged when multi-token disabled.

## Tip unchanged

Default ON stack: dual (decode) + hybrid mode2 + dense dual shexp + moe-down two-step.  
Multi-token dual **OFF**. Multi-token mmid **OFF**. Topk+bias full fuse **OFF**.

## Next levers

1. Synthetic graph diff dual multi-token vs stock for first failing token/expert.
2. Device multi-token mmid bitexact fix (same correctness class).
3. Other prefill wins that don't touch fused dual shapes.
