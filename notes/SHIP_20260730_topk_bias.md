# Ship note — Laguna topk-moe bias fuse (2026-07-30)

## Status: **research / opt-in only** — not scored default

| Config | tg128 (ub4k) | golden | notes |
|--------|-------------:|:------:|-------|
| dual ON, topk bias **OFF** (default) | **~110** | **OK** | scored tip |
| dual ON, topk bias **ON** (bitonic) | **~97** | **FAIL** | fuse hits; slower + wrong tokens |
| earlier iterative top-k ON | ~115 | FAIL | faster but not golden-safe |

## What we built

1. **Correct Laguna graph pattern** (measured live):
   ```
   SIGMOID → RESHAPE → ADD(exp_probs_b) → ARGSORT → VIEW → GET_ROWS
   [→ weight norm → scale]
   ```
2. **CUDA-style `has_bias` semantics** in the fused kernel: select on `wt+bias`, emit unbiased `wt`.
3. **Bitonic DESC sort** on selection scores (same network family as SYCL `k_argsort`) for the bias path.
4. Env:
   - `GGML_SYCL_ENABLE_TOPK_MOE_BIAS=1` — opt-in
   - `GGML_SYCL_DISABLE_TOPK_MOE=1` — hard kill

Patches:
- `patches/0003-control-topk-moe-bias-optin.patch`
- rolled into refreshed `patches/0001-control-q4k-moe-dual-swiglu.patch`

## Why not default / scored

- **Golden mismatch** even with bitonic (still diverges from the unfused chain).
- **Speed regression** when ON (~97 vs ~110 tg): full 256-key bitonic inside the fuse does not pay for itself vs the already-GPU argsort path.

## Scored tip remains

**Control dual SwiGLU only** — formal ~**+2%** vs pin, golden OK.

## Next experiments (not yet)

1. Fuse only **sigmoid+add+get_rows+norm**, leave **device argsort** as the existing kernel (true bitexact selection).
2. Partial top-k bitonic that matches `k_argsort` prefix without full 256 if possible.
3. Profile barrier cost of in-fuse bitonic vs standalone argsort.

## Artifacts

- Pattern discovery + hit: fuse log `[lx-control-topk-moe] laguna bias fuse HIT (bitonic)`
- A/B: ON ~97 tg / OFF ~110 tg (2026-07-30)
