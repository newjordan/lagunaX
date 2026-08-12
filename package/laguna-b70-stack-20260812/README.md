<p align="center">
  <img src="laguna_b70_turbo.png" alt="Laguna 2.1 B70 Turbo" width="60%"/>
</p>

# Laguna XS 2.1 on Intel Arc Pro B70 — the long-context stack (2026-08-12)

llama.cpp SYCL kernel stack for **Laguna-XS-2.1 Q4_K_M** on **Arc Pro B70**,
tuned for **max-context workloads** (RL loops, long-document serving at the
full 131,072-token window).

## Headline numbers (23K-token real text, `-c 131072`, one B70)

| | before | after |
|---|---|---|
| Context ingest (prefill) | 307 t/s (74 s) | **1,764 t/s (14 s) — 5.75x** |
| Decode @ 23K depth | 81.5 t/s | **90 t/s (+10%)** |
| Decode @ 122K depth (full window) | 36.0 t/s | **40.8 t/s (+13%)** |
| Short-context decode (tg128) | 152.5 t/s | 152.5 t/s (unchanged) |
| 1536-token generation @ 131K | — | 0 nan/inf, 87.9 t/s sustained |

Quality: bit-parity with the previous champion when the knobs are off;
knobs-on output measures **closer to canonical fp16/oneMKL math than the
previous champion did** (KLD 0.045 vs 0.056 against a linear-weight
reference) with slightly better wikitext perplexity.

## What's in this package

- `laguna-b70-stack-bin-20260812.tar.gz` — the built binaries + libs
  (llama-server / llama-cli / llama-bench / llama-perplexity),
  branch `lx/reorder-multicol-mkl`, tag `lx-stack-1.4092-20260812`.
- `serve-laguna.sh` — the validated serving script (full env, receipts in
  the header). Model path expected:
  `Laguna-XS-2.1-Q4_K_M.gguf` (see MANIFEST for the source location).
- `CLOSEOUT_20260812_laguna_b70.md` — the complete engineering record:
  mechanisms, receipts, falsified approaches, open frontiers.
- `MANIFEST.txt` — shas, commits, env contract.
- `HF-README-update-draft.md` — drafted section for the public
  Frosty40/Laguna-XS-2.1-ArcB70-GGUF card (publish at your discretion).

## The three changes (each env-gated, individually kill-switchable)

1. `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` — a warmup decode permanently
   latches weight reordering, after which a blanket guard shredded every
   wide matmul into 8-column MMVQ chunks; narrowing the guard to decode
   widths recovers 5x prefill with decode untouched.
2. `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=16` — widens flash-attention
   decode split-K from 4 to 16, shortening each workgroup's serial KV walk;
   the win grows with context depth (+13% at 122K). 16 is depth-flat optimal.
3. `GGML_SYCL_LX_EXPERT_TILE_GEMM=1` — an XMX joint_matrix kernel fuses
   dequantization and GEMM for small-batch MoE expert slices (q4_K + q6_K),
   replacing ~100K per-call oneMKL launches per long prompt.

## Run

```bash
tar xzf laguna-b70-stack-bin-20260812.tar.gz
# edit REPO/MODEL paths in serve-laguna.sh if relocating, then:
bash serve-laguna.sh          # serves :8092, full 131072 ctx
```
