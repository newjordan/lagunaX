# FINDING 2026-08-10 — real-text MoE routing makes serving prefill ~6x slower than llama-bench claims; ub is irrelevant on real text

## Receipts (champion build ae6407a4, identical env/geometry unless noted)

| instrument | prompt | geometry | prefill t/s |
|---|---|---|---|
| llama-bench -p 16384 | synthetic uniform-random tokens (`std::rand()%n_vocab`, llama-bench.cpp:2125) | ub2048 b4096 | **1895.9** |
| llama-bench -p 16384 | synthetic | ub1024 | 1467.3 |
| llama-bench -p 16384 | synthetic | ub4096 | 2023.1 |
| llama-server 23K request | wikitext | ub2048 b4096 c131072 | **310.8** |
| llama-server 23K request | wikitext | ub2048 b4096 c32768 | 308.5 (allocated-ctx scaling: DEAD) |
| llama-server 22K request #2 (warm) | wikitext | ub2048 | 311.5 (warmup: DEAD) |
| llama-cli -f 23K file | wikitext | ub2048 b4096 c32768 | **308.8** (server overhead: DEAD) |
| llama-cli -f 23K file | wikitext | ub512 (defaults) | 300.5 |

Steady-state (flat rate from first to last progress line; no fallback warnings;
champion fuse lines active).

## Conclusion

The gap is **prompt content = MoE routing shape**. Uniform-random tokens route
~evenly (2048-tok ubatch x k=8 / 256 experts ≈ 64 rows/expert); real text
routes Zipf-skewed (many active experts with few rows), and the pp expert loop
pays per-expert fixed costs (host GEMM submission, the 17-21 us oneDNN-floor
class from FRONTIER notes) that dominate at low rows/expert. ub 512→2048
moves real-text prefill only 300→309 t/s: the loop is expert-count-bound, not
ubatch-bound.

## Implications

1. Every llama-bench prefill number in the campaign (incl. board pp512
   1170.98) is a uniform-routing best case. Decode receipts transfer to real
   usage (149.7 t/s short-chat vs 152.5 bench); prefill receipts do NOT
   (~310 t/s real long-context ingest).
2. **The real-serving prefill headroom is ~6x** — the biggest untouched
   optimization surface in the campaign. Attack class: per-expert overhead in
   the multi-token expert loop (submission coalescing, gate+up concat, expert
   batching across layers). NOTE: `lx-gate-up-concat` was REJECTED on a
   synthetic-prompt measurement (and during the env-contamination window) —
   that verdict is unsafe; re-measure ON REAL TEXT.
3. Any future pp candidate must be A/B'd with a real-text instrument
   (llama-cli -f / server request), not llama-bench alone.
4. mlxfast challenge: the ranked eval uses a hidden real prompt — expert-loop
   overhead attacks may overdeliver there relative to synthetic local
   estimates (their baseline pays the same skew), and uniform-only wins will
   underdeliver.
5. Operational: the GGUF's embedded chat template does not parse on this build
   ("Unknown statement: include") — `--jinja --chat-template-file
   poolside-Laguna-XS-2.1.jinja` is REQUIRED for llama-server/llama-cli, not
   cosmetic. serve-laguna.sh carries it.

## Serve config verdict (2026-08-10)

Keep `-ub 2048 -b 4096`: 4096 wins only the synthetic sweep (+6.7%); on real
text ub is a wash, and the 2026-08-05 depth A/B (old binary) plus VRAM cost
still favor 2048. Champion serving receipts: short-chat decode 149.7 t/s,
23K ingest 310 t/s prefill / 83.8 t/s decode at depth, full 131072 ctx.
