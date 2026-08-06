# Frontier — MoE full-bank weight-reorder amortization (2026-08-01)

## Direction (new; not dirs 1–24)

**Quantized-weight layout reorder as a first-use full-expert-bank tax** — the
`opt_for_reorder` / `opt_for_reorder_id` → `reorder_qw_*_moe` path that
SoA-scatters every Q4_K/Q6_K weight once, allocating a **full-tensor** device
temp and memcpy'ing **all 256 experts** even though decode routes only k=8.
Distinct from:
- dir 7 (steady-state f16 dequant BW of oneDNN expert path)
- dir 16 (native mul_mat dispatch share)
- dir 21 (MoE fusion knobs that *require* reorder as a precondition)
- dir 22 (pool allocator under staging temps)
- dir 24 / open-lead-24 (in-order queue + the 16 wait sites — those are the
  *mechanism*; this direction is the *payload size and expert-count waste*)

## Evidence sources

- Code: `/home/frosty40/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp`
- Trace: `results/ktrace-tip-20260730/decode-ggml/trace.log` (buffer_init_tensor)
- Scored bench: `results/20260731T172351Z/llama-bench.log` (tg sample tightness)

## Findings

1. MoE expert weights are 3D `[ne0,ne1,n_expert=256]`. `reorder_qw()` branches on
   `ne[2] > 1` into `reorder_qw_{q4,q5,q6}_k_moe`, which sets
   `total_bytes = expert_bytes * n_expert` and runs `parallel_for(total_blocks)`
   over **every** expert. [ggml-sycl.cpp:4195–4207, 3843–3884]

2. Per MoE layer the three expert banks are ~144 MiB (gate q4_K) + 144 MiB (up
   q4_K) + 210 MiB (down q6_K) = **498 MiB**; ×39 layers ≈ **18.97 GiB** of
   first-use reorder payload (device memcpy volume). [trace init nb[3]/nb[2]]

3. Each MoE reorder allocates a **full-bank temp equal to the live weight**
   (`sycl_reorder_temp_buffer tmp(stream, total_bytes)`), so peak extra VRAM
   during down-exps reorder is ~210 MiB (plus any VMM 2 MiB rounding). Failure
   skips the reorder and leaves the flag unset. [3846–3851]

4. Fused gate/up and down paths **hard-require** `optimized_feature.reorder`:
   after `opt_for_reorder_id` they `return false` if the flag is still clear,
   so a failed/disabled reorder silently drops out of the dual-swiglu /
   weighted-down fused kernels. [4647–4653, 5019–5022]

5. Amortization is a sticky per-tensor flag
   (`extra->optimized_feature.reorder = true`); subsequent steps skip. With
   async mem-op unavailable the first call pays
   `copy_event.wait()` + `reorder_event.wait_and_throw()` per bank.
   [4250–4251, 4272–4273, 3857–3882]

6. Scored tg128 samples_ns are flat (max/min ≈ 1.0015) — llama-bench's
   warmup absorbs the cold reorder, so **steady-state tg128 does not price this
   tax**; it is a first-token / cold-start / peak-VRAM axis. [20260731T172351Z]

## Hypotheses

1. **Routed-expert-only reorder** (reorder the 8 active expert slices on first
   hit, lazy) would cut first-use device traffic ~32× for a given layer step and
   shrink the temp from 144–210 MiB to ~0.6–0.8 MiB — needs a per-expert dirty
   bit instead of a tensor-wide flag.
2. **Load-time eager reorder** of all banks (outside the timed decode loop)
   would move the 19 GiB / ~40 GiB R+W off the critical path without changing
   kernels; peak VRAM during load is the tradeoff.
3. If B70 ever gains `ext_oneapi_async_memory_alloc`, the existing
   `GGML_SYCL_USE_ASYNC_MEM_OP` path removes the 2 host waits per bank without
   changing payload size.
4. `GGML_SYCL_DISABLE_OPT=1` would disable reorder entirely and force fused MoE
   to fall back — a zero-recompile A/B for "is reorder net-win after warmup".
