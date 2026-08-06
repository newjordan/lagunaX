# Frontier — Score-harness inter-token barrier (2026-08-01)

## Direction (PIVOT — not dirs 1–24)

**Invert "optimize kernels inside a decode step."** The scored tg128 wall clock is
defined by llama-bench's `test_gen` loop, which **fully drains the device after every
token** and feeds a **random next token that never depends on logits**. Cross-token
GPU overlap is structurally illegal in the measurement framework. Distinct from:
- dir 24 (in-order queue *inside* a step)
- dir 9 (oneDNN host create:cache_hit serialization)
- host↔device leaf traffic (set/get barriers as a substrate, not the harness contract)
- dir 23 / supports_op (host query cadence)
- lm_head ROI (kernel time only)

## Evidence opened this iteration

- `/home/frosty40/llama.cpp/tools/llama-bench/llama-bench.cpp` (`test_prompt` / `test_gen`)
- `/home/frosty40/llama.cpp/src/llama-context.cpp` (`synchronize`, `llama_synchronize`)
- `/home/frosty40/llama.cpp/ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_backend_sycl_synchronize`, `set_tensor`)
- `/home/frosty40/llama.cpp/src/llama-batch.cpp` (`llama_batch_get_one`, logits-null default)
- `results/ktrace-tip-20260730/decode-ggml/trace.log` (inter-get sync cadence)
- `results/LATEST_SCORE.json` (138.02 tok/s → 7.245 ms/tok)

## Findings

1. **tg syncs every token; pp syncs once.** `test_gen` does `llama_decode` →
   `llama_synchronize` → random token, N times. `test_prompt` does N `llama_decode`
   batches with a **single** `llama_synchronize` at the end.
2. **Next token ignores logits.** `token = std::rand() % n_vocab` — no
   `llama_get_logits` / argmax. Yet `llama_batch_get_one` leaves `logits=nullptr`,
   which defaults to **last-token output true**, so lm_head + 401 KB D2H still run.
3. **Steady-state gap is 11 syncs + 9 set_tensors between logits pulls.**
   Pattern after each `get_tensor_async(result_output)`: 3× `synchronize`, then
   embd + 6 leaves + 2× kq_mask set_tensors each preceded by sync (set_tensor's
   `queues_wait_and_throw` + `.wait()`). Zero `device_supports_op` in that gap.
4. **SYCL synchronize is a full default-queue wait:** `stream->wait()` only — no
   partial event, no "wait for logits only."

## Hypotheses

1. A diagnostic `test_gen` that skips `llama_synchronize` between tokens (or only
   syncs the logits event) would measure true multi-token submit fill — **not** a
   valid score, but a bound on harness-forced idle.
2. A diagnostic graph with `n_outputs=0` / no lm_head on the random-token arm
   isolates MoE+attn floor vs the ~0.3 ms lm_head + D2H tax under the same harness.
3. Real sampling needs logits→token, so ship code cannot drop the post-lm_head
   drain; the win is making that drain **event-scoped** (logits only) and starting
   next-step H2D leaf uploads on a BCS/copy path while lm_head still runs.

## Not claimed

- Wall-time fraction of the 11 syncs vs 7.245 ms/tok (needs host timers).
- That removing harness sync is a legal score change (it is not).
