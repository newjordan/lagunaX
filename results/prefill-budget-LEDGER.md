# pp512-attributed dispatch budget (first ever prefill-side capture)

Run: results/prefill-budget-pp512b-20260806T105903Z  (rc=0, GGML_SYCL_TIMER_ALL=1, pp512+tg4, r=1)

pp512 = 1126.61 t/s; tg4 = 135.17 t/s (timer live, quality-invisible again)

## Bucket budget (dispatch-gap wall, us / calls / us-per-call)
[layer-timer] bucket attn_o:      10117 us         280 calls     36.13 us/call
[layer-timer] bucket ffn_shexp:      73424 us         273 calls    268.95 us/call
[layer-timer] bucket ffn_out:     864729 us           7 calls  123532.71 us/call
[layer-timer] bucket other:    3031935 us       45696 calls     66.35 us/call

## Headline
- ffn_out (MoE down): 7 calls @ 123.5 ms/call = 864.7 ms (21.7% of span) — 18x the decode per-call (6.7 ms, finding 23): prefill's down-proj is ONE fused 512-token GEMM per layer, not per-token MMVQ.
- other (attn/norm/embd): 45,696 calls @ 66.35 us/call = 3.03 s (76%) — same 76% share as decode (finding 23), same CPU-side-span structure.
- attn_o 280 calls @ 36.1 us; ffn_shexp 273 calls @ 269 us.

## Gotchas
- per-layer arrays (all_layer_us/calls) are declared+printed but NEVER incremented — per-layer budget rows are dead code (only bucket timer is live).
- with-gpu-lock run of the same script produced 0 timer lines on an rc=0 run (env dropped); direct invocation works. Use direct invocation for probe runs.
