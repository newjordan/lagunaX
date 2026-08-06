# pp512-attributed dispatch budget (first prefill-side capture)

Run: results/prefill-budget-pp512b-20260806T105903Z (r=1, pp512+tg4)
Confirm: results/prefill-budget-pp512-fixed-20260806T110342Z (r=2, pp512+tg16, script fixed)

## Bucket budget (dispatch-gap wall, us / calls / us-per-call) — r=1 run
[layer-timer] bucket attn_o:      10117 us         280 calls     36.13 us/call
[layer-timer] bucket ffn_shexp:      73424 us         273 calls    268.95 us/call
[layer-timer] bucket ffn_out:     864729 us           7 calls  123532.71 us/call
[layer-timer] bucket other:    3031935 us       45696 calls     66.35 us/call

## Confirm run (r=2, both phases)
[layer-timer] bucket attn_o:      23041 us        1440 calls     16.00 us/call
[layer-timer] bucket ffn_shexp:     117056 us        1404 calls     83.37 us/call
[layer-timer] bucket ffn_out:     867221 us          36 calls  24089.47 us/call
[layer-timer] bucket other:    3732556 us       73029 calls     51.11 us/call

## Headline
- ffn_out (MoE down): r=1: 7 calls @ 123.5 ms/call = 864.7 ms; r=2: 36 calls @ 24.1 ms/call = 867 ms — prefill down-proj is ONE fused 512-token GEMM per layer (123 ms), vs decode's per-token MMVQ (6.7 ms, finding 23).
- other (attn/norm/embd): 73,029 calls @ 51.1 us/call = 3.73 s (74.8%) in the r=2 run — same ~75% CPU-span share as decode (finding 23: 76%).
- attn_o 16-36 us/call; ffn_shexp 83-269 us/call.

## ROOT CAUSE found: with-gpu-lock SILENTLY SHADOWS the probe .so
- env.sh:89 does LD_LIBRARY_PATH="$LX_BIN:$LD_LIBRARY_PATH" and LX_BIN=
  /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin
  whose libggml-sycl.so.0.17.0 has 0 layer-timer strings (vs 4 in $BINTREE).
- Any probe run THROUGH with-gpu-lock loads the timer-free lib -> rc=0, zero timer lines.
- The first prefill-budget run (105416Z, via with-gpu-lock, '0 timer lines') was NOT an env-drop;
  it was library shadowing. Direct invocation (this script now) keeps $BINTREE first.
- wlprobe2/3/4/5 (rc=134 'No device' or 0-timer) all reproduce this: wrapper prepends LX_BIN.

## Gotcha
- per-layer arrays (all_layer_us/calls) are declared+printed but NEVER incremented — per-layer budget rows are dead code (only bucket timer is live).
