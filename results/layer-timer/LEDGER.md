# layer-timer — per-layer decode budget probe (GGML_SYCL_TIMER_ALL)

Instrumentation: env-gated std::chrono around the fused mul_mat dispatch in
ggml-sycl.cpp (LmheadProbe site, ~line 4405). Source snapshot:
`ggml-sycl.cpp.layer-timer` in this dir. Marker: `.lmhead-layer-timer-patched`.

## Build loop (proved end-to-end 2026-08-06, 07:58–08:02 CDT)
The src-lmhead-build tree's CMake was configured with
CMAKE_HOME_DIRECTORY=/home/frosty40/turbo/worktrees/treebeard-base-control-latest
(write-denied), so `cmake --build` does NOT recompile edited sources. The working
loop (mirrors scripts/lmhead-probe-cycle.sh):
1. edit src-lmhead/ggml/src/ggml-sycl/ggml-sycl.cpp
2. cp edited file over src-lmhead-build/ggml/src/ggml-sycl/ggml-sycl.cpp
3. icpx compile using the exact flags from src-lmhead-build/compile_commands.json
   (the `-c` path replaced by the edited file) → object
4. `bash CMakeFiles/ggml-sycl.dir/link.txt` (cwd ggml/src/ggml-sycl) relinks
   libggml-sycl.so into src-lmhead-build/bin
5. cp .so → results/src-repro-20260806T035656Z/bin/ (champion binary tree)
6. bench with env prepended, not clobbered (see env trap below)

## Env trap (rc=134 "No device of requested type" — root cause found)
setvars.sh adds /opt/intel/oneapi/umf/1.1/lib to LD_LIBRARY_PATH (holds
libumf.so.1, required by libur_adapter_level_zero.so). Any CLOBBERING export of
LD_LIBRARY_PATH drops it → Level Zero adapter fails to load → sycl-ls "No
platforms found" → llama-bench rc=134 at device select. Working pattern (same as
decode-trace.sh): PREPEND:
    export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/2026.0/lib:\
/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/mkl/2026.0/lib:<bin>:\
${LD_LIBRARY_PATH:-}"
Also required: ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0.

## Measurement (2026-08-06 08:03 CDT, p4/n128 decode run, rc=0)
[layer-timer] layer 39: 1672 us, 131 calls, 12.76 us/call

Interpretation: the fused lm_head group (l_out-39, Q6_K, 168 MB) is dispatched
131× over the whole run (131 tokens ≈ 1 dispatch per decode token), CPU-side
submission gap = 12.76 µs/token → **0.17% of the ~7.3 ms decode cycle**. Launch
overhead is NOT the decode bottleneck — decode is GPU memory-bound. (Corrected:
earlier "~500 µs/token ≈ 7%" guess was wrong arithmetic; per-layer dump shows
only l_out-* dispatches are counted, 1/token.) The 353.6 µs/token lm_head GPU
time (skip-diff, lmhead-probe-ledger) is hidden behind async queueing in this
CPU-gap probe, which is why sycl-trace per-kernel timing also came up empty.
Next probe (bucket timer, staged): category the CPU gaps across ALL fused
dispatches (attn_o/ffn_shexp/ffn_out/lmhead) to bound total CPU submit.

## Status
- instrumented champion .so installed in results/src-repro-20260806T035656Z/bin
- bench-champion-cycle --bin=<that tree> running (proof-suite in progress,
  13:05Z start) to certify speed-neutrality of the instrumented binary

## Measurement 2 (2026-08-06 09:28 CDT, pp512/tg128 official shape, rc=0)
[layer-timer] bucket order: 0=attn_o 1=ffn_shexp 2=ffn_out 3=lmhead 4=tok_embd 5=other
- attn_o:   5240 calls,  11.67 us/call  (61.1 ms)
- ffn_shexp: 5109 calls,  52.09 us/call (266.1 ms)
- ffn_out:    131 calls, 6703.51 us/call (878.2 ms)   <- MoE down-proj, longest GPU drain
- other:    65669 calls,  57.71 us/call (3.79 s)      <- attention K/Q/V/norm dispatches
- lmhead:    131 entries = exactly 1 per decode token, 12.76 us/call CPU gap (0.17% of cycle)

Interpretation: CPU-side dispatch gaps partition ~5 s of wall time; 'other'
(attention/norm ops) dominates at 76%, MoE down-proj (ffn_out) at 17.6% with
6.7 ms per-call drains — the GPU is the bottleneck, CPU submit is cheap. The
fused ffn_shexp/ffn_out/l_out names only appear for layer 39 (final fused
group); lm_head fires exactly once per decode token. Decode = GPU memory-bound.

## 2026-08-07 — bucket-instrumentation audit (run-20260806T111546Z)
- Bucket call totals exactly match the per-suffix dispatch histogram: bucket 5240+5109+131+65687 = 76167 = histogram total. No dispatch is lost or double-counted.
- The lmhead bucket (3) and tok_embd bucket (4) rows NEVER print — zero calls. Cause: the fused lm_head group's mul_mat node is NAMED "ffn_shexp-39" (fuse log: mm='ffn_shexp-39' add='ffn_out-39' add2='l_out-39'), and the classifier checks "ffn_shexp" before "l_out-" → all 917 fused lm_head GEMVs are counted in the ffn_shexp bucket. "l_out-" is unreachable as a mul_mat dst name in the fused regime; the 131 l_out-39 calls are the NON-fused fallback dispatches (bucket ffn_out = 131 calls, 6619.60 µs/call — resolves the 917-vs-131 mystery from the earlier lead).
- Per-layer spans and bucket spans use different bases: per-layer sum 65,313,203 µs vs bucket sum 4,995,486 µs (~13×) — per-layer rows are token-cycle-phase counters (finding 39), bucket rows are dispatch-to-dispatch deltas.
- lm_head's true wall contribution stays small: 917 × ~130 µs (probe) ≈ 0.12 s of the ~5.0 s dispatch span ≈ 2.4% — consistent with the +5.12% max / +1.5-2.5% realistic lm_head ceiling. The bucket audit explains WHY lm_head can never carry the 1.25 bar.
