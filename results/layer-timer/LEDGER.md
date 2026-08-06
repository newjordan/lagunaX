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
131× (≈ 4 prefill + 128 decode − warmup), with a CPU-side submission gap of
12.76 µs between successive entries. The CPU submits ~40 dispatches/token →
~500 µs/token CPU submit (≈7% of the ~7.3 ms decode cycle) — decode is GPU
memory-bound, NOT launch-bound. The 353.6 µs/token lm_head GPU time (skip-diff,
lmhead-probe-ledger) is hidden behind async queueing in this probe, which is why
the earlier sycl-trace per-kernel timing also came up empty.

## Status
- instrumented champion .so installed in results/src-repro-20260806T035656Z/bin
- bench-champion-cycle --bin=<that tree> running (proof-suite in progress,
  13:05Z start) to certify speed-neutrality of the instrumented binary
