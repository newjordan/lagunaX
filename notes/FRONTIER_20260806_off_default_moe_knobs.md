# FRONTIER 20260806 — OFF-by-default MoE fusion knobs on the native quantized path

## New direction
The scored binary compiles in **6 native-path MoE optimization features that
default to OFF** and are never set in env.sh. These operate on the 88.7%
native-quantized GEMV path (dir 18), NOT the oneDNN path that all prior fusion
directions (5, 10) targeted. Every one is a **runtime env var** — zero recompile.

## Evidence

### Gate 1 resolved: GGML_SYCL_GRAPH IS compiled in
- `grep -o GGML_SYCL_GRAPH build-base/compile_commands.json` → 86 matches
- `grep -o GGML_SYCL_GRAPH build/compile_commands.json` → 87 matches
- CMake: `option(GGML_SYCL_GRAPH ... ON)` at ggml/CMakeLists.txt:250
- `target_compile_definitions(ggml-sycl PRIVATE GGML_SYCL_GRAPH)` at ggml-sycl/CMakeLists.txt:194
- **Kills open lead #17**: the env var is NOT inert; graph code IS compiled.
  The block is purely runtime `check_graph_compatibility()` returning false.

### The 6 OFF-default MoE knobs (all read from env, all default 0)
Source: ggml-sycl.cpp:81-93 (declarations), 277-290 (env reads)

| Knob | Default | What it does | Line |
|------|---------|-------------|------|
| `MOE_DOWN_WEIGHTED_SUM_FUSION` | 0 | Fuses down-expert × weight-reduce into one kernel (modes 2/3 = atomic/local). Eliminates separate weighted-sum dispatch on 60,229 down calls (31.5% decode) | 5038-5075 |
| `MOE_Q6_DOWN_NX2_SPECIALIZE` | 0 | Specialized Q6_K GEMV for nx2 layout — down experts are ALL Q6_K (F33) | mmvq.cpp:2831 |
| `MOE_ACT_Q8_CACHE` | 0 | Caches f32→Q8_1 activation across expert calls in one decode step; avoids redundant `quantize_row_q8_1_sycl` launches | 4660-4735 |
| `MOE_GATE_UP_Q8_HANDOFF` | 0 | Produces fused gate/up output in Q8_1 format → direct feed to down expert, no re-quant | 4703-4735 |
| `MOE_SHARED_GATE_UP_FUSION` | 0 | Gate/up fusion for the shared expert (M=1024 tier, 88 ms, 2.4%) | 4944 |
| `MOE_WEIGHTED_TAIL_FUSION` | 0 | Tail-variant of expert weighted-sum fusion | 280 |

### The activation-cache leverage (the biggest single lever)
At line 4660-4735, the fused gate/up path (`moe_gate_up_fusion==3`, already
ON by default) calls `quantize_row_q8_1_sycl` to convert the f32 activation → Q8_1.
Without `MOE_ACT_Q8_CACHE`, this fires on **every expert call** even though the
activation vector is IDENTICAL across the 8 experts in the same layer-stage.

Per decode step:
- Gate/up stage: 8 experts × 39 MoE layers = **312 redundant quantize kernels**
  (1 unique activation per layer, quantized 8× instead of 1×)
- Down stage: each expert's input differs (swiglu output), so cache doesn't
  directly apply — BUT `MOE_GATE_UP_Q8_HANDOFF` makes the fused gate/up kernel
  output Q8_1 directly, feeding the down expert with zero re-quant

With cache ON: gate/up stage drops from 312 → 39 quantize launches (87.5% reduction).
With handoff ON: down stage eliminates its 312 quantize launches entirely.
Combined: **624 → 39 kernel launches per decode step** on the activation-quant axis.

### env.sh explicitly does NOT set these
env.sh only sets: DISABLE_GRAPH=1, DISABLE_MUL_MAT_ADD_FUSE=0,
DISABLE_MOE_DUAL_DOWN=1, DISABLE_MOE_DUAL_MULTITOKEN=1.
It never exports any of the 6 MOE_ENABLE knobs above — so binary defaults (all 0) apply.

## Experiment plan (zero recompile)
Test each knob individually via env override on the scored binary:
1. `MOE_ACT_Q8_CACHE=1` alone
2. `MOE_GATE_UP_Q8_HANDOFF=1` (requires `MOE_GATE_UP_FUSION=3` which is already default)
3. `MOE_DOWN_WEIGHTED_SUM_FUSION=3` (local-reduce mode)
4. `MOE_Q6_DOWN_NX2_SPECIALIZE=1`
5. All-on combination
