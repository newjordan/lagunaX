# FINDING 2026-08-09 — structural frame flip: host-thread submission + XMX non-use

Scout iteration. Sources opened this run; claims below are pinned to paths.

## Frame inverted

Prior campaign treated pp512 as a pure *device* expert-loop / launch-count wall and decode as residual serial edges. External B70 production kits + in-tree markers invert two structural assumptions:

1. **Host submission concurrency** is a first-order knob on B70 SYCL (not just kernel shape).
2. **`SYCL_USE_XMX` is a gate name, not an XMX consumer** — the live decode quant path is dp4a MMVQ.

## Local evidence

### A. Scored harness runs `-t 16`, not `-t 1`

`results/20260809T191208Z/metrics.json` (board promotion of Q+K rope merge):

- `flags.threads` = 16
- `flags.threads_batch` = 16
- `flags.sycl_disable_graph` = 1
- `flags.flash_attn` = -1
- board absolute: tg128 152.55, pp512 1170.98, score 1.31047

External community kit (Hal9000AIML/arc-pro-b70-ubuntu-gpu-speedup-bugfixes README + docs/tuning.md, crawled 2026-08-09) states as a *measured* B70 rule:

> `-t 1` for all GPU tiers. More host threads fight for the GPU submission queue. Single-thread dispatch wins.

That kit is not our tree; it is an independent 4×B70 production deployment. Our score contract pins flags — changing `-t` alone would re-baseline — but the *disagreement* is new: nobody in prior lx directions treated host-thread count as the structural constraint.

### B. In-order queue + multi-thread host is the documented serialization model

`ggml-sycl.cpp` documents ids-memo safety as depending on the in-order queue serializing pool-slot reuse. Under that model, N host threads cannot create true multi-stream overlap; they only contend for submit-side serialization into one in-order L0 queue.

### C. `SYCL_USE_XMX` is self-documented as non-XMX

`common.hpp`:

```
// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif
```

So the compile-time symbol that gates MMQ batch windows is *named* XMX but explicitly does not drive XMX silicon. Live decode quant GEMV is MMVQ via `vecdotq.hpp` `dpct::dp4a(...)`.

External (Donato Capitella B70 walkthrough, 2026): "custom quantized GGUF kernels do not utilize the XMX matrix engines, instead falling back to dp4a" — consistent with local code.

### D. External "K-quant DMMV subgroup 32→16" lead is largely already landed here

- `CMakeLists.txt` sets `GGML_SYCL_WARP_SIZE=16` for INTEL targets; build flags confirm `-DGGML_SYCL_WARP_SIZE=16`.
- `dmmv.cpp` Q4_K body already uses the `for (int im = 0; im < 2; ++im)` half-loop + `WARP_SIZE/2` reduce (Hal9000 patch ada8c01bc shape), not CUDA-era 32-lane half-split.
- Open lead #15 (DMMV subgroup retune) should be reclassified: *kernel already 16-wide*; remaining question is whether scored path *engages* DMMV at all (`GGML_SYCL_PRIORITIZE_DMMV` defaults 0 → MMVQ preferred when reorder is on).

### E. External MoE prefill +70% (llama.cpp #23142 counting-sort) is also already in-tree

`mmid_counting_sort_rows` + precomputed `k_copy_src1_to_contiguous` (no device atomic scan) present around ggml-sycl.cpp:4794 / 6560 / 6751. Not a free 70% left on the table for Laguna.

### F. External B70 open bug (2026-07)

ggml-org/llama.cpp#25455 (open): `MUL_MAT_ID` prefill path wrong results on Arc Pro B70 — 28/792 `test-backend-ops` failures, large mismatches, *not* fixed by reverting counting-sort; points at lower shared `ggml_sycl_op_mul_mat` path. Decode ne12==1 MMVQ path claimed unaffected. Laguna KLD is green so our Q4_K/Q6_K shapes may be outside the failing set — still a structural risk for any new multi-token expert path.

### G. External small-f32 oneMKL bypass is already present

`ggml-sycl.cpp:2878-2881`: `use_mkl_direct = gemm_flops < 256*256*256` already gates DNN vs oneMKL for the float GEMM path.

## Score state (this run)

- `results/LATEST_SCORE.json`: score ≈ 1.31047, decode_speedup ≈ 1.421, prefill_speedup ≈ 1.028
- `scripts/loop-accept.sh` target default 1.40 → still red

## Hypotheses (untested this iteration)

1. Same-session A/B of scored binary with `-t 1` vs `-t 16` (and forced re-pin of baseline under `-t 1`) moves pp512 more than any remaining fusion <1% lever, because 16 host threads contend on one in-order queue.
2. Forcing a true XMX consumer on decode dense (bf16/fp16 slab MM or oneDNN GEMM for the residual class) beats further dp4a MMVQ micro-tuning.
3. Porting llm-scaler's "persistent zero-gap MoE GEMM" (2 SYCL groups / XeCore) framing into ggml-sycl expert loop — keep experts resident, zero inter-expert gap — is the only structure that can cut the 417 ms MoE wall without MMQ multi-col wedge.

## What this direction is NOT

Not another env-knob sweep, not another norm/rope fuse, not another DNN dead-name A/B, not DMMV 32→16 (already done), not counting-sort (already done).
