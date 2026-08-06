# Frontier — PCIe fabric + copy-engine topology (2026-08-01)

## Direction (new; not dirs 1–24)

**Host↔device PCIe transfer regime and hardware engine topology** — invert the
"optimize kernels / graph / oneDNN" frame: measure the physical fabric and the
GPU engines that sit *under* every H2D/D2H and host barrier. Distinct from:
- dir 24 (software `in_order` queue property)
- dir 9 (oneDNN `create:cache_hit` host serialization)
- prior host↔device traffic note (call counts + barrier pattern only; no BW measure)
- dual-queue counts D2H experiment (SHIP copy-q; functional overlap only)

## Evidence

- Probe: `results/pcie-bw-probe-20260801/probe.txt` (this iteration)
- Code: `ggml-sycl.cpp` set_tensor; `dpct/helper.hpp` create_queue_impl
- Score: `results/LATEST_SCORE.json` decode_tok_s=138.018 → 7.245 ms/token
- Champion log: `results/20260731T172351Z/llama-bench.log` model_size / ngl / mmap

## Findings

1. **Measured PCIe is healthy, not Gen1×1.** 256 MiB USM device memcpy:
   H2D **14.829 GB/s**, D2H **7.584 GB/s** (≈2:1 asymmetry). [probe:34-35]
2. **Sysfs PCIe attributes are false.** Endpoint `0000:0d:00.0` reports
   `current/max = 2.5 GT/s ×1` even in `power_state=D0` during/after 14+ GB/s
   transfers. Parent bridge `0000:0b:00.0` correctly shows 16 GT/s ×16
   (max 32 GT/s ×16). Sysfs link status cannot diagnose this platform. [probe:3-11]
3. **Logits-sized D2H is small but non-zero wall tax.** 401408 B D2H =
   **0.118 ms/call** (3.396 GB/s effective) ≈ **1.63%** of the 7.245 ms/token
   decode budget @ 138 t/s. Small-transfer efficiency collapses vs 256 MiB
   bulk D2H (7.58 → 3.40 GB/s). [probe:36; LATEST_SCORE]
4. **Dual-GT is compute+media, not dual-compute.** `gt0-rc` engines =
   `{bcs, ccs, rcs}`; `gt1-mc` engines = `{vcs, vecs}` only. Media GT cannot
   absorb MoE GEMV. [probe:12-15]
5. **BCS exists as a hardware queue family and is unused by ggml-sycl.**
   OpenCL reports `ccs (1)` + `bcs (1)`. `create_queue_impl` builds
   `sycl::queue(*this, eh, property_list{in_order...})` with **no**
   `queue_family` / BCS selection — all `set_tensor`/`get_tensor` memcpys use
   `default_queue()` (CCS compute path). [helper.hpp:795-809; ggml-sycl.cpp:573-580,605-606]
6. **Linux set_tensor still applies the PVC mmap bounce on Arc B70.**
   `#ifndef _WIN32` → `malloc` + host `memcpy` + device `memcpy().wait()`,
   comment names PVC. Load mode is `mmap`, model_size **20,270,574,592** B
   (~18.9 GiB), `n_gpu_layers=99`. Bounce doubles host DRAM traffic for every
   weight upload leaf. [ggml-sycl.cpp:575-581; llama-bench.log:31,41,50]
7. **SYCL-reported device memory topology is sparse/suspicious.**
   `memory_clock_rate=2800`, `memory_bus_width=64` → naive DDR peak
   **44.8 GB/s**. If that query is trustworthy, it reframes the entire
   expert-stream "BW-bound" story as a **~45 GB/s device-memory ceiling**,
   not a kernel-dispatch tax — but the unit/meaning of the query on Xe2 is
   uncertain (may under-report multi-channel GDDR). [probe:23-25]

## Hypotheses (untested)

1. Pinpoint whether naive 44.8 GB/s is real via a pure device-side STREAM/copy
   kernel (no PCIe); if real, weight-layout / quant / fusion strategy should
   optimize for ~45 GB/s not hundreds.
2. Route `get_tensor`/`set_tensor` through a BCS-family queue *without*
   sharing the compute in-order stream — prior counts-copy-q dual queue lost
   ~170 pp on a different path; BCS-specific family select may differ.
3. Drop the Linux mmap bounce for non-PVC discrete (`usm_device` + pageable
   or pinned host) once B70 is confirmed non-PVC; cut per-leaf host copy +
   alloc free on every decode step leaf upload.
4. Device-side argmax/top-k of logits eliminates the 0.118 ms D2H + post-sync
   chain (complements lm_head ROI ~0.3 ms GEMV).

## Not claimed

- Wall-time share of `queues_wait_and_throw` vs pure PCIe wire time (need
  host timers around barrier vs memcpy).
- That sysfs Gen1×1 is a kernel bug (could be wrong attribute mapping for
  multi-function switch topology).
- That BCS memcpy will win on B70 (prior dual-queue D2H regressed).
