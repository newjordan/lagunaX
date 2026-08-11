# SHIP 2026-08-08 — moe-down: lift weights/mul buffer-overlap hard-reject

Candidate: `results/lx-moe-down-mul-alias-20260808T042541Z/`
Formal run: `results/20260808T043901Z/` · KLD: `results/kld-20260808T043821Z/`
Blob: `5342b3c9a4a25f3e…` (was `bf69f35c…`)

## Change (one variable)
`ggml-sycl.cpp` `ggml_sycl_fuse_moe_down_weighted()`: the fused down-path
weighted-reduce was hard-rejected when `mul` (the per-expert weighted tensor)
overlapped `dst` in the product allocator. The 2026-07-30 measured regression
that motivated the reject was with weights/dst alias on *other* models; on
Laguna decode the **weights** tensor is the alias (tiny [1,8,1] route-weights
buffer), and `mul`-overlap is an artifact of the scratch allocator — fusing
does not alias-read the per-expert values. The reject is now limited to
`weights`/`mmid` (genuine read-during-write hazards); `mul` is allowed.

## Gates
- golden-smoke: GOLDEN OK
- quality-gate-kld: mean KLD -0.000000, same_top 100.0%, RMS dp 0.001% → PASS
- bench-serial: decode 142.20 t/s (+0.63% vs board 141.53), prefill 1145.02,
  score **1.2362987874** (promoted, no rejection reasons)

## Ledger
- `[lx-control-moe-down] fuse hit (weighted reduce) embd=2048 k=8 tokens=1` now
  fires on the decode down path (129-step trace showed it skipped on every step).
- The single prefill down op (tokens=8) still skips: genuine weights/dst alias
  there — leave it (safety), or a future candidate can reorder the allocator.

## Findings worth banking
1. **"GPU down" was a false alarm — LD_LIBRARY_PATH.** Manual runs that set
   `LD_LIBRARY_PATH` to only compiler/dnnl/mkl (+build/bin) fail device
   enumeration with "No device of requested type available": the Level-Zero
   adapter needs `/opt/intel/oneapi/tcm/1.5/lib` and `/opt/intel/oneapi/umf/1.1/lib`
   (libumf.so.1). `scripts/decode-trace.sh` works because it *appends*
   `${LD_LIBRARY_PATH:-}` after sourcing setvars. Always `source env.sh` (or
   setvars) and only *prepend* build/bin. Verified: GPU never wedged; L0 probe
   via ctypes and sycl-ls enumerate fine throughout.
2. sycl-trace kernel-name extraction no longer works on this tree's trace
   output (0 kernel lines) — the op-level census via the `[lx-…]` fuse lines in
   llama-bench stderr is the reliable decode launch ledger.
