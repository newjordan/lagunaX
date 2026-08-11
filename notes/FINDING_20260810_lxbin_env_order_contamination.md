# FINDING 20260810 — LX_BIN env-order bug contaminates every leg that overrides LX_BIN AFTER sourcing env.sh

## The bug (proven with LD_DEBUG=libs)
- env.sh:97 does `export LD_LIBRARY_PATH="${LX_BIN}:${LD_LIBRARY_PATH:-}"` — captured at SOURCE time.
- Any script that does `source env.sh` and THEN `export LX_BIN=...` leaves LD_LIBRARY_PATH pointing at
  the OLD LX_BIN (default = /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin).
- The loader searches LD_LIBRARY_PATH BEFORE RUNPATH → llama-bench loads the TREEBEARD WORKTREE's
  libggml-sycl.so.0 (sha abb033c4) instead of benchmark/kernel/build/bin's (sha 24e7c8c8/fbb3ef5e).
  Proof: `LD_DEBUG=libs ... llama-bench -h` → "calling init: .../treebeard-base-control-latest/build-mmadd-decode/bin/libggml-sycl.so.0".
- The treebeard .so has its OWN control-line instrumentation ("device counting-sort+prefix+ev",
  fused moe-dual/moe-down engines) and NONE of the kernel tree's diag gates (no LX_DIR, no
  GGML_SYCL_LX_MMVQ_PREFILL, no lx-gb-fix). Its default pp512 MoE path differs (fused mmvq engines;
  the oneMKL per-expert loop never runs → cap probes and DIAG_SKIP_QUANT appear dead).

## Correct procedure (the only valid one)
`export LX_BIN=$ROOT/benchmark/kernel/build/bin` MUST precede `source env.sh`.

## Corrected measurements (kernel .so, env correct)
- ctrl pp512: 1172.6 t/s ≈ board 1173.3 — oneMKL per-expert loop IS the default: LX_DIR probes
  (temporarily added then removed) counted op_mul_mat_sycl=30,633 and mm_dispatch=67,974 over r=2
  (~15,317/pass), LX_IDS_ONCE hits=228. The findings' oneMKL-loop narrative is VALID for this build.
- GGML_SYCL_LX_GEMM_BATCH=1 (OOB/step5/step2 fixed): reproducible hang (rc=124 at 240 s, host blocked
  in S state, no output) — the "strided gemm_batch dead on B70" verdict stands even on the fixed shape.
- GGML_SYCL_LX_MMVQ_PREFILL=1 (Q6_K expert slices chunked to ncols<=8 q8_1 MMVQ): engages (op_mul_mat_sycl
  drops 30,633→25,851) but 974.7 t/s vs 1172.6 ctrl = -17%. Multi-col q8_1 MMVQ loses to oneMKL fp16
  per-expert for ncols 2-8 on B70. Dead end.

## Recommended next steps
- Add a sha256 assert of $LX_BIN/libggml-sycl.so.0.17.0 against the board receipt inside bench-serial.sh
  (open lead 16) — the env-order bug is exactly the failure mode it would catch.
- Audit prior results/ legs for the same contamination before trusting any A/B whose runner exported
  LX_BIN after sourcing env.sh.
