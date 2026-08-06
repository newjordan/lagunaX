# Emb probe (get_rows / tok_embd) — batch-512 embedding budget

## 2026-08-06 — first get_rows measurement at pp512 (lead 19 closed)

Cycle: `scripts/embd-probe-cycle.sh` → patches/embd-bucket-timer.patch (env-gated
GetRowsTimer chrono inside `ggml_sycl_get_rows`, ggml-sycl.cpp:3303) → probe build
→ golden smoke (probe lib, GOLDEN OK) → official-geometry pp512 bench (r=5) →
revert + champion .so restore (md5 2361042a… unchanged).

Run: results/embd-probe-20260806T165247Z/

### Measured
- `[layer-timer] get_rows (embedding gather): 50180 us, 12 calls, 4181.67 us/call`
  (bench.stderr tail). That is the first-ever wall attribution of the batch-512
  embedding gather. 12 calls for 5 reps × pp512 (tok_embd once per rep + warmup).
- vs dispatch span: bucket total = 13188+65064+872628+4298447 = 5,249,327 µs;
  get_rows = 50,180 µs → 0.96% of the dispatch span. Embedding is NOT a pp512 lever.
- pp512 in probe run: 1112.1 ± 9.0 t/s (timer-live; same-window shadowed runs are
  ~1157, so the ~4% is timer+geometry, not the probe payload — golden passed).

### NEW harness bug class (root cause of the "bin64/llama-server" golden failure)
- oneAPI setvars.sh (sourced by env.sh) exports the GENERIC name `BIN_DIR=bin64`
  via /opt/intel/oneapi/advisor/2026.0/env/vars.sh:68 (and vtune/2026.1/env/vars.sh:68).
- Any script that sources env.sh AFTER defining its own `BIN_DIR` gets it clobbered:
  `$BIN_DIR/llama-server` → `bin64/llama-server` → "missing llama-server" rc=1.
- Fix: renamed the script-local var to EMB_BIN (commit pending). Watch for `BIN_DIR`
  in any future harness script that sources env.sh.

### Verdict on lead 19
- The embedding gather is measured, small (~1% of dispatch span, ~4.2 ms/call at
  batch 512), and therefore NOT the missing pp512 cost. The 1.018× prefill gap is
  not hiding in tok_embd; remaining prefill attention stays on the fused 512-token
  down-GEMM (ffn_out bucket: 872,628 µs / 6 calls / 145 ms per call) and the "other"
  bucket (4.3 s / 135,519 calls / 31.7 µs per call).
