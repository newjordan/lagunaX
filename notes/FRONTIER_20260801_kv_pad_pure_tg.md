# Frontier — Pure-tg128 score under hard n_kv≥256 pad (2026-08-01)

## Direction (PIVOT — not dirs 1–24)

**Score-workload geometry × graph-constancy KV pad.** Invert "tg128 = decode after pp512"
and "attention seq_len=256 is just model window." The scored decode arm is pure generation
from empty context under a hard `get_n_kv` floor of 256, so every attention pays full-pad
width while true used cells average 64.5. Distinct from:
- dir 23 / supports_op burst (host query cadence)
- dir 19 (SYCL command-graph capture)
- dir 15 (prefill M-class)
- host↔device logits traffic
- FA GQA head asymmetry (finding #8 shapes only)

## Evidence opened this iteration

- `scripts/bench-serial.sh:83-85` — tg arm is `-p 0 -n $LX_TG`
- `llama-kv-cache.cpp:1226-1233` — `n_pad_cur = max(n_pad, 256u)` then pad used cells
- `results/ktrace-post-brownout-20260731/trace.log` — all 1280 SDP execs seq=256; total 74.10 ms
- `results/ktrace-tip-20260730/decode-ggml/trace.log` — cache_v ne1=256; mask init 48 vs set 1088

## Findings (admitted)

See panel scout output this turn.

## Hypotheses

1. Temporary `n_pad_cur` floor 32/64 for pure-tg128 would cut attention FLOPs ~4× if kernels scale with n_kv; measure tg128 + graph-reuse rate (`n_reused`).
2. Harness alternate: `-p 512 -n 128` changes score semantics (full window) — report separately, do not re-pin baseline silently.
3. Pad exists for graph reuse (comment at 1226); reuse already works (set≫init) — ROI is device FA work, not host rebuild.

## Uncertainty

- Whether SYCL FA/SDP time is linear in n_kv on B70 (needs A/B with patched pad).
- SDP 74 ms is same-capture aggregate as prior matmul 3704 ms; attention share ~2% of that matmul+sdp sum — upper bound on pad ROI if linear is modest unless host/FA path is larger than oneDNN sdp timer shows.
