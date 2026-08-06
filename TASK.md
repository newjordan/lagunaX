# lx — Laguna XS 2.1 serial on Arc B70

Shadow of [mlx.fast](https://mlx.fast/) / [Layr-Labs/mlxfast-challenge](https://github.com/Layr-Labs/mlxfast-challenge)
on Intel Arc Pro B70 (SYCL), not Apple Silicon (MLX).

## What this is

Optimize **serial** (one token stream) prefill + decode for Poolside Laguna
XS 2.1 on B70 while preserving greedy correctness.

```text
score = decode_speedup^0.75 * prefill_speedup^0.25
decode_speedup_floor  = 0.95
prefill_speedup_floor = 0.95
```

Speedups are candidate tok/s over a **pinned same-box baseline** (not vs M5).

## What this is not

| Out of scope | Why |
|---|---|
| Multi-slot / `-np N` aggregate tok/s | Capacity game; different bottleneck |
| `~/Laguna-XS-2.1-B70-Turbo` max-push / absolute-limit waves | That campaign ranked concurrent decode sum |
| Package multi-slot ship (np64 ~513) | Valid product number; not this score |
| Cross-silicon comparison to M5 Max 106/4050 | Different quant, memory system, kernels |

Re-using binaries or env knobs from the multi-slot campaign is fine when they
help **solo** pp/tg. Claiming multi-slot numbers as an mlx.fast shadow is not.

## Frozen timed window (mlx.fast-shaped)

| Phase | Shape | Proxy tool |
|---|---|---|
| Prefill | 512-token prompt | `llama-bench -p 512 -n 0` → **pp512** |
| Decode | 128 tokens after short prompt | `llama-bench -p 0 -n 128` → **tg128** |

Official mlx.fast also uses a 512-token decode seed + teacher-forced 128 steps
and hidden correctness. This harness starts with llama-bench stand-ins and a
local greedy golden smoke; upgrade later if we want bit-exact parity with
their fixtures.

## Correctness

Hard gate before score counts:

1. **Smoke golden** — fixed prompt, greedy `temp=0`, first N tokens must match
   a checked-in reference captured from the baseline binary.
2. Floor check — both speedups ≥ 0.95 or score is null.

Token mismatch → no score. Regression vs baseline is allowed only if floors
still pass (e.g. large prefill win with tiny decode loss).

## Baseline contract

Pinned once under thermal/power-stable conditions, written to
`baseline/baseline.json`. Never silently re-baselined.

- Model: Laguna XS 2.1 GGUF (default Q4_K_M)
- Binary: treebeard **base-control** (solo-fast path)
- Device: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_AFFINITY_MASK=0`
- Flags: see `env.sh` (`NGL`, `UBATCH`, KV type, threads)

## Editable surface (this track)

Anything that changes serial pp512 / tg128 without breaking golden:

- llama.cpp / treebeard SYCL kernels (GEMM, MoE, attention, RMSNorm, quant)
- Runtime flags and `GGML_SYCL_*` env knobs (solo path only)
- Offline quant / weight transform of Laguna (if golden still holds)
- Graph / fusion / batching choices that affect single-stream decode

Not editable for scored claims: harness scoring code, baseline pin, golden
files (except regenerating golden after an intentional contract bump).

## Local loop

```bash
source env.sh
./scripts/bench-serial.sh --baseline   # once: pin baseline/
./scripts/bench-serial.sh              # candidate → results/<stamp>/score.json
./scripts/golden-smoke.sh              # greedy match gate
```

**B70:** one GPU client at a time. Concurrent Level-Zero jobs wedge the driver.
See `notes/B70_NO_CONCURRENT_GPU.md`. Harness scripts take `results/.b70-gpu.lock`.

## Scoring detail

```
decode_speedup  = candidate_tg128 / baseline_tg128
prefill_speedup = candidate_pp512 / baseline_pp512
score           = decode_speedup**0.75 * prefill_speedup**0.25
increase_pct    = (score - 1) * 100
```

Higher is better. Report raw tok/s always; percent increase alone is not enough.
