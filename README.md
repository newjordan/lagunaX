# lagunaB70

**Faster decode for Laguna-XS 2.1 on the Intel Arc Pro B70. 107 → 138 tokens/sec.**

Replacement SYCL kernels for llama.cpp, plus the harness that measures them.

- Model: Laguna-XS 2.1 (33B mixture-of-experts), Q4_K_M
- GPU: one Intel Arc Pro B70, one request at a time

## Numbers

`control` = same GPU, same model file, same flags, these kernels off.

| | control | lagunaB70 | |
|---|---:|---:|---:|
| Writing an answer (128 tokens) | 107 tok/s | **138 tok/s** | +29% |
| Reading a 512-token prompt | 1139 tok/s | 1173 tok/s | +3% |
| Reading a 2048-token prompt | 1954 tok/s | 2029 tok/s | +4% |
| Reading an 8192-token prompt | 1880 tok/s | 1950 tok/s | +4% |

Every run is banked under [`results/`](results); the side-by-side that produced
the prompt rows is [`results/laguna-ab-20260730T235650Z/`](results/laguna-ab-20260730T235650Z).

## How it gets faster

A decode step burns most of its time launching tiny GPU jobs and re-reading the
same weights. Each patch in [`patches/`](patches) glues jobs together so the data
is touched once:

- gate + up + SwiGLU as one expert kernel
- the expert router (top-8 of 256) as one kernel, replacing a five-kernel chain
- residual adds folded into the matmul that produced them
- RMS-norm + multiply, RoPE + row-store, softplus + multiply — all fused

52 patches are banked, each with a note in [`notes/`](notes) and its measured gain.

## Quality gate

A change ships only if the model produces byte-identical greedy output
(`scripts/golden-smoke.sh`) and wikitext-2 perplexity holds at 12.60. Several
faster fuses are off by default because they failed that bar; the notes say which.

## Try it

```bash
source env.sh                          # device, binary, model, flags

./scripts/bench-serial.sh --baseline   # pin control, once
./scripts/golden-smoke.sh --capture    # record expected output, once

./scripts/golden-smoke.sh              # output must still match
./scripts/bench-serial.sh --note "what changed"
cat results/LATEST_SCORE.json
```

## Warning: one GPU job at a time

Two Level-Zero programs on this card at once wedge the `xe` driver and cost you a
reboot. The scripts take an exclusive lock automatically. For ad-hoc work, borrow it:

```bash
./scripts/gpu-status.sh                        # who holds it
./scripts/with-gpu-lock --reason ppl -- "$LX_BIN/llama-perplexity" -m "$LX_MODEL" ...
```

## Layout

```text
env.sh          device, binary, model, serial flags
patches/        the kernel patches
notes/          one note per change, with its measured gain
scripts/        bench-serial.sh, golden-smoke.sh, score.py, GPU lock
baseline/       pinned control numbers
results/        every run, timestamped
```

## Scope

All numbers are single stream, one request at a time, measured against the pinned
B70 control above.
