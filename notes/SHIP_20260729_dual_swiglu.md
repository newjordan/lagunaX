# Ship note — Q4_K MoE dual-SwiGLU on control (2026-07-29)

## Mission

Serial (one stream) Laguna XS 2.1 on Intel Arc Pro B70. Score is mlx.fast-shaped vs pinned same-box baseline, not multi-slot 513.

```
score = decode_speedup^0.75 * prefill_speedup^0.25
floors: both speedups ≥ 0.95
```

## Result (keep)

| config | pp512 | tg128 | score |
|--------|------:|------:|------:|
| baseline pin | 1139.15 | 107.35 | 1.000 |
| **control + dual ON** (formal harness ub2k) | **1135.27** | **109.82** | **+1.63%** |
| control + dual ON (A/B ub4k) | 1143.98 | 110.13 | +2.04% |
| control dual OFF (A/B ub4k) | 1133.79 | 108.17 | +0.45% |

- Golden smoke: **OK**
- Floors: **OK**
- Formal artifact: `results/20260729T232021Z/score.json`
- A/B: `results/ctrl-q4k-dual-20260729T231807Z/`

## What changed

**Patch:** `patches/0001-control-q4k-moe-dual-swiglu.patch`  
**Tree:** `treebeard-base-control-latest` (branch was `private/treebeard-base-control-20260728`)

Fuses `MUL_MAT_ID + MUL_MAT_ID + GLU(swiglu)` for MoE expert gate/up on serial decode (`ne12==1`):

- Dual Q4_K/Q5_K/Q6_K MMVQ reorder kernel in `mmvq.cpp`
- Fuse dispatch in `ggml-sycl.cpp`
- Hooked from `ggml_sycl_fuse` in `topk-moe.cpp` (before topk fuse)

Default **ON**. A/B off:

```bash
export GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1
```

Hit log:

```
[lx-control-moe-dual] fuse hit (gate+up+swiglu)
[lx-control-moe-dual] n_experts=8 nrows=512 ncols=2048 (first entry)
```

## Discovery path (why not package)

1. Kernel trace: MoE expert GEMMs dominate decode (tiny-N 512×2048 @ N≈9–29).
2. Package already had dual-SwiGLU but **type-rejected Q4_K** (Q5/Q6 only).
3. Enabling Q4_K on package (`patches/0002-package-enable-q4k-moe-dual.patch`) helped decode ~105→107 but package still **pp~820** vs control **~1140** (not dual-related).
4. Port dual onto **control** (champion solo base) → real score win.

## Apply patch

```bash
cd /path/to/llama.cpp   # control-like SYCL tree
git apply /path/to/lagunaX/patches/0001-control-q4k-moe-dual-swiglu.patch
# rebuild llama-bench / llama-cli with SYCL
```

## Harness

```bash
cd /home/frosty40/turbo/lx   # this repo
source env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "control dual"
cat results/LATEST_SCORE.json
```

## Do not

- Re-pin baseline to invent wins
- Claim multi-slot ~513 as serial score
- Wholesale-port package tree for solo (pp collapse)

## Next levers

1. Multi-token / grouped dual for prefill on control
2. Multi-row dual WG packing (package geometry) on control launch path
3. MoE down fuse after dual
4. Quant experiments that keep golden
