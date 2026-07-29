# Mount Doom status — 2026-07-29 evening (dual win)

## LIVE NOW — first kernel win

**Control + Q4_K MoE dual-SwiGLU fuse (default ON)**

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **ctrl dual ON (A/B ub4k)** | **1144** | **110.1** | **+2.04%** |
| ctrl dual OFF (A/B ub4k) | 1134 | 108.2 | +0.45% |
| **formal harness (ub2k env)** | **1135** | **109.8** | **+1.63%** |
| baseline pin | 1139 | 107.35 | 1.0 |

- **Golden:** OK
- **Floors:** OK
- **Hit log:** `[lx-control-moe-dual] fuse hit (gate+up+swiglu)`
- **Formal:** `results/20260729T232021Z/score.json` · `LATEST_SCORE.json`
- **A/B:** `results/ctrl-q4k-dual-20260729T231807Z/`
- **Code:** `treebeard-base-control-latest` (`mmvq.cpp`, `ggml-sycl.cpp`, `topk-moe` fuse)
- Disable A/B: `GGML_SYCL_DISABLE_MOE_DUAL_SWIGLU=1`

### Why this worked
Package dual was **type-rejecting Q4_K** (Q5/Q6 only). Enabling Q4_K on package helped decode (~105→107) but package still loses ~28% prefill. Porting dual onto **control** (champion base) stacks the fuse without the package solo tax.

## POST-REBOOT A/B (pre-dual, serial ship flags)

| tag | pp512 | tg128 | note |
|-----|------:|------:|------|
| ctrl | 1136.18 | 107.58 | pre-dual champion |
| pkg | 818.25 | 104.95 | package solo tax |
| serial_pkg | 816.62 | 104.85 | dual-MoE package tree ≅ pkg |
| wf | FAIL | — | `failed to load model` |

## DONE (pre-reboot)

### Kernel trace (smoking gun)
- Decode oneDNN: **100% matmul time**
- Dominant: MoE expert GEMMs `512×2048 @ N≈9–29` (gate/up) + down `2048×512`
- Hot single shape: `1x512x2048:1x2048x9` — 9004 calls, 5.6% of GPU time
- Artifacts: `results/kernel-trace-20260729T210618Z/HOT_DECODE.md`, `HOT_PREFILL.md`, `TRACE_BOARD.md`

### Power
- Decode: **86 W / 37% of 230 W** → memory-bandwidth bound MoE

### Flag plateau
- Wave1 serial env/flag ladder: ±0.4% noise only

### Infrastructure
- `lx/` harness + baseline pin (pp1139 / tg107.35)
- Quest daemon scripts ready
- Worktrees: `lx-serial-kernel-pkg` (package dual-MoE), `lx-serial-kernel` (control hybrid abort)

## NEXT KERNEL LEVERS (in order)

1. **Stay on control binary as champion** for scored claims.
2. **Surgical dual/fusion port onto control** (not wholesale package tree):
   - `mmvq.cpp` MoE dual-SwiGLU for Q4_K tiny-N experts only
   - Avoid package paths that tank solo pp (~818 vs ~1136)
3. **Tiny-N MMVQ launch geometry** for N≈9–29 expert groups (trace smoking gun)
4. Optional: re-trace control ship path (native MMVQ, not oneDNN force) after any kernel change
5. Keep quest daemon as continuous rebench of control / candidates

## Target

Beat **tg128 107.9 / pp512 1147** on control+Q4 — first real win must be kernel, not knobs.

## Commands

```bash
# status
tail -f /home/frosty40/turbo/lx/results/quest-mount-doom/quest.log
cat /home/frosty40/turbo/lx/results/post-reboot-ab-20260729T224751Z/SUMMARY.json

# score a candidate
export LX_BIN=/path/to/candidate/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "kernel change"
```
