# Mount Doom status — 2026-07-30 (fused sigmoid+add tip; mmid prefix-sum default)

## LIVE NOW — tip + research track

**Scored tip:** MoE dual + hybrid mode2 + **fused sigmoid+add** + dense dual + moe-down + device mmid sort  
**Default infra (not tip stamp):** mmid **prefix-sum + counts event-wait** (pack overlap)  
**Research (golden FAIL / opt-in):** full fused norm · integrated down · multi-token MMVQ · dual multi-token · multi-sg dual

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| **tip fused sigmoid+add** | **1148.9** | **120.2** | **+9.09%** |
| + prefix + **event-wait** | 1144.2 | 120.3 | +9.04% |
| + mmid prefix-sum (stream wait) | 1143.0 | 120.3 | +8.99% |
| prior + device counting-sort | 1144.8 | 118.6 | +7.91% |
| baseline pin | 1139 | 107.35 | 1.0 |

Formal tip: `results/20260730T053204Z/` · event-wait: `results/20260730T055755Z/`  
Notes: `SHIP_20260730_router_sigmoid_add.md`, `SHIP_20260730_mmid_counts_event.md`  
Patches: `0011` sigmoid+add · `0012` prefix · `0013` event-wait  
Kill fused sig+add: `GGML_SYCL_DISABLE_ROUTER_SIGMOID_ADD=1`  
Kill mmid device sort/prefix/ev: `GGML_SYCL_DISABLE_MMID_DEVICE_SORT=1`

### Prefill mmid wait lesson
- Full `stream->wait()` **after pack** → ~−20 pp (host bubble before GEMMs).  
- Full wait **before pack** → pack serialized behind host.  
- **`sycl::event` wait on counts D2H only** → pack overlaps host offsets; GEMMs stay after pack.

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

1. **Stay on control binary as champion** for scored claims (dual ON).
2. **True zero mid-op sync** — pinned/USM counts or compact non-empty list; event-wait already shipped (`0013`).
3. **Bitexact hybrid gather-norm** — still golden-fails when fused beyond stock sum/div.
4. ~~Tiny-N / multi-sg dual MMVQ packing~~ — **tried sgs=16; golden OK, pp regress, reverted** (`SHIP_20260730_mmvq_multisg.md`).
5. Bitexact multi-token dual/MMVQ only after oracle vs GEMM rows.
6. lm_head prune — high golden risk; largest byte stream.

## Target

Beat **tg128 120+ / pp512 1150+** on control+Q4 with golden — next real tip needs decode or clean prefill without host bubble.

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
