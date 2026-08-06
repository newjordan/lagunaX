# QUEST: Mount Doom — Laguna serial absolute limit on B70

**Status:** PAUSED afternoon 2026-07-31 — handoff `notes/HANDOFF_20260731_afternoon.md`  
**Mission:** max serial pp512 / tg128. Kernel path. No multi-slot cosplay.  
**2-day goal:** formal score **≥ 1.250** by **2026-08-02 14:00 UTC** (stretch 1.300).  
**Loop:** paused; relaunch `bash scripts/quest-2day-launch.sh` when ready

## Live infrastructure

| Piece | Path |
|-------|------|
| Harness / score | `/home/frosty40/turbo/lx` |
| Kernel worktree | `/home/frosty40/turbo/worktrees/lx-serial-kernel` (`lx/serial-kernel-max`) |
| Build | `.../lx-serial-kernel/build-serial-kernel` |
| Baseline | `baseline/baseline.json` · pp≈1139 · tg≈107.35 |
| Quest daemon | `scripts/quest-launch.sh` → `results/quest-mount-doom/` |
| Kernel traces | `results/kernel-trace-20260729T210618Z/` |

## Proven so far

1. **Env/flag plateau** — wave1 ±0.4% (done)
2. **BW-bound decode** — 86 W / 37% of 230 W
3. **Kernel smoking gun** — MoE expert matmuls dominate:
   - `1x512x2048 : 1x2048xN` (N≈9–29) gate/up
   - `1x2048x512 : 1x512xN` down
   - thousands of tiny-N calls
4. **Control beats package solo** — post-reboot A/B confirms:
   - ctrl **1136 / 107.6** · pkg **818 / 105.0** · serial_pkg **817 / 104.8**
5. **Package dual-SwiGLU tree ≠ serial win** — wholesale port loses ~28% pp
6. **GPU wedge cleared** (reboot 2026-07-29 ~17:41); resume A/B + quest up
7. **FIRST KERNEL WIN — Q4_K MoE dual-SwiGLU on control**
   - Package dual was type-rejecting Q4_K (Q5/Q6 only)
   - Control dual fuse: **pp~1144 / tg~110.1 / score ~+2%** vs baseline; golden OK
   - Code in `treebeard-base-control-latest` (default ON; disable via env)
8. **Frontier map** — `notes/FRONTIER_20260729.md` (superseded live tip below)
   - Full **vocab=100352 lm_head** every token (~116 MB Q4 traffic) — still open
   - Prefill multi-token dual golden still hard
9. **LIVE TIP (2026-07-31, post-brownout reconfirm)** — **quality-safe** tip + **decode-only mm-add**
   - Formal: **pp~1183 / tg~139.3 / score 1.227 (+22.74%)** vs pin (`results/20260731T141436Z/`)
   - Golden OK · wikitext PPL ~12.6 (not 1e5+)
   - Binary: `build-mmadd-decode` (patched tip lib, `ne11==1` only for mm-add)
   - Still **killed**: `MOE_DUAL_DOWN`, `MOE_DUAL_MULTITOKEN` (PPL/neg-logprob breaks)
   - Invalid claim: full-default +63% (prefill dual multitoken wrecked quality)
   - See `notes/SHIP_20260731_mmadd_decode_only.md` · `notes/SHIP_20260730_quality_safe_tip.md`
10. **Post-brownout (2026-07-31 ~reboot)** — sycl B70 up, lock free, golden OK, formal rebench OK

## Loop (autonomous)

```text
kernel-trace hot shapes
  → edit mmvq / fusion (lx-serial-kernel)
  → rebuild llama-bench
  → golden-smoke + bench-serial
  → keep if tg/pp win + floors
  → quest daemon tracks champion
```

## Commands

```bash
# relaunch continuous rebench
bash /home/frosty40/turbo/lx/scripts/quest-launch.sh
tail -f /home/frosty40/turbo/lx/results/quest-mount-doom/quest.log

# score a candidate binary
export LX_BIN=/home/frosty40/turbo/worktrees/lx-serial-kernel/build-serial-kernel/bin
source /home/frosty40/turbo/lx/env.sh
./scripts/bench-serial.sh --note "kernel port dual-moe"

# re-trace after a win
bash ~/.claude/skills/b70-kernel-trace/ktrace.sh --mode onednn \
  --model "$LX_MODEL" --bin "$LX_BIN" --out results/ktrace-next -- \
  -p 0 -n 128 -r 1 -ub 4096 -b 8192 -fa on
```

## Do not

- Re-pin baseline to invent wins
- Claim multi-slot 513 as serial score
- Ship FA-off or graph-on (measured losses)
- **Run two GPU jobs at once** (bench+PPL, dual benches, …) — wedges xe; see `notes/B70_NO_CONCURRENT_GPU.md`
