# lagunaX — absolute serial speed on Arc B70

Private campaign repo: **Laguna XS 2.1** serial (one stream) on Intel Arc Pro B70.

**Mission:** push serial tok/s to the hardware limit. mlx.fast-shaped score is the *yardstick* (decode-weighted speedup + golden), not multi-slot capacity.

- one stream (not multi-slot aggregate)
- score = `decode_speedup^0.75 * prefill_speedup^0.25` vs pinned B70 baseline
- hard floors 0.95 · greedy golden before claims
- absolute-limit waves bank every arm under `results/abs-serial-w*`

It is **not** a continuation of `~/Laguna-XS-2.1-B70-Turbo` (multi-slot capacity ~513).

See [TASK.md](TASK.md) for the full contract.  
**First kernel win:** [notes/SHIP_20260729_dual_swiglu.md](notes/SHIP_20260729_dual_swiglu.md) (+1.6% formal score).

## Quickstart

```bash
cd /home/frosty40/turbo/lx
source env.sh

# Pin baseline once (control binary + Q4_K_M + serial flags)
./scripts/bench-serial.sh --baseline

# Capture greedy golden (once)
./scripts/golden-smoke.sh --capture

# After a change: correctness then score
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "what changed"
cat results/LATEST_SCORE.json
```

## Layout

```text
env.sh                 # device, binary, model, serial flags
TASK.md                # problem statement + claim boundary
scripts/
  bench-serial.sh      # pp512 + tg128 → metrics/score.json
  score.py             # mlx.fast formula
  golden-smoke.sh      # greedy match gate
baseline/baseline.json # pinned B70 reference (do not silent-rewrite)
correctness/golden.json
results/<stamp>/       # metrics.json + score.json
```

## Claim boundary (short)

| Allowed | Not allowed |
|---|---|
| Solo pp512 / tg128 vs pinned baseline | Multi-slot aggregate as “the score” |
| Kernel/flag/quant wins that keep golden | Claiming parity with M5 106/4050 absolute |
| % increase from **B70 baseline** | Re-baselining to inflate scores |
