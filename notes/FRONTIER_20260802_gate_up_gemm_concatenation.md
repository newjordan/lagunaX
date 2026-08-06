# FRONTIER: Gate/Up GEMM Concatenation — Exact 2:1 Call Ratio Proves Separate Gate & Up Matmuls

## Direction (distinct from all 9 tried)
Fuse the two independent gate_proj and up_proj matmuls into a single concatenated
GEMM (weight stacked along N → 2N output columns), eliminating one entire launch
per expert-step. This is an **input-side GEMM fusion**, NOT launch-coalescing
across experts (dir 1), NOT output-precision (dir 4), NOT epilogue post-op fusion
(dir 5), NOT kernel selection (dir 6), NOT descriptor rank (dir 8), NOT weight
dequant (dir 9).

## Smoking-gun evidence: exact 2:1 ratio per N value

For EVERY N value in the trace, M=512×K=2048 (gate/up) calls are exactly 2× the
M=2048×K=512 (down) calls:

| N  | gate/up (M=512,K=2048) | down (M=2048,K=512) | ratio |
|----|------------------------|---------------------|-------|
| 9  | 12,220                 | 6,110               | 2.000 |
| 10 | 10,200                 | 5,100               | 2.000 |
| 11 | 8,674                  | 4,337               | 2.000 |
| 12 | 7,072                  | 3,536               | 2.000 |
| 13 | 6,130                  | 3,065               | 2.000 |
| 14 | 5,414                  | 2,707               | 2.000 |

Source: `results/ktrace-post-brownout-20260731/trace.log`, awk on field 12
(problem_desc) grouped by M and N.

This proves gate and up issue as **two separate oneDNN matmul primitives** per
expert-step, while down issues one. The `[lx-control-moe-dual] fuse hit
(gate+up+swiglu)` fusion is at the **epilogue/swiglu level only** — the actual
GEMMs that produce gate_out and up_out are separate calls.

## Totals
- gate/up family: ~121,666 calls (64% of 190,211 matmul execs)
- down family: ~60,833 calls (32%)
- dense: ~7,712 calls (4%)
- gate/up is exactly 2× down → gate = up = ~60,833 calls each

## Savings estimate (quality-neutral, identical math)
Concatenating gate+up weights → single GEMM with 2N columns:
- Eliminates ~60,833 primitive exec calls + 60,833 create:cache_hit lookups
- Each surviving call has 2N columns; from the N-tile ladder (finding #9):
  N=9-16 → 17.61µs, N=17-32 → 18.89µs, so 2N stays in same/next tile
- Per expert-step: old = 2 × 17.61µs = 35.22µs; new = 18.89µs → 46% savings
- Conservative total: 60,833 × (17.6µs launch + 1.18µs cache_hit) ≈ 1,142 ms
  = 29.8% of the 3,828 ms matmul exec total
- Even at 50% efficiency: ~570 ms = 15% of matmul time

## Implementation path
1. At model load: pre-concatenate gate_proj and up_proj weights for each expert
   along output dim → [intermediate × 2×hidden] or equivalently stack to produce
   2N output columns per expert-step. Zero runtime cost (static layout change).
2. MoE forward: issue single matmul → 2N output, then split in the existing
   `[lx-control-moe-dual]` swiglu kernel (already reads both gate+up outputs).
3. The swiglu kernel already receives both outputs — it just needs to read them
   from contiguous halves of one buffer instead of two separate buffers.

## Why this is quality-neutral
silu(gate(x)) * up(x) where gate(x) = W_gate^T @ x, up(x) = W_up^T @ x.
Concatenated: [W_gate; W_up]^T @ x = [gate(x); up(x)] — bit-exact same values,
just computed in one GEMM instead of two. The split+silu+mul is unchanged.
