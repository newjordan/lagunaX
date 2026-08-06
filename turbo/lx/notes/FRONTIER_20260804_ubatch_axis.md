# FRONTIER 20260804 — Ubatch-axis / prefill M-dimension regime

## The angle
All 40 prior findings decompose the expert matmul stream from a single trace
(`ktrace-post-brownout-20260731`), which is a **decode** capture. No prior
direction (1–16) examined the **n_ubatch / n_batch axis** as an optimization
lever, despite the fact that ubatch *directly controls M* (the token-batch
dimension of every GEMM). The M-class distributions that findings #29, #32,
#36 treat as structural are actually functions of ubatch.

## Evidence opened this iteration

### 1. The prefill trace operates in a completely different M regime
The `prefill-onednn` trace (captured 2026-07-29) runs at **n_batch=8192,
n_ubatch=4096** — 2× the ubatch of the scored champion (env.sh UBATCH=2048).
Its first `cache_miss` is a **M=6144×K=2048×N=256** GEMM (6.78 ms JIT, 0.328 ms
exec/call), followed by **M=1024×K=2048×N=256** (4.08 ms JIT). These M values
are 6–12× the decode gate/up M=512 and are absent from the decode budget.

### 2. M=6144 is prefill-exclusive
Finding #36 lists M=6144 as only 23.18 ms (0.6%) of **decode** compute. But in
the prefill trace it is the FIRST and largest matmul shape. The prefill and
decode matmul streams have **completely disjoint M-class distributions** at the
top end — yet all 40 findings and all 16 directions targeted only decode.

### 3. Prefill is even more primitive-dominated than decode
The prefill trace has **240 graph execs for 48,201 matmul execs** (201:1 ratio),
vs decode's 148.7:1 (finding #19). The graph-compiler path has even less
coverage in prefill.

### 4. Both prefill M-tiers use the IDENTICAL kernel as decode
M=6144 and M=1024 prefill calls both resolve to `jit:gemm:any` with
`attr-fpmath:f16` — the same micro-kernel and accumulation mode as all 190K
decode calls. No shape-specialized kernel is selected for the larger prefill M.

### 5. Two decode traces have 25% different matmul call counts
`decode-onednn` (2026-07-29) has 141,855 matmul execs vs
`ktrace-post-brownout`'s 190,211 — a 25% gap between two decode runs. This
means the per-call statistics (especially finding #34's "20 distinct N values"
power-law) are run-specific, not structural constants.

## Optimization lever — HONEST REASSESSMENT (corrected)
**Original hypothesis (ubatch increase) is a DEAD END for the scored
benchmarks.** Reasoning:
- `tg128` (decode): llama-bench generates tokens one-at-a-time (batch=1), so
  `n_ubatch` does not enter the decode path at all. Decode M is NOT ubatch.
- `pp512` (prefill): 512 prompt tokens < UBATCH=2048, so they already fit in a
  single ubatch window. Raising ubatch to 4096 changes nothing for pp512.
- Therefore the ubatch axis has ZERO leverage on the scored pp512/tg128 pair.
  It would only matter for pp2048+ prompts, which are not in the score formula.

The genuinely-new, file-evidenced findings that survive are the **trace-structure
observations** below (disjoint prefill M-regime, 201:1 primitive ratio, 25%
inter-run call-count variance) — none of which yield an actionable speedup for
the current pp512/tg128 score without source-level investigation of the decode
M dimension (whether M=512 is token-derived or architecture-derived).

## Status
Evidence: file-based only. Needs a benchmark at UBATCH=4098 with golden PPL
check to confirm no quality regression and measure the decode speedup.
