# FRONTIER: fpmath Accumulation-Mode Asymmetry (f16 expert GEMM vs strict attention)

## Direction (distinct from all 13 tried)
The oneDNN **accumulation/intermediate-rounding mode** (`attr-fpmath`) is the axis
no prior direction examined. It is NOT:
- output precision (dir 4: dst:f32)
- weight/input precision (dir 9: wei:f16 dequant)
- kernel selection (dir 6: jit:gemm:any)
- N-tile (dir 3), launch count (dir 1), cache tax (dir 2)

It controls the internal multiply-accumulate rounding: `f16` = XMX f16 dot-product
with f16 intermediate rounding; `strict` = higher-precision (f32) accumulation.

## Verified findings
1. ALL 190,211 matmul execs carry `attr-fpmath:f16` (field 10) — the expert GEMM
   path (96.8% of decode compute) uses f16 accumulation.
2. ALL 1,280 graph SDP execs carry `fpm:strict` — the attention path (1.9%) uses
   full-precision accumulation. Two modes coexist, set per-path.
3. The gemm create:cache_miss lines carry only `attr-fpmath:f16` (no
   `attr-scratchpad:user`), while matmul exec lines carry
   `attr-scratchpad:user attr-fpmath:f16` — the fpmath mode is baked into the
   JIT-compiled kernel at the gemm level; the scratchpad:user is added by the
   matmul wrapper. So the 11 compiled kernel binaries are f16-XMX-specific.

## Key hypothesis
Because the expert GEMM stream is BW-bound at ALL N values (finding #34:
BW-time > compute-time even at N=256), switching `attr-fpmath:f16` →
`attr-fpmath:strict` on the expert matmul may be SPEED-NEUTRAL — the XMX unit is
data-starved regardless of accumulation mode — while improving numerical quality.
This is a potential "free quality upgrade" (or conversely, confirmation that the
current f16 mode has no quality cost that strict would fix) that needs a benchmark.
