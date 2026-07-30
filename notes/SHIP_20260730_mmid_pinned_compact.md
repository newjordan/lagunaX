# Research note — mmid pinned counts + compact experts (2026-07-30)

## Status: **NOT SHIPPED** (reverted; golden OK, no tip beat)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| tip fused sigmoid+add | 1148.9 | 120.2 | **+9.09%** | OK |
| event-wait (pageable vector) | 1144.2 | 120.3 | **+9.04%** | OK |
| **pinned host_pool + compact active** | **1141.6** | **120.3** | **+8.99%** | **OK** |
| baseline pin | 1139 | 107.35 | 0 | — |

Formal: `results/20260730T060225Z/` · hit  
`[lx-control-mmid] device counting-sort+prefix+ev+pin n_tokens=512 k=8 n_experts=256`

## What tried

1. **Pinned** counts D2H via `ctx.host_pool()` / `sycl::malloc_host` (hope: async L0 transfer vs pageable `std::vector`).
2. **Compact active expert list** after wait — GEMM loop only over non-empty experts.

## Findings

- Golden OK (same GEMM regroup numerics).
- Decode flat (~120.3).
- Prefill **slightly worse** than pageable event-wait (−2–3 t/s formal); host_pool alloc churn and/or Laguna routing already filling most of 256 experts (compact loop ≠ big save).
- **Keep event-wait + pageable vector** (`SHIP_20260730_mmid_counts_event.md`).

## Tip unchanged

Scored tip remains fused sigmoid+add (`20260730T053204Z`, +9.09%).  
Default mmid: device counting-sort + prefix-sum + event counts wait.

## Next

1. Bitexact multi-token dual/MMVQ oracle (still golden-fail when defaulted).  
2. Fused router sum/div (still golden-fail).  
3. Decode-side levers (lm_head / tiny-N already multi-sg-closed).  
4. Avoid re-trying host_pool pinned counts without persistent buffer + A/B proving D2H was blocking.
