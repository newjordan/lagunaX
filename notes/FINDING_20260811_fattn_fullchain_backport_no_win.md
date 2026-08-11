# FINDING 2026-08-11 — full fattn-chain backport: quality-clean, zero prefill win, depth-decode regression. NOT the serving vehicle.

Branch `lx/serve-fattn-backport` (worktree `turbo/worktrees/lx-serve-fattn-backport`)
= champion tag `lx-champion-1.3105-20260810` + the complete fattn sequence from
7e1e28cae..dd1ea5243: 9d9a6d29f (oneMKL XMX FA) + 66fa168a5 (oneDNN SDPA
non-F16 KV) + eef5f3e34 (dispatch restructure / arc770 fix — the commit v1
skipped). One conflict, fattn-vec.hpp: upstream's `if constexpr` structure
kept, champion's nbatch=256 single-block launch re-cut into it.

## Receipts (results/fattn-backport-20260811T18{1346,1820}Z)

Gates: golden-smoke OK; KLD gate PASS (same-top 100.000%, ln(PPL/base) +0.016).

| leg | pp512 / tg128 (d0) | real 23K prefill | real 23K decode |
|---|---|---|---|
| champion reference | 1174 / 152.5 | 308.8 | 92.4 |
| backport DNN off | 1162 / 153.2 | 307.4 | **81.4 (−12%)** |
| backport DNN on | 1170 / 153.0 | 307.3 | **54.0 (−41%)** |

(real-text instrument: llama-cli -f wikitext-23k, c32768 b4096 ub2048, n128,
--jinja --chat-template-file required — embedded template still doesn't parse,
and --no-jinja is NOT forwarded to the embedded server by this fork's llama-cli.)

## Reading

1. **The 4.84x master prefill advantage is NOT carried by the fattn chain.**
   Full chain, MKL FA default-on: real-text prefill unchanged (307 vs 309).
   The earlier attribution ("master's XMX prompt paths attack exactly this")
   is falsified for Laguna; master's win lives in the other ~182 commits
   (MoE/GLU prefill work) and/or the combination.
2. **Likely mechanism for the attention no-op: Laguna's SWA geometry.** The
   XMX prompt paths gate on K>=1024; Laguna runs 3:1 SWA:global with
   window 512, so ~75% of attention layers can never take the fast path.
   The chrono ledger already placed 69.9% of real-text wall in the MoE
   expert loop — attention was never the dominant term.
3. **Depth-decode regression (92.4 → 81.4 DNN-off) despite d0 parity
   (152.5 → 153.2).** The chain's dispatch changes interfere at large K even
   when the new paths shouldn't fire (Q=1 < 32 gate). Un-diagnosed; branch
   preserved for autopsy. DNN-on additionally costs 27 t/s at depth
   (54.0) — the decode-era DNN-off verdict extends to this tree at all depths.
4. **Serving verdict: canonical champion stays.** serve-laguna.sh unchanged.
5. **The real serving-prefill attack is the expert loop** (per-expert fixed
   costs at Zipf-skewed routing: submission coalescing, expert batching
   across layers, ids-sync removal — FINDING_20260810_serving_prefill_routing_skew
   implication 2). That is the ~6x headroom, and it is untouched by attention
   work.

## Corrections this supersedes

- FINDING_20260810_realtext_prefill_device_bound_master_4x consequence (a)
  ("BACKPORT ... bounded") — the bounded backport is DONE and does not win;
  vehicle (b) (PR-series all-stack on master) is the remaining path, plus the
  expert-loop attack which works on either base.
