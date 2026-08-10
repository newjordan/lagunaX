# FINDING 2026-08-10 — upstream master prefill is nondeterministic on B70/Laguna; KLD gate cannot arbitrate master-series candidates

Chain of receipts (all Laguna XS 2.1 Q4_K_M, wikitext c512 x16, B70):

| run | candidate | reference | mean KLD | same-top |
|---|---|---|---|---|
| kld-20260810T155300Z | PR-A (master+nbatch) | old 7e1e28cae pin | 0.0531 | 91.5% |
| kld-20260810T155357Z | clean master dd1ea5243 | old pin | 0.0536 | 91.4% |
| kld-20260810T155600Z | PR-A | fresh master ref | 0.0276 | 94.5% |
| kld-20260810T155843Z | clean master | fresh master ref (own capture!) | 0.0291 | 94.2% |
| kld-20260810T155941Z | clean master (rerun) | same ref | — | 94.3% |

1. Master-vs-old-pin drift (0.054) is upstream's 185-commit numerics shift, not
   any candidate's. 2. **Master-vs-its-own-reference fails at ~94%** and two
   consecutive identical-binary runs differ — run-to-run nondeterministic
   prefill logits (suspect: atomics in the reworked MoE/GLU paths; the
   2026-07-28 base was deterministic, base-vs-base scored ~0/100%).

Consequences:
- The 0.010/99% KLD gate stays valid ONLY for the 7e1e28cae-based campaign
  builds. Master-series PR candidates cannot be KLD-gated until the
  nondeterminism source is isolated. Quality evidence for the PR series is
  test-backend-ops parity vs clean master (per-op NMSE tolerances).
- PR-A specifically: its nbatch constant is decode-only and unreachable in
  the perplexity geometry; its divergence (0.0276/94.5%) is inside master's
  own noise band. Exonerated.
- Upstream-relevant: run-to-run nondeterminism of SYCL prefill on Battlemage
  may itself be worth an upstream issue after isolating the op (candidate
  experiment: GGML_SYCL_DISABLE_* bisect / -ub 1 / expert-count sweep).
- Ops hardening added this session: quality-gate-kld.sh now asserts loader
  resolution per binary (4th LD_LIBRARY_PATH-trap near-miss); run_ppl already
  forced per-binary lib dirs (which is why captures were clean).

Master reference file (separate from the campaign pin, keep both):
/mnt/data2tb/lx-kld/Laguna-XS-2.1-Q4_K_M-c512-n16-master-dd1ea5243.kld
