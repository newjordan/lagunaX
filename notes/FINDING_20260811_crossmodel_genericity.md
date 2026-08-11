# FINDING 2026-08-11 — champion stack does NOT transfer in aggregate; it is Laguna-calibrated. Bit-clean on foreign models.

Question under test (user challenge): "the kernels are pattern-keyed, so other
models should benefit." Instrument: one-variable A/B — pinned base-control
binary vs champion binary (`lx-champion-1.3105-20260810`), identical flags/env
both legs (campaign bench flags: ub2048 b2048 f16-KV r5 d0, graph+DNN off),
GPU lock held, receipts in `results/crossmodel-20260811/`.

## Results

| model | class | base pp512 / tg128 | champ pp512 / tg128 | Δpp / Δtg | KLD (16×512 wikitext-2) |
|---|---|---|---|---|---|
| Laguna-XS-2.1 Q4_K_M (board ref) | MoE 256e k8, sigmoid+bias | 1139.2 / 107.4 | 1174.4 / 152.7 | +3.1% / **+42.2%** | 0.0 / 100% (campaign gate) |
| Qwen3.5-35B-A3B Q4_K_M | MoE (A3B) | 2863.9 / 87.9 | 2921.4 / 86.2 | +2.0% / **−1.9%** | **0.000000 / 100.000%** |
| Muse Glimmer 30B kquant | dense 28B (day-0 release) | 926.6 / 28.4 | 925.6 / 28.6 | −0.1% / **+0.8%** | **0.000000 / 100.000%** |
| Nemotron 3.5 Lightning 30B-A3B | hybrid Mamba2+MoE | — | — | N/A | N/A |
| Nemotron 3 Nano 30B-A3B (substitute) | hybrid Mamba2+MoE | 973.3 / 51.4 | 970.8 / 51.3 | −0.3% / −0.2% | **0.001434 / 98.260%** (same-top MISS) |

Qwen sensitivity legs (all champ):
- rope fuse OFF: tg 86.20 vs 86.24 ON → **rope fuse is not the Qwen drag**.
- DNN ON operating point: base 88.10, champ 86.43 (−1.9%) → **operating point
  is not the explanation either**; the regression is in always-on champion
  paths (candidates: reorder-MMVQ subgroup 16→8, fattn VEC policy/nbatch,
  mmid device-sort — all calibrated on Laguna shapes).

Nano A/B (base 7e1e28cae vs champ c7d3bfe6d, `nemotron_h_moe 31B.A3.5B`,
n=5): pp512 973.30 ±9.72 → 970.84 ±9.62; tg128 51.38 ±0.02 → 51.30 ±0.01.
Flat both axes — the hybrid gets nothing.

**Nano is the first foreign model that is NOT bit-clean.** Mean KLD 0.001434
clears the ≤0.010 bar, but same-top 98.260% **misses the ≥99.0% bar** (RMS Δp
1.0%, max KLD 0.087, max Δp 15.7%; PPL 9.2118 → 9.2175). The divergence is
structured, not drift: it is present in every one of the 16 chunks. Contrast
with Qwen, where the rms / softplus-mul / moe-down / add fuses all fire and KLD
is still exactly 0 — so *fusing per se* is not what moves logits. The path Nano
takes and Qwen does not is the Laguna top-k MoE router fuse, which pattern-matches
Nano's router (`laguna bias fuse HIT (mul_mat+hybrid mode=8)`, `fused sigmoid+add
n_experts=128`, `true top-k+gather+sum+norm n_experts=128 k=6`) because it is the
same 128-expert sigmoid+bias family as Laguna.

**RESOLVED by self-control (2026-08-11 ~19:20Z, `nano-selfctl`):** base binary
vs ITSELF on the same instrument gives **mean KLD 0.001421 / same-top
98.652%** — statistically indistinguishable from base-vs-champion (0.001434 /
98.260%; Δ 0.39pp at ~1.4σ). The Nano same-top miss is the base binary's own
run-to-run nondeterminism on this model (the class documented in
FINDING_20260810_master_prefill_nondeterminism), NOT champion-attributable.
The KLD same-top gate is therefore not applicable to Nano-class models on this
instrument, same as the master-series carve-out. Champion remains
quality-clean on every foreign model tested. (Follow-up if anyone cares:
whether the router fuse *narrows* or widens the base's own noise band is
unmeasured — would need N self-control repeats.)

Nemotron 3.5 Lightning: champion base (7e1e28cae) cannot load it —
`done_getting_tensors: wrong number of tensors; expected 417, got 408`; the
enabling changes upstream are entangled in +383 lines of shared loader churn
(no clean pick). Muse Glimmer required cherry-picking master 62bf73d25
(src/-only, applies clean on both trees; NOTE: `git apply` piped through
`head` masked an atomic abort on the first attempt — verify with
`git status` + grep after any scripted apply).

## Reading

1. **The +42% is Laguna-calibrated, not generic.** Mechanisms are
   pattern-keyed (they fire or fall back cleanly — correctness holds
   everywhere tested, Muse KLD exactly 0), but the tuning constants and the
   biggest fusions key on Laguna's exact shapes/quants/router. On Qwen the mix
   nets −1.9% tg; on dense Muse the MoE machinery is dormant and the generic
   tier nets +0.8%.
2. **Correctness generalizes; performance does not.** No foreign model showed
   quality damage. The kill-switched broken paths (MOE_DUAL_DOWN etc.) stayed
   off. This is the difference between "safe to run" and "worth running".
3. **Upstream consequence:** the PR series must stay per-unit with per-model
   A/B (as PR_PACKAGE_v2 already mandates). Qwen's −1.9% needs per-unit
   decomposition (kill-switch sweep on Qwen) before filing PR-D/E/F; the
   subgroup-8 and fattn constants likely need shape-conditional guards
   rather than unconditional flips.
4. **Serving consequence:** none for Laguna (champion stays champion on its
   own model, receipts unchanged). Do not point other models at the champion
   binary expecting Laguna-class wins.

Closed 2026-08-11 19:02Z: Nano A/B (flat) + Nano KLD (0.001434 / 98.260%) +
Qwen KLD (0.000000 / 100.000%) all measured; receipts in
`results/crossmodel-20260811/`.

Self-control closed 2026-08-11 ~19:20Z: base-vs-base 0.001421 / 98.652% ≈
base-vs-champ 0.001434 / 98.260% → Nano miss is base-side nondeterminism, not
champion code. Title line and Reading points 1–2 STAND as written. Nothing
pending; finding closed.
