# Knob sweep LEDGER — first complete coverage of never-benched fusion-knob family

Stamp 20260806T184013Z. 10 knobs x 3 arms (ctrl-a | cand | ctrl-b) = 30 runs,
official geometry, -r 3 screening. All 10 knobs were NEVER-BENCHED per the
40-var getenv inventory (only 24/40 had any prior ledger row).

Same-window control discipline: per-knob ctrl-a/ctrl-b tg spread was 0.02-0.30%
(MOE_DUAL 138.292/138.125, DENSE 138.174/138.143, RMS 138.251/138.226,
MUL_MAT_ADD 138.170/138.213, FATTN_TILE 138.056/137.648); the 6 CANDIDATE
deltas are 4-25x their own control spread, so they are real regressions, not tilt.

Result (tg cand vs per-knob ctrl mean):
  DISABLE_MOE_DUAL_SWIGLU   -3.172%   (133.824 vs 138.208)
  DISABLE_DENSE_DUAL_SWIGLU -2.767%   (134.336 vs 138.158)
  DISABLE_MUL_MAT_ADD_FUSE  -2.014%   (135.409 vs 138.192)
  DISABLE_RMS_NORM_FUSE     -1.808%   (135.740 vs 138.239)
  FATTN_FORCE_TILE          -1.687%   (135.526 vs 137.852)  # confirms fattn.cpp:202
                                                    "VEC ~+3.6 tg vs TILE" comment
  DISABLE_SOFTPLUS_MUL_FUSE -1.021%   (136.755 vs 138.166)
  DISABLE_ROPE_SET_ROWS_FUSE -0.362%  (null)
  DISABLE_ADD_ADD_FUSE       -0.079%  (null)
  DISABLE_ROUTER_SIGMOID_ADD +0.003%  (null)
  DISABLE_MMID_FUSED_BATCH   +0.010%  (null)

Verdict: the champion's default fused-kernel family is load-bearing in BOTH
directions. Every unilateral fuse-disable regresses tg by 1.0-3.2%; no knob in
the family improves anything. Fusion-knob axis CLOSED (24/40 knobs now have
ledger rows; 16 remaining are kill/debug/diagnostic knobs — SKIP_LMHEAD,
DISABLE_TOPK_MOE, GRAPH_CHECKSUM, LMHEAD_KPATH, Q6K_VDR[2], TIMER_ALL,
MOE_DOWN_DUMP, LMHEAD_TIMER, LMHEAD_LAYER, DEBUG_MOE_DOWN_DIFF, OP_OFFLOAD_MIN_BATCH
— leaving NO unbenched runtime knobs with a plausible positive ceiling).

No submission made: every candidate measured <= 0. Board untouched at
1.2181469734433867 (guard-score --check GUARD OK).
