# lm_head kernel-path A/B ledger (GGML_SYCL_LMHEAD_KPATH on fused lm_head group)

Window 1 (20260806T114031Z, 7 arms, interrupted + resumed):
  ctrl-a   tg 135.9008   (default selection)
  kp-dmmv  tg 136.9129   (+0.74% vs ctrl-a)
  ctrl-b   tg 135.9216   (default)
  kp-mmvq  tg 135.8853   (-0.03% vs ctrl-b, null)
  ctrl-c   tg 135.4769   (default, drift low)
  kp-mmq   tg 110.8228   (-18.5% vs ctrl-d, catastrophic)
  ctrl-d   tg 135.9502   (default)

Window 2 (20260806T123045Z, 3-arm confirm, same binary rebuilt):
  ctrl-e   tg 136.0261   (default)
  kp-dmmv2 tg 135.7129   (-0.23% vs ctrl-e -> NOT reproduced)
  ctrl-f   tg 135.9039   (default)

Conclusions:
  - mmq forced on the single-row fused lm_head group is catastrophic (-18.5%) —
    mmq is the multi-token (n>1) quantized matmul; for n=1 decode it is wrong. The
    shipped default never selects it for this group. Negative control confirmed.
  - mmvq forced = null (-0.03%).
  - dmmv showed +0.74% in window 1 bracketed by two ctrls at 135.90/135.92, but the
    same binary re-benched in window 2 came back -0.23% vs ctrl-e and below ctrl-f:
    NOT reproducible -> null. Likely the window-1 dmmv run hit a good ambient moment.
  - Shipped default kernel selection for the fused lm_head is optimal. The
    kernel-path axis (dmmv|mmvq|mmq) is exhausted: null-or-worse everywhere.
