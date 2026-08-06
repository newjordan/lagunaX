# lmhead-prefetch A/B ledger — GGML_SYCL_LMHEAD_PREFETCH (default OFF)

Same-window sandwich (CTRL-a / candidate / CTRL-b), official geometry
(pp512/tg128, -ngl 99 -t 16 -sm layer -mg 0 -ts 0 --device auto -b 2048 -ub 2048
-ctk f16 -ctv f16 -r 5, ONEAPI_DEVICE_SELECTOR=level_zero:gpu, ZE_AFFINITY_MASK=0,
GGML_SYCL_DISABLE_GRAPH=1, GGML_SYCL_DISABLE_DNN=1). Decision bound: ±0.68%
between-run card-state drift; positive rows beyond it only may promote (with the
full proof-suite gate). tg in tok/s.

| stamp | ctrl-a tg | cand tg | ctrl-b tg | delta vs ctrl mean | ctrl mean |
|---|---|---|---|---|---|

Payload: results/lmhead-prefetch/q6k.patch (marker-anchored, git-apply-checked,
abort rc=20 on drift; gate never on by default).
| 20260806T153609Z | 138.204 | 138.193 | 138.267 | -0.031% | 138.236 |
