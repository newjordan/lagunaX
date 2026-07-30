# Laguna tip proof suite — proof-20260730T175906Z

**Generated:** 2026-07-30 (post-reboot B70)  
**Binary:** `treebeard-base-control-latest/build-base-control` tip stack  
**Model:** Laguna-XS-2.1 Q4_K_M (`sha256=771a73e1…`)  
**Device:** Intel Arc Pro B70 · SYCL / Level-Zero  
**Track:** serial only (not multi-slot)

---

## Executive verdict

| Gate | Result | Proof status |
|------|--------|--------------|
| **Serial speed (pp512/tg128)** | **pp~3724 / tg~139 / +63%** | **PASS — real** |
| **Prefill ladder** | 512→8192 stays ~3600–3900 t/s | **PASS** |
| **Agent throughput (np=1)** | tool-turn tg **p50 ≈ 139 t/s** | **PASS (speed)** |
| **Golden greedy smoke** | matches recaptured golden | **PASS (self-consistent)** |
| **Perplexity (wikitext / tiny)** | tip **PPL ~5e5–1.6e6** vs fuses-off **~1.0** | **FAIL — tip breaks logprobs** |
| **Long-ctx needles (chat)** | server abort / 0/3 | **FAIL** |
| **Chat quality (tool/agent path)** | tip emits `__.__` garbage | **FAIL** |
| **Held-out ho-pack** | **0/46** (timeouts / server death) | **FAIL** |
| **Public Agent Bench 69** | not completed (blocked by agent quality) | **BLOCKED** |

**Bottom line:** the **+63% serial speed is real and reproducible**.  
It is **not** yet a shippable quality tip: tip fuses destroy distributional correctness (PPL) and the multi-token MoE dual-down path aborts / corrupts chat/agent use.

---

## 1. Formal serial (pp512 / tg128, 5 reps)

Pin: pp 1139.15 · tg 107.35

| run | pp512 | tg128 | score |
|-----|------:|------:|------:|
| prior peak `T144111Z` | 3734.9 | 139.4 | **+63.67%** |
| post-reboot #1 `T170411Z` | 3728.0 | 138.56 | +62.87% |
| post-reboot #2 `T170518Z` (LATEST) | 3723.9 | 138.89 | **+63.12%** |

Formula: `decode^0.75 * prefill^0.25` vs pin. Floors OK.

Artifacts: `results/20260730T170518Z/score.json`

---

## 2. Prefill / decode ladder (`llama-bench`, r=3)

| n_prompt | n_gen | t/s |
|---------:|------:|----:|
| 512 | 0 | **3734.6** |
| 2048 | 0 | **3898.5** |
| 4096 | 0 | **3741.4** |
| 8192 | 0 | **3601.4** |
| 0 | 128 | **139.8** |

Prefill holds ≥3.6k t/s out to 8k. Decode matches formal.

Artifact: `bench/ladder.json`

---

## 3. Perplexity

### Tip fuses ON (default stack)

| corpus | chunks | ctx | **PPL** |
|--------|-------:|----:|--------:|
| wikitext-2 test | 32 | 2048 | **1,653,299 ± 35,935** |
| repeated “quick brown fox” | 2 | 512 | **503,086 ± 49,875** |

### Major fuses OFF (same binary, kill switches)

| corpus | **PPL** |
|--------|--------:|
| repeated “quick brown fox” | **1.0005 ± 0.0001** |

Repeated-string PPL≈1 is expected (near-deterministic next token).  
Tip PPL in the **10⁵–10⁶** range means **logits are numerically wrecked** under the fuse stack — not a “chat model is OOD on wiki” story.

Kill set used for OFF arm:
`TOPK_MOE, ROUTER_*, MOE_DUAL_*, MOE_DOWN_*, DENSE_DUAL_*, RMS/ROPE/SOFTPLUS/MM_ADD/ADD_ADD, PACKED_REDUCE`.

Artifacts: `ppl/`, `/tmp/ppl-on.err`, `/tmp/ppl-alloff.err`

---

## 4. Long context

### Server needles (package harness, chat API, 80 paragraphs)

- Prefill ladder via server ok to ~5.5k tokens (~3.5k pp t/s)
- At ~48k chars / larger needle doc: **server aborted**
- Backtrace: `ggml_sycl_mul_mat_id_dual_down_multitoken` → `ggml_sycl_fuse_moe_dual_swiglu`
- Needles **0/3**, dossier **0/14** (connection refused after crash)

Artifact: `longctx-REPORT.md`, `logs/server.log`

### Speed at long prefill (still useful)

| path | prompt_n | pp t/s |
|------|--------:|-------:|
| agent depth_2048 | 3161 | 2695 |
| agent depth_8192 | 13955 | 3007 |
| ladder pp8192 | 8192 | 3601 |

Long **speed** is fine; long **quality/stability** is not under tip multitoken MoE dual-down.

---

## 5. Agent throughput (np=1 chat, max_tokens capped)

6 tool-style turns × 2 reps + 2 depth prompts:

| metric | value |
|--------|------:|
| requests ok | 8/8 |
| tool-turn tg **p50** | **139.05 t/s** |
| tool-turn tg mean | 129.4 t/s |
| tool-turn tg max | 139.18 t/s |
| depth_2048 pp | 2695 t/s |
| depth_8192 pp | 3007 t/s |

**Speed proof for agent decode matches formal tg128.**  
Content quality on chat path is **not** proven (empty/garbage content — see §6).

Artifact: `agent/throughput_summary.json`

---

## 6. Agent / chat quality

### Golden smoke (`/completion`, greedy, seed 42)

- Re-run twice post-reboot: **GOLDEN OK**
- Matches recaptured tip golden (instruction-style continuation of Fibonacci prompt)

### Chat completions (tip default)

Fibonacci prompt → pure garbage:

```
__.__
__.__
...
```

tg still **~138 t/s** while emitting nonsense.

### Held-out `ho-pack-v1.1`

| metric | value |
|--------|------:|
| score | **0.0%** (0/46) |
| outcomes | 23 fail (+1 stress fail) |
| wall | ~699 s |
| failure mode | timeouts / runaway gen / server death |

Runner does **not** pass `max_tokens`; Laguna tip path does not stop cleanly → multi-thousand-token runaways then crash.

### Public Agent Bench 69

**Not run to completion** this session:

- `tool-eval-bench 2.1.0 @ 8b3259b` reinstalled at `/tmp/tool-eval-bench`
- Blocked after held-out hang + chat quality failure (would not be a meaningful score)

---

## 7. What is proven vs not

### Proven

1. **+63% serial composite** on Laguna Q4_K_M / B70 is real and stable post-reboot.
2. Prefill scales: **~3.6–3.9k t/s** from 512–8192.
3. Single-stream **agent decode ~139 t/s** (same order as formal tg).
4. Golden greedy `/completion` is **self-consistent** under tip (recaptured golden).
5. Tip fuse stack is **active** (rms, rope, FA VEC, softplus×mul, dense dual, topk full-norm, dual+down, mm-add+add).

### Not proven / regressions

1. **PPL** — tip fuses destroy logprob quality vs same binary fuses-off.
2. **Multi-token MoE dual-down** — abort in `dual_down_multitoken` under long chat/prefill.
3. **Chat/agent quality** — garbage outputs and held-out 0%.
4. **Agent Bench 69** — not a valid ship gate until quality fixed.
5. **Long-ctx needle/dossier quality** — blocked by crash.

---

## 8. Recommended ship bar (next)

Before claiming a product tip:

1. **KL / PPL A/B** tip vs fuses-off on a fixed short corpus; tip must be near-neutral.
2. **Fix or kill** `MOE_DUAL_MULTITOKEN` / dual-down multi-token path (abort + PPL suspect).
3. Re-run **needles + dossier** with `max_tokens` hard caps and server `-n`.
4. **Held-out** only after runner gets `max_tokens` (or server default n_predict).
5. Then **Agent Bench 69** (tool-eval-bench pin already restored).

Suggested interim safe profile for agent/longctx:
```bash
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1   # if still aborting
# keep decode fuses that stay golden + PPL-safe after A/B
```

---

## Artifacts

```
results/proof-20260730T175906Z/
  PROOF.md                 ← this file
  meta/PINS.txt
  bench/ladder.{json,md}
  ppl/{ppl.log,ppl.err}
  longctx/ + longctx-REPORT.md
  agent/{throughput*.json,heldout.json}
  quality/{chat-tip.json,golden-*.log,srv-*.log}
  run.log
```

Harness: `scripts/proof-suite.sh`  
tool-eval-bench: `/tmp/tool-eval-bench` @ `8b3259be7411fe27c7610d0de64ae1d3b622b9ef`
