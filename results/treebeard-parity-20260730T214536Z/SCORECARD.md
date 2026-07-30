# Laguna quality-safe — Treebeard-parity scorecard

Generated: 2026-07-30 22:02 UTC

Stack: Laguna-XS-2.1 Q4_K_M · control tip binary · **quality-safe kills**
(`MUL_MAT_ADD` + `MOE_DUAL_DOWN` + `MOE_DUAL_MULTITOKEN` off via env.sh).

Same **instruments** as Treebeard scorecard; **not** matched Qwen product A/B.

## Scoreboard

| Gate | Result | Treebeard ref (Qwen) |
|------|--------|----------------------|
| Formal serial vs pin | **+20.3%** (pp 1174 / tg 136.0) | n/a (Laguna pin) |
| Prefill ladder 512→16k | 1169→1862 t/s | n/a |
| Needles (100 para) | **3/3** | 3/3 |
| Dossier QA | **12/14** | 14/14 |
| Single-agent tg p50 | **135.5** tok/s · 10/10 content | 77.1 / 88.9 (Qwen) |
| Held-out ho-pack | **30.4%** (14/46) | 91.3% |
| Agent Bench 69 | **not valid @ Treebeard pin** (see §6) | 91/100 |
| Multi-slot np=4 | 8/8 · tg p50/req **36.3** | np12 fleet different claim |

## 1. Formal serial

- pp512 **1173.9** · tg128 **136.0** · **+20.32%** vs pin
- decode× 1.267 · prefill× 1.031

## 2. Prefill ladder

# Prefill / decode ladder

| n_prompt | n_gen | t/s |
|--------:|------:|----:|
| 512 | 0 | 1169.07 |
| 2048 | 0 | 2035.25 |
| 4096 | 0 | 2009.40 |
| 8192 | 0 | 1955.38 |
| 16384 | 0 | 1862.32 |
| 0 | 128 | 135.11 |


## 3. Long-context

# Long-context package eval

- base: `http://127.0.0.1:18930`
- model: `laguna-quality-safe`

## Prefill / short decode ladder

| chars | prompt_n | pp tok/s | tg tok/s | wall_s | ok |
|------:|---------:|---------:|---------:|-------:|:--:|
| 2000 | 359 | 417.4306966209043 | 43.846668201300055 | 1.6 | yes |
| 8000 | 1402 | 1780.0416951278658 | 121.21166207703757 | 1.1 | yes |
| 16000 | 2793 | 2165.7173075938904 | 110.93854004881297 | 1.6 | yes |
| 32000 | 5577 | 2404.0637600643495 | 106.09833326149591 | 2.6 | yes |
| 48000 | 8359 | 2354.7911647581936 | 100.79279835455756 | 3.9 | yes |

## Needle-in-haystack

- doc_chars: 42739
- paragraphs: 100
- score: **3/3**

- depth 0.1: PASS — Carousel → 'TB-CAR-1945-X\n'
- depth 0.5: PASS — Company → 'TB-COM-1970-Q\n'
- depth 0.9: PASS — Hadestown → 'TB-HAD-2019-Z\n'

## Broadway long dossier QA

- doc_chars: 61092
- score: **12/14**

- PASS: What year did Oklahoma! open? → 'Oklahoma! opened in 1943.\n'
- PASS: Who were the writers associated with Oklahoma! in this dossier? → 'Rodgers and Hammerstein were the writers associated with Oklahoma! according to the dossier.\n'
- PASS: In what year did West Side Story premiere on Broadway according to the dossier? → 'West Side Story premiered on Broadway in 1957.\n'
- PASS: Who wrote the lyrics for West Side Story per the dossier? → 'Stephen Sondheim wrote the lyrics for West Side Story per the dossier.\n'
- PASS: Which 1968 musical is linked to counterculture rock energy in the dossier? → 'Hair (1968)\n'
- PASS: Which 1975 show about dancers is mentioned as a long-running phenomenon? → 'A Chorus Line\n'
- PASS: Which composer is tied to mega-musicals like Phantom in the dossier? → 'Andrew Lloyd Webber\n'
- FAIL: What year does the dossier give for Phantom on Broadway? → 'The dossier does not provide a specific year for The Phantom of the Opera on Broadway.\n'
- FAIL: Who wrote Rent according to the dossier? → 'The dossier does not specify who wrote Rent.\n'
- PASS: What year did Rent open per the dossier? → 'Rent opened in 1996 per the dossier.\n'
- PASS: Which 2003 fantasy blockbuster is named in the dossier? → 'The 2003 fantasy blockbuster named in the dossier is **Wicked**.\n'
- PASS: What year did Hamilton open on Broadway per the dossier? → 'Hamilton opened on Broadway in 2015.\n'
- PASS: Who is credited with book, music, and lyrics for Hamilton here? → 'Lin-Manuel Miranda is credited with book, music, and lyrics for Hamilton.\n'
- PASS: What major disruption to Broadway does the 2020s section mention? → 'The 2020s section mentions "pandemic shutdowns" as the major disruption to Broadway.\n'

## Verdict

**STRONG — long-context speed path + retrieval/comprehension held**



## 4. Single-agent sequential

- tg p50 **135.53828413757924** · wall p50 **0.8717168569564819** s · content ok **10/10**

## 5. Held-out

- **30.4%** · 14/46 · outcomes `{"pass": 6, "fail": 15, "partial": 2}`
- Tool-use quality is weaker than Treebeard/Qwen; model capability, not only kernels.

## 6. Public Agent Bench 69

**Not a valid score.** Treebeard instrument uses c=262144; Laguna Q4 19G on B70:
- c=32768: tool schemas often **~34–37k tokens** → exceed_context (many automatic fails)
- c=65536: **server abort/OOM** after load
- Full 69 run at 64k produced 0/100 after server death (not a model score)

Partial 32k progress before interrupt (first attempt): a few TC pass (e.g. TC-04/05/10/12/13) and several context-size fails.
Laguna tool-agent gate needs either larger GPU memory, smaller tool schemas, or q-KV experiments — **not measured at Treebeard pin.**

Treebeard reference: **91/100** on Qwen Q5 @ c=262144 — different model + ctx.

## 7. Multi-slot appendix (np=4, c=16k)

- ok 8/8 · tg p50/request 36.3434169635532
- Not Treebeard np12 fleet claim.

## Verdict

**Quality-safe Laguna tip is real for serial speed + long-context retrieval/chat.**
- Bank: **~+20% formal**, **tg~136**, needles **3/3**, dossier **12/14**, chat content OK.
- Agent tool suites (held-out 30%, Agent69) show Laguna is not a Treebeard-level tool agent on this pin; Agent69 cannot even fit at Treebeard 262k on this GPU.

Artifacts: `/home/frosty40/turbo/lx/results/treebeard-parity-20260730T214536Z`

