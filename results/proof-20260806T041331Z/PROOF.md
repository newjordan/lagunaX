# Laguna tip proof suite — proof-20260806T041331Z

Generated: 2026-08-06 05:01 UTC

Serial track only (one stream). Model: Laguna-XS-2.1 Q4_K_M. Device: Arc Pro B70 SYCL.

## 0. Pins

```
stamp=20260806T041331Z
binary=results/src-repro-20260806T035656Z/bin/llama-bench
-rwxrwxr-x 1 frosty40 frosty40 564624 Aug  5 22:59 results/src-repro-20260806T035656Z/bin/llama-bench
-rwxrwxr-x 1 frosty40 frosty40 570024 Aug  5 23:12 results/src-repro-20260806T035656Z/bin/llama-perplexity
-rwxrwxr-x 1 frosty40 frosty40 662784 Aug  5 23:12 results/src-repro-20260806T035656Z/bin/llama-server
8fc1ed100ba41b2361056374def3b911  results/src-repro-20260806T035656Z/bin/libggml-sycl.so.0.17.0
model=/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf
model_sha256=771a73e1249b9bc08e17d3fca59f5c49b7b9c8a6a47b5a6ac82f95c6e76923c4
wiki=/home/frosty40/data/wikitext-2-raw/wiki.test.raw
srv_ctx=16384
template=/home/frosty40/turbo/worktrees/treebeard-pr-private-latest/models/templates/poolside-Laguna-XS-2.1.jinja
INFO: Output filtered by ONEAPI_DEVICE_SELECTOR environment variable, which is set to level_zero:gpu.
To see device ids, use the --ignore-device-selectors CLI option.

[level_zero:gpu] Intel(R) oneAPI Unified Runtime over Level-Zero V2, Intel(R) Arc(TM) Pro B70 Graphics 20.2.0 [1.15.38308+1]

```

## 1. Formal serial (pp512 / tg128)

| metric | value |
|--------|------:|
| pp512 | 1173.6 |
| tg128 | 137.8 |
| score vs pin | **+21.51%** |
| decode speedup | 1.284x |
| prefill speedup | 1.030x |
| floors_ok | True |
| candidate | `/home/frosty40/turbo/lx/results/20260806T035917Z/metrics.json` |

## 2. Prefill / decode ladder

# Prefill / decode ladder

| test | n_prompt | n_gen | t/s |
|------|--------:|------:|----:|
|  | 512 | 0 | 1141.66 |
|  | 2048 | 0 | 1937.99 |
|  | 4096 | 0 | 1904.46 |
|  | 8192 | 0 | 1860.94 |
|  | 0 | 128 | 136.64 |


## 3. Perplexity (wikitext-2)

```
perplexity: tokenizing the input ..
perplexity: tokenization took 433.356 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 5.33 seconds per pass - ETA 2.34.354.143 I Final estimate: PPL = 13.7703 +/- 0.27263
perplexity: tokenizing the input ..
perplexity: tokenization took 433.356 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 5.33 seconds per pass - ETA 2.34.354.143 I Final estimate: PPL = 13.7703 +/- 0.27263
Final estimate: PPL = 13.7703 +/- 0.27263
Final estimate: PPL = 13.7703 +/- 0.27263
```

## 4. Long-context quality

# Long-context package eval

- base: `http://127.0.0.1:18911`
- model: `laguna-tip-proof`

## Prefill / short decode ladder

| chars | prompt_n | pp tok/s | tg tok/s | wall_s | ok |
|------:|---------:|---------:|---------:|-------:|:--:|
| 2000 | 359 | 303.7541131606758 | 43.89225548584612 | 1.9 | yes |
| 8000 | 1402 | 325.5616519026083 | 120.76793309456507 | 4.6 | yes |
| 16000 | 2793 | 321.58213343551506 | 110.65695651873214 | 9.0 | yes |
| 32000 | 5577 | 321.2093182379736 | 105.75156314029266 | 17.7 | yes |
| 48000 | 8359 | 318.3243715826018 | 100.37357791028484 | 26.6 | yes |

## Needle-in-haystack

- doc_chars: 34044
- paragraphs: 80
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
- PASS: What major disruption to Broadway does the 2020s section mention? → 'The 2020s section mentions "pandemic shutdowns" as a major disruption to Broadway.\n'

## Verdict

**STRONG — long-context speed path + retrieval/comprehension held**



## 5. Agent throughput (np=1 chat)

| metric | value |
|--------|------:|
| tool-turn tg mean | 134.58369657019307 |
| tool-turn tg p50 | 134.63131053207397 |
| tool-turn tg min/max | 134.36627406520964 / 134.65500965728899 |
| requests ok | 8/8 |

Depth prefill samples:
```json
{
  "n": 2,
  "mean": 316.4009694489887,
  "values": [
    [
      "depth_2048",
      3161,
      316.3506129555834,
      107.46116286410864
    ],
    [
      "depth_8192",
      13955,
      316.45132594239396,
      90.15963891064617
    ]
  ]
}
```

## 6. Held-out pack

- score_pct: **13.0**
- points: 6/46
- outcomes: `{"fail": 20, "pass": 3}`

## Artifacts

Directory: `/home/frosty40/turbo/lx/results/proof-20260806T041331Z`

- `bench/` prefill ladder
- `ppl/` perplexity
- `longctx/` needles + dossier
- `agent/` throughput + held-out + public69
- `logs/server.log`

