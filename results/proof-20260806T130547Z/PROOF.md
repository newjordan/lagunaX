# Laguna tip proof suite — proof-20260806T130547Z

Generated: 2026-08-06 13:55 UTC

Serial track only (one stream). Model: Laguna-XS-2.1 Q4_K_M. Device: Arc Pro B70 SYCL.

## 0. Pins

```
stamp=20260806T130547Z
binary=/home/frosty40/turbo/lx/results/src-repro-20260806T035656Z/bin/llama-bench
-rwxrwxr-x 1 frosty40 frosty40 564624 Aug  6 07:58 /home/frosty40/turbo/lx/results/src-repro-20260806T035656Z/bin/llama-bench
-rwxrwxr-x 2 frosty40 frosty40 570024 Aug  5 23:12 /home/frosty40/turbo/lx/results/src-repro-20260806T035656Z/bin/llama-perplexity
-rwxrwxr-x 2 frosty40 frosty40 662784 Aug  5 23:12 /home/frosty40/turbo/lx/results/src-repro-20260806T035656Z/bin/llama-server
45df4cc686ba1dfd4083a3c22da00c27  /home/frosty40/turbo/lx/results/src-repro-20260806T035656Z/bin/libggml-sycl.so.0.17.0
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
| pp512 | 1159.7 |
| tg128 | 138.8 |
| score vs pin | **+21.81%** |
| decode speedup | 1.293x |
| prefill speedup | 1.018x |
| floors_ok | True |
| candidate | `/home/frosty40/turbo/lx/results/20260806T060605Z/metrics.json` |

## 2. Prefill / decode ladder

# Prefill / decode ladder

| test | n_prompt | n_gen | t/s |
|------|--------:|------:|----:|
|  | 512 | 0 | 1117.76 |
|  | 2048 | 0 | 1908.50 |
|  | 4096 | 0 | 1880.43 |
|  | 8192 | 0 | 1836.07 |
|  | 0 | 128 | 136.32 |


## 3. Perplexity (wikitext-2)

```
perplexity: tokenizing the input ..
perplexity: tokenization took 438.368 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 5.38 seconds per pass - ETA 2.36.150.497 I Final estimate: PPL = 13.7703 +/- 0.27263
perplexity: tokenizing the input ..
perplexity: tokenization took 438.368 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 5.38 seconds per pass - ETA 2.36.150.497 I Final estimate: PPL = 13.7703 +/- 0.27263
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
| 2000 | 359 | 300.60254482243516 | 43.887138736845856 | 1.9 | yes |
| 8000 | 1402 | 325.53534542968805 | 121.05255193911056 | 4.6 | yes |
| 16000 | 2793 | 321.6313862485482 | 110.72587732956865 | 9.0 | yes |
| 32000 | 5577 | 321.4274722567805 | 105.70754685817352 | 17.7 | yes |
| 48000 | 8359 | 318.4852189177248 | 100.41735965104967 | 26.6 | yes |

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
- FAIL: Who wrote Rent according to the dossier? → 'The dossier does not mention who wrote Rent.\n'
- PASS: What year did Rent open per the dossier? → 'Rent opened in 1996 per the dossier.\n'
- PASS: Which 2003 fantasy blockbuster is named in the dossier? → 'The 2003 fantasy blockbuster named in the dossier is **Wicked**.\n'
- PASS: What year did Hamilton open on Broadway per the dossier? → 'Hamilton opened on Broadway in 2015.\n'
- PASS: Who is credited with book, music, and lyrics for Hamilton here? → 'Lin-Manuel Miranda is credited with book, music, and lyrics for Hamilton.\n'
- PASS: What major disruption to Broadway does the 2020s section mention? → 'The 2020s section mentions "pandemic shutdowns" as the major disruption to Broadway.\n'

## Verdict

**STRONG — long-context speed path + retrieval/comprehension held**



## 5. Agent throughput (np=1 chat)

| metric | value |
|--------|------:|
| tool-turn tg mean | 134.7894899927055 |
| tool-turn tg p50 | 134.85930494135567 |
| tool-turn tg min/max | 134.34051402039177 / 134.93417742157655 |
| requests ok | 8/8 |

Depth prefill samples:
```json
{
  "n": 2,
  "mean": 316.36944190534666,
  "values": [
    [
      "depth_2048",
      3161,
      316.302813363929,
      107.52344935225521
    ],
    [
      "depth_8192",
      13955,
      316.4360704467644,
      90.15455872160835
    ]
  ]
}
```

## 6. Held-out pack

- score_pct: **13.0**
- points: 6/46
- outcomes: `{"fail": 20, "pass": 3}`

## Artifacts

Directory: `/home/frosty40/turbo/lx/results/proof-20260806T130547Z`

- `bench/` prefill ladder
- `ppl/` perplexity
- `longctx/` needles + dossier
- `agent/` throughput + held-out + public69
- `logs/server.log`

