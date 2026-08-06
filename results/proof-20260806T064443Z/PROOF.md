# Laguna tip proof suite — proof-20260806T064443Z

Generated: 2026-08-06 07:30 UTC

Serial track only (one stream). Model: Laguna-XS-2.1 Q4_K_M. Device: Arc Pro B70 SYCL.

## 0. Pins

```
stamp=20260806T064443Z
binary=/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench
-rwxrwxr-x 1 frosty40 frosty40 564624 Jul 30 10:02 /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench
-rwxrwxr-x 1 frosty40 frosty40 570024 Jul 30 05:29 /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-perplexity
-rwxrwxr-x 1 frosty40 frosty40 662784 Jul 30 05:29 /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-server
2fb52e1bea8a21cf4a0d500caef33b78  /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/libggml-sycl.so.0.17.0
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
| pp512 | 1157.6 |
| tg128 | 137.9 |
| score vs pin | **+21.14%** |
| decode speedup | 1.284x |
| prefill speedup | 1.016x |
| floors_ok | True |
| candidate | `/home/frosty40/turbo/lx/results/20260806T063740Z/metrics.json` |

## 2. Prefill / decode ladder

# Prefill / decode ladder

| test | n_prompt | n_gen | t/s |
|------|--------:|------:|----:|
|  | 512 | 0 | 1163.56 |
|  | 2048 | 0 | 2000.21 |
|  | 4096 | 0 | 1980.11 |
|  | 8192 | 0 | 1935.79 |
|  | 0 | 128 | 137.47 |


## 3. Perplexity (wikitext-2)

```
perplexity: tokenizing the input ..
perplexity: tokenization took 437.458 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 3.13 seconds per pass - ETA 1.20.951.326 I Final estimate: PPL = 13.7970 +/- 0.27340
perplexity: tokenizing the input ..
perplexity: tokenization took 437.458 ms
perplexity: calculating perplexity over 32 chunks, n_ctx=2048, batch_size=2048, n_seq=1
perplexity: 3.13 seconds per pass - ETA 1.20.951.326 I Final estimate: PPL = 13.7970 +/- 0.27340
Final estimate: PPL = 13.7970 +/- 0.27340
Final estimate: PPL = 13.7970 +/- 0.27340
```

## 4. Long-context quality

# Long-context package eval

- base: `http://127.0.0.1:18911`
- model: `laguna-tip-proof`

## Prefill / short decode ladder

| chars | prompt_n | pp tok/s | tg tok/s | wall_s | ok |
|------:|---------:|---------:|---------:|-------:|:--:|
| 2000 | 359 | 410.9705085731424 | 44.07276413358455 | 1.6 | yes |
| 8000 | 1402 | 1777.730439259186 | 122.54948893033443 | 1.1 | yes |
| 16000 | 2793 | 2144.232452301768 | 112.1260857834635 | 1.6 | yes |
| 32000 | 5577 | 2371.332768102619 | 107.11010249097933 | 2.7 | yes |
| 48000 | 8359 | 2347.2589805331104 | 101.94426834280034 | 3.9 | yes |

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
- PASS: Who were the writers associated with Oklahoma! in this dossier? → 'Rodgers and Hammerstein.\n'
- PASS: In what year did West Side Story premiere on Broadway according to the dossier? → 'West Side Story premiered on Broadway in 1957.\n'
- PASS: Who wrote the lyrics for West Side Story per the dossier? → 'Stephen Sondheim wrote the lyrics for West Side Story per the dossier.\n'
- PASS: Which 1968 musical is linked to counterculture rock energy in the dossier? → 'Hair (1968) brought rock and counterculture energy onto Broadway stages.\n'
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



## 5. Agent throughput (np=1 chat)

| metric | value |
|--------|------:|
| tool-turn tg mean | 136.67579905629356 |
| tool-turn tg p50 | 136.70300347324667 |
| tool-turn tg min/max | 136.39366621912495 / 136.89781006284466 |
| requests ok | 8/8 |

Depth prefill samples:
```json
{
  "n": 2,
  "mean": 1788.0681172553095,
  "values": [
    [
      "depth_2048",
      3161,
      1672.8107136090905,
      108.95044967595753
    ],
    [
      "depth_8192",
      13955,
      1903.3255209015283,
      91.30387642020327
    ]
  ]
}
```

## 6. Held-out pack

- score_pct: **13.0**
- points: 6/46
- outcomes: `{"fail": 20, "pass": 3}`

## Artifacts

Directory: `/home/frosty40/turbo/lx/results/proof-20260806T064443Z`

- `bench/` prefill ladder
- `ppl/` perplexity
- `longctx/` needles + dossier
- `agent/` throughput + held-out + public69
- `logs/server.log`

