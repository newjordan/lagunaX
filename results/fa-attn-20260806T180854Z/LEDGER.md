# fa-attn LEDGER — 20260806T180854Z

Axis: flash-attention type on the SYCL backend (-fa auto|on|off), official geometry,
same-window 3-arm, champion binary (src-repro-20260806T035656Z/bin/llama-bench),
GGML_SYCL_DISABLE_DNN=1 + DISABLE_GRAPH=1 env, golden-gated (lock wrapper).

Results (llama-bench -o json, pretty-printed; parsed with JSONDecoder over whole file):
- ctrl-auto (-fa auto): pp 1118.82 (SD 11.79), tg 136.13 (SD 0.20), json flash_attn=-1
- fa-on    (-fa on):   pp 1118.03 (SD 11.61), tg 137.16 (SD 0.25), json flash_attn=1
- fa-off   (-fa off):  pp 1034.58 (SD 8.08),  tg 95.11  (SD 0.18), json flash_attn=0

Conclusions:
1. fa-off is CATASTROPHIC: tg -30.1%, pp -7.5% vs ctrl-auto. Flash attention is
   load-bearing in this build on both sides; the champion can never ship -fa off.
2. ctrl-auto behaves like fa-on (tg 136.1 vs 137.2, both far from fa-off's 95.1):
   AUTO already resolves to FA-on in the champion env. JSON flash_attn=-1 is a
   sentinel (llama-bench keeps the raw param for auto; resolution is internal).
   => the FA axis is CLOSED: no submission lever, champion already on FA.
3. Attribution flag: this env -i window's ctrl-auto (pp 1118.82, tg 136.13) sits
   ~3.5% below the day's full-env runs (pp 1155-1168, tg 137.5-138.8, board
   pp 1159.74/tg 138.83). All three arms share the stripped env so the RELATIVE
   comparison is valid, but the ABSOLUTE level gap vs full-env windows is NOT yet
   attributed (stripped SYCL_PI_LEVEL_ZERO_* / OMP / governor state? first-arm
   cold effect?). Do not use this window's absolute numbers as a board baseline.

Parser note: llama-bench -o json here emits PRETTY-PRINTED multi-line objects
(indented), not one object per line — the fa-attn-probe-cycle.sh receipt grep
pattern 'avg_ts':[0-9.]* therefore printed empty. Use the JSONDecoder scan over
the whole file (strip [lx-*] stderr lines first). Keep the pattern for the
per-line-emitting builds (finding 20) and switch by detecting leading '{' at
line start without indentation.
