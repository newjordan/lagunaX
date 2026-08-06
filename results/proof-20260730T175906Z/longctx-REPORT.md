# Long-context package eval

- base: `http://127.0.0.1:18911`
- model: `laguna-tip-proof`

## Prefill / short decode ladder

| chars | prompt_n | pp tok/s | tg tok/s | wall_s | ok |
|------:|---------:|---------:|---------:|-------:|:--:|
| 2000 | 359 | 463.9309815435591 | 44.268416006905866 | 1.5 | yes |
| 8000 | 1402 | 2657.945874212048 | 123.66719611685004 | 0.8 | yes |
| 16000 | 2793 | 3155.7681747615375 | 113.17578321178726 | 1.2 | yes |
| 32000 | 5577 | 3478.1057700882902 | 108.22693836137651 | 1.9 | yes |
| 48000 | — | — | — | — | FAIL |

## Needle-in-haystack

- doc_chars: 34044
- paragraphs: 80
- score: **0/3**

- depth None: FAIL — Carousel → '<urlopen error [Errno 111] Connection refused>'
- depth None: FAIL — Company → '<urlopen error [Errno 111] Connection refused>'
- depth None: FAIL — Hadestown → '<urlopen error [Errno 111] Connection refused>'

## Broadway long dossier QA

- doc_chars: 61092
- score: **0/14**

- FAIL: What year did Oklahoma! open? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Who were the writers associated with Oklahoma! in this dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: In what year did West Side Story premiere on Broadway according to the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Who wrote the lyrics for West Side Story per the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Which 1968 musical is linked to counterculture rock energy in the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Which 1975 show about dancers is mentioned as a long-running phenomenon? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Which composer is tied to mega-musicals like Phantom in the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: What year does the dossier give for Phantom on Broadway? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Who wrote Rent according to the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: What year did Rent open per the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Which 2003 fantasy blockbuster is named in the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: What year did Hamilton open on Broadway per the dossier? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: Who is credited with book, music, and lyrics for Hamilton here? → '<urlopen error [Errno 111] Connection refused>'
- FAIL: What major disruption to Broadway does the 2020s section mention? → '<urlopen error [Errno 111] Connection refused>'

## Verdict

**WEAK — long-context quality or prefill failures**

