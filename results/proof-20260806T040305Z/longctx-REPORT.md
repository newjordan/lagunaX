# Long-context package eval

- base: `http://127.0.0.1:18911`
- model: `laguna-tip-proof`

## Prefill / short decode ladder

| chars | prompt_n | pp tok/s | tg tok/s | wall_s | ok |
|------:|---------:|---------:|---------:|-------:|:--:|
| 2000 | 359 | 403.43424808396827 | 43.928106163250575 | 1.6 | yes |
| 8000 | 1402 | 1762.3936372812877 | 122.64483588587898 | 1.1 | yes |
| 16000 | 2793 | 2143.429425686772 | 112.19094899518981 | 1.6 | yes |
| 32000 | 5577 | 2351.9786436796635 | 107.19729326834498 | 2.7 | yes |
| 48000 | 8359 | 2334.021058404137 | 102.01706873081203 | 3.9 | yes |

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
- PASS: Who were the writers associated with Oklahoma! in this dossier? → 'Rodgers and Hammerstein were the writers associated with Oklahoma! in this dossier.\n'
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

