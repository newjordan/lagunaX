# Frontier: oneDNN verbose-logging I/O contamination + decode compute decomposition

## Direction
The ktrace trace is 382,797 lines of synchronous host-side fprintf — every
primitive create/exec event writes a ~200-byte CSV line before the next GPU
command can be enqueued. No prior direction (1-11) examined whether this
logging I/O contaminates the per-call timing measurements that ALL findings
depend on. This is a measurement-methodology axis, distinct from every
runtime-optimization direction tried.

## Finding 1: Prefill timing is inflated 2.24× by verbose logging
The trace's own benchmark table reports `pp256 = 522.16 t/s`, while the clean
scored benchmark (LATEST_SCORE.json) reports `prefill_tok_s = 1167.15 t/s`.
Ratio: 1167.15 / 522.16 = **2.236×**.

But decode is NOT inflated: trace `tg128 = 139.64 t/s` vs scored
`decode_tok_s = 138.02 t/s` → ratio **0.988×** (trace is actually 1.2% faster).

This asymmetry is consistent with logging I/O being rate-bound: prefill
generates 5,866 matmul events in ~123 ms (47.5K events/s) while decode
generates 184,345 in ~3705 ms (49.8K events/s) — similar rates, but prefill's
compute is much faster per event, so the fixed logging cost per event is a
larger fraction of prefill wall time.

Caveat: the trace build (`7e1e28cae`) may differ from the scored build
(`treebeard-base-control-latest/build-mmadd-decode`), so the 2.24× factor
confounds build-difference with logging. But the decode match (0.988×) makes
pure build-difference unlikely — a slower build would inflate decode too.

## Finding 2: Decode compute decomposition (new, not in any prior finding)
Decode region (lines 12494+):
- Expert matmul: **3704.7 ms** (184,345 calls) = **96.8%** of decode compute
- Attention SDP: **71.5 ms** (1,240 calls) = **1.9%**
- Create overhead: **~232 ms** = **6.1%** (but overlaps with matmul)
- Dense/shexp matmul: included in the 184,345 count

The attention path has **zero leverage** (1.9%) — every microsecond saved on
expert matmul is 50× more impactful than the same saving on attention.

## Finding 3: Attention SDP has zero host-overhead signature
The 960 SDP calls (shape 1x8x8x256x128) show:
- p50 = 0.0591 ms, p99 = 0.0688 ms, max = 0.0999 ms
- p99/p50 = **1.16×**, max/p50 = **1.69×**

vs expert matmul (M=512 gate/up):
- p50 = 0.0188 ms, p99 = 0.0278 ms, max = 0.200 ms
- p99/p50 = **1.48×**, max/p50 = **12.5×** (from finding #40)

The attention path's tight distribution (1.16×) vs the matmul path's wide
distribution (12.5×) confirms: the attention graph-exec path has NO
host-side serialization stalls, while the matmul primitive path is dominated
by them. The graph compiler eliminates the per-call host lookup that the
primitive path pays.

## Finding 4: 148.7 matmul launches per attention launch in decode
184,345 decode matmul execs / 1,240 decode attention execs = **148.7×**.
The GPU launches 148.7 expert GEMM kernels for every 1 attention kernel —
quantifying the dispatch imbalance that makes the expert stream the sole
optimization target.

## Hypothesis: the 1.18 µs cache_hit cost is partly logging I/O
The create:cache_hit mean is 1.185 µs (finding #1). If each cache_hit event
triggers a synchronous ~200-byte fprintf before the next exec can be
enqueued, the measured 1.185 µs conflates the actual hash-lookup cost with
I/O syscall overhead. A clean re-measurement (verbose disabled, using
DPC++ profiling events instead of verbose logging) would establish the true
floor. If I/O accounts for even 30% of the 1.185 µs, the real cache_hit tax
drops from 224.8 ms to ~157 ms — still significant, but lower priority than
the 1142 ms gate/up-concatenation win.

## Roadmap to 2.0× score
Current: decode 1.286×, prefill 1.025× → score 1.215×.
Target: score 2.0× requires decode_speedup ≈ 2.51× (at prefill 1.025×).
Decode expert matmul = 3704.7 ms; need ≤ 1480 ms.

| Lever | Savings | Mechanism |
|-------|---------|-----------|
| Gate/up concat (dir 10) | ~1142 ms | Eliminate 60,833 launches |
| Weight Q4_K direct (dir 9) | ~300-400 ms | 3.6× BW reduction on BW-bound calls |
| Graph capture (dir 3) | ~225-500 ms | Eliminate cache_hit + enable pipelining |
| **Combined** | **~1665-2042 ms** | **Score ≈ 1.9-2.2×** |

The three levers are additive and independent. Gate/up concat alone gets to
~1.57×; adding weight-dequant gets to ~1.75×; adding graph capture reaches
~1.9-2.1×.
