# FRONTIER: oneDNN Primitive-Path Completeness Audit

## Direction (distinct from all 13 tried)
The trace exercises exactly **2 primitive types** (matmul exec + gemm create) across
381,430 GPU events. There are ZERO data-movement, synchronization, reduction, eltwise,
or softmax primitives at the oneDNN primitive layer. oneDNN is a pure-GEMM offload
engine in this build — it has no visibility into any non-GEMM operation and therefore
zero cross-op fusion opportunity at the primitive layer, regardless of attribute hints
or kernel selection.

## Key evidence (verified this iteration)
- `grep -cE 'primitive,(exec|create).*,(reorder|concat|sum|binary|reduction|eltwise|softmax|conv)'` = **0**
- `grep -ciE 'barrier|wait|sync|fence|event'` = **0** — no explicit sync primitives anywhere
- Only graph-path ops (sdp/attention, 1,280 calls) carry `fpm:strict`; all matmul carry `fpm:f16`
- Graph path has NO scratchpad attr; all matmul carry `attr-scratchpad:user`
