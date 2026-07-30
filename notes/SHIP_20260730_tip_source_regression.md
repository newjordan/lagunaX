# Incident — tip source regression + QKV shared quant (2026-07-30)

## Status: **partial restore / no scored tip change this fire**

Live scored tip remains formal **+63.67%** (`20260730T144111Z`) on tip binary backup path.
Control **source tree was gutted** mid-campaign by concurrent edits:

| file | tip state | found mid-fire |
|------|-----------|----------------|
| `topk-moe.cpp` | ~1860 lines (GEMV + full-norm hybrid) | **~620 lines stock CUDA port** |
| `topk-moe.hpp` | full fuse decls | dual-only / stripped |
| `ggml-sycl.cpp` fuse bodies | softplus/mm-add/rms/rope/add_add | **missing** (dual/dense/down remain) |

## Restored this fire

1. Re-copied `patches/0048-...fullsnippet.cpp` → `topk-moe.cpp` (router tip stack).
2. Narrowed `ggml_sycl_fuse()` to dual + dense dual + moe-down + topk (linkable).
3. Documented missing fuses need re-wire from fullsnippets when free.

## Unmeasured binary feature: QKV shared quant

Live `libggml-sycl.so` (10:02) logs:
```
[lx-control-qkv] chain done n_mm=3
[lx-control-qkv] launch 0/3 'Qcur-*' ...
```
Kill: `GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1`

Golden vs tip oracle **FAIL** with QKV on (observed this fire). Not shipped.
Next fire should A/B QKV kill vs on once GPU exclusive + source coherent.

## Tip claim

Still **+63.67%** full-norm tip until a new formal with golden OK beats it.
Do **not** rebuild/deploy half-stripped source as tip binary.

## Next

1. Finish restoring softplus/mm-add/rms/rope/add_add from patches into `ggml-sycl.cpp`.
2. QKV shared quant: golden + formal under exclusive GPU or kill by default.
3. Avoid concurrent thrash on `treebeard-base-control-latest` without tip binary backup.
