# Ship note — Hybrid router mode2 (gather + fused scale) 2026-07-30

## Status: **SCORED TIP** (default hybrid mode)

| arm | pp512 | tg128 | score | golden |
|-----|------:|------:|------:|:------:|
| **hybrid mode2 default** | **1141.4** | **118.8** | **+7.94%** | **OK** |
| prior mode1 tip | 1141.2 | 118.1 | +7.44% | OK |

Formal: `results/20260730T050505Z/`

## What

Default hybrid router is now **mode 2**:

1. Stock sigmoid / add / argsort  
2. **Fused gather** → get_rows buffer  
3. Stock sum_rows / clamp / div  
4. **Fused scale** (`x * s + b`)

Full fused sum+div+scale still **golden-fails** (isolated: stock get_rows + pure math norm fails; stock-div + fused scale OK).

## Modes (`GGML_SYCL_TOPK_MOE_HYBRID_MODE`)

| mode | behavior | golden |
|-----:|----------|:------:|
| 0 | stock oracle | OK |
| 1 | gather + stock norm/scale | OK |
| **2** | **gather + stock div + fused scale (default)** | **OK** |
| 3–5 | isolation paths | 3 OK; pure fused norm FAIL |

## Findings

- Fused **scale** is golden-safe.
- Fused **sum/div** is not (even with sequential sum / stock sum buffers).
- Mode2 is a small step past mode1; tip stack otherwise unchanged.

## Tip stack (default ON)

1. MoE dual SwiGLU  
2. Hybrid **mode2**  
3. Dense dual shexp  
4. MoE down weighted reduce (two-step)  

## Next

1. Bitexact fused sum/div (elementwise oracle vs stock DIV).  
2. Integrated weighted-MMVQ down golden fix.  
3. Multi-token mmid golden fix.
