# Ship posture — quality-safe tip (2026-07-30)

## Status: **DEFAULT FOR LX HARNESS** (env kills; binary still has fuses compiled ON)

Broken tip formal **+63% does not count**. PPL/chat/long-ctx failed under full default stack.

### Isolation

| arm | tiny-corpus PPL |
|-----|----------------:|
| all major fuses off | ~1.00 |
| only `MUL_MAT_ADD` ON | ~8.8e5 |
| only `MOE_DUAL_DOWN` ON | FAIL (neg logprob stddev) |
| tip minus mm-add + dual-down + dual-multitoken | **~1.00** |

Bisect: `results/ppl-enable-20260730T194100Z/`, `results/ppl-bisect-20260730T193454Z/`

### Quality-safe env (now in `env.sh`)

```bash
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
```

### Formal (quality-safe)

| arm | pp512 | tg128 | score |
|-----|------:|------:|------:|
| pin | 1139 | 107.35 | 0 |
| broken tip (invalid) | ~3730 | ~139 | +63% |
| **quality-safe** `20260730T194852Z` | **1185** | **135.7** | **+20.4%** |

Prefill 3× was the broken dual multitoken path. Surviving win is mostly **decode**.

### Gates

| gate | quality-safe |
|------|:------------:|
| PPL tiny | OK ~1.0 |
| Golden greedy | OK |
| Formal serial | +20.4% |
| Chat / agent / long-ctx needles | re-check after env default |

### Next

1. Leave three fuses dead until fixed in source (default OFF or corrected).
2. Prove chat + long-ctx + agent on quality-safe stack.
3. Only then reclaim mm-add / dual-down with bitexact/PPL gates.
