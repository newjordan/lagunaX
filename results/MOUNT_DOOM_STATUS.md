# Mount Doom status — 2026-07-30 (quality-safe tip)

## Where we are

| claim | status |
|-------|--------|
| Full tip +63% (default binary fuses) | **INVALID** — broken logprobs / multitoken |
| **Quality-safe tip** (3 env kills, now `env.sh` default) | **LIVE bankable floor** |
| Formal | **pp 1185 / tg 135.7 / +20.4%** (`20260730T194852Z`) |
| PPL tiny | **1.0004** |
| Golden | OK |
| Chat | **OK** (real fib code, tg~136) |
| Short needle | weak fail (echo, not crash/garbage) |
| Agent Bench 69 | not re-run yet |

## Kills (required)

```bash
# in env.sh defaults
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
```

Culprits: `MUL_MAT_ADD` alone → PPL~1e6; `MOE_DUAL_DOWN` → PPL fail; dual multitoken needed for tip restore.

## Not done

- Fix mm-add / dual-down in source (or leave dead)
- Full long-ctx needles + agent 69 on quality-safe stack
- Source default-OFF rebuild (env only for now; tip binary still compiles fuses ON)
