# B70: no concurrent GPU clients

**Canonical ops rule for Arc Pro B70 on this box.**

## Why

xe + Level-Zero: **two concurrent llama GPU clients** routinely wedge the
driver (`xe_validation_lock`, hung `sycl-ls`, device-lost → reboot). This is
job mismanagement, not random hardware. Hit hard during laguna speed work when
bench + PPL (or two benches) overlapped.

## Rule

**One Level-Zero GPU owner at a time.** Never start:

- `llama-bench` + `llama-perplexity`
- two benches
- bench + `llama-server` on GPU
- agent + harness both launching GPU work

…while another GPU llama job is live.

## How (harness)

| piece | role |
|-------|------|
| `scripts/lib-gpu-lock.sh` | `flock` + foreign-proc check |
| `scripts/with-gpu-lock` | wrap ad-hoc commands |
| `scripts/gpu-status.sh` | show holder + foreign procs |
| `scripts/test_gpu_lock.sh` | structural test (no GPU) |

Lock file: `$LX_ROOT/results/.b70-gpu.lock` (+ `.meta` with pid/reason/time).

Wired into: `bench-serial.sh`, `golden-smoke.sh`, `proof-suite.sh`,
`laguna-ab-suite.sh`, `quest-mount-doom.sh`.

```bash
./scripts/gpu-status.sh
./scripts/with-gpu-lock --reason ppl -- \
  "$LX_BIN/llama-perplexity" -m "$LX_MODEL" ...
```

- Default: **fail-fast** if busy (`LX_GPU_LOCK_WAIT=0`)
- Queue: e.g. `LX_GPU_LOCK_WAIT=600`
- CPU-only (`-ngl 0` / embedding) is ignored by the foreign check

## Escape hatches (don't use casually)

```bash
LX_GPU_LOCK_SKIP=1     # disable lock entirely
LX_GPU_ALLOW_BUSY=1    # flock only; ignore foreign procs
```

## If already wedged

1. Reboot
2. `bash scripts/resume-after-reboot.sh` (or: golden-smoke + bench-serial on tip)
3. See `notes/SHIP_20260731_b70_exclusive_lock.md`

## Ship note

`notes/SHIP_20260731_b70_exclusive_lock.md`
