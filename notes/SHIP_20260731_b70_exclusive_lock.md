# Ship — exclusive B70 GPU lock (anti-wedge) 2026-07-31

**Canonical ops rule:** `notes/B70_NO_CONCURRENT_GPU.md` (read that first).

## Why

Arc Pro B70 + xe + Level-Zero: **two concurrent llama GPU clients** routinely
wedge the driver (`xe_validation_lock`, hung `sycl-ls`, need reboot). This hit
during the laguna speed goal when an agent ran bench + PPL overlapping — not
random hardware; **job mismanagement**.

## What

| piece | role |
|-------|------|
| `scripts/lib-gpu-lock.sh` | `flock` + foreign-proc check; source from harness |
| `scripts/with-gpu-lock` | wrap ad-hoc commands |
| `scripts/gpu-status.sh` | show holder + procs |
| `scripts/test_gpu_lock.sh` | structural test (no GPU) |

Wired into: `bench-serial.sh`, `golden-smoke.sh`, `proof-suite.sh`, `laguna-ab-suite.sh`.

Lock file: `$LX_ROOT/results/.b70-gpu.lock` (+ `.meta` with pid/reason/time).

## Rules for agents / humans

1. **Never** start PPL/bench/server-on-GPU while another GPU llama job is live.
2. Prefer harness scripts (they take the lock). For raw binaries use `with-gpu-lock`.
3. Default is **fail-fast** if busy (`LX_GPU_LOCK_WAIT=0`). Queue with e.g. `LX_GPU_LOCK_WAIT=600`.
4. CPU-only embedding (`-ngl 0` / `--embedding`) is ignored by the foreign check.
5. If already wedged: reboot, then `scripts/resume-after-reboot.sh`.

## Escape hatches (don't use casually)

```bash
LX_GPU_LOCK_SKIP=1     # disable lock entirely
LX_GPU_ALLOW_BUSY=1    # flock only; ignore foreign procs
```
