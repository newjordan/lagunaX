# Handoff — Laguna serial (lx) — afternoon pause 2026-07-31

**Paused:** quest-2day loop stopped · GPU free · no llama GPU procs  
**Resume:** not required today; when ready see “How to resume” below.

---

## Goal (still open)

| Bar | Score | Notes |
|-----|------:|-------|
| 2-day goal | **≥ 1.250** | deadline was 2026-08-02 14:00 UTC (`GOAL_2DAY.md`) |
| **Live champion (ship)** | **1.227** | tg **139.3** · pp **1183** · golden OK · PPL ~12.6 |
| Tip-freeze QS bar | 1.209 | do not go below |

Champion formal: `results/20260731T141436Z/` (post-brownout rebench)  
Seed champ also in `results/quest-2day-20260731/CHAMPION.json`

**Do not re-pin baseline or golden.**

---

## What is shipped / default

| Piece | Path / value |
|-------|----------------|
| Harness | `/home/frosty40/turbo/lx` |
| **Ship binary** | `treebeard-base-control-latest/build-mmadd-decode/bin` |
| Model | Laguna XS 2.1 Q4_K_M under `/mnt/data2tb/laguna/archived/...` |
| `env.sh` | dual_down **OFF**, dual_multitoken **OFF**, mm-add **ON** (decode-only binary patch) |
| Ops | one GPU client · `notes/B70_NO_CONCURRENT_GPU.md` · `scripts/with-gpu-lock` |

Tip binary has full quality-safe fuse stack + decode-only mm-add patch (`scripts/patch-mmadd-decode-only.py`).

---

## Breakthrough this session (not yet tip score)

**Root cause of dual_down expert-loop PPL -nan:**  
Reordered Q4/Q5/Q6_K MoE weights (SoA) + multi-row `mul_mat` falling through to MMQ/oneDNN (linear layout) → garbage.

**Fix in source** (`treebeard-base-control-latest/ggml/src/ggml-sycl/ggml-sycl.cpp`):

1. **Reorder-safe multi-col:** if `src0` already reordered, chunk `N` by `MMVQ_MAX_BATCH_SIZE` and only run reorder-MMVQ — never GEMM on reordered quant.
2. **dual_down expert-loop** restored (patch 0028 intent) + **packed weighted reduce** (skip scatter).
3. Also earlier: decode-only mm-add gate, FA VEC GQA default (partial tip restore).

**Proof (source build `build-base-control`):**

| Gate | dual_down ON + fix |
|------|---------------------|
| Expert-loop hit | yes (`packed_reduce=1`, n_tokens=512) |
| Wiki PPL 2×512 | **~13.07** (was -nan on tip binary) |
| Golden | **OK** |
| Formal score | **~1.182** (tg~134 · pp~1149) |

Source stack is still **missing tip residual fuses** (rms/softplus/add_add/rope parity) → score below tip champ **1.227**.

Detail: `notes/SHIP_20260731_dual_down_mul_mat_reorder_fix.md`  
Results: `results/src-dual-down-packed-20260731T170108Z/`, `results/src-dual-down-fix-20260731T165656Z/`  
Bisect trail: `notes/SHIP_20260731_dual_down_expert_loop_ppl.md`, `results/eloop-bisect-20260731T163909Z/`, `results/push-wave-20260731T1622/`

---

## What NOT to ship yet

- Tip binary dual_down **ON** without mul_mat reorder fix → PPL -nan  
- `GGML_SYCL_ENABLE_OPT=0` as a “fix” → PPL OK but decode/golden die  
- dual_multitoken ON alone on tip → also PPL -nan (same class of bug)  
- any-batch mm-add (`ENABLE_MUL_MAT_ADD_ANY_BATCH`) → PPL 1e5+  

---

## Next (when Laguna resumes)

Ordered for score toward **1.250**:

1. **Tip-parity residual fuses in source** — rms/softplus/add_add/rope (patches 0018–0020/0024 are mostly stubs; tip binary still has the bodies). Goal: source dual_down + full tip stack ≥ 1.227 golden+PPL.
2. **Land mul_mat chunk fix on full tip stack** — either rebuild tip-parity source as ship binary, or carefully merge fix into a tip-complete tree. Drop `build-mmadd-decode` binary patch only after source golden matches tip oracle.
3. **Re-measure dual_down formal** on tip-parity build (expect prefill reclaim; tip invalid arm was ~pp 1215 / score ~1.231).
4. Optional: packed_reduce + expert-loop polish; decode MoE micro-kernels if prefill goal is met.

Env research (source only, after rebuild):

```bash
export LX_BIN=.../build-base-control/bin
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=0
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0
# leave OPT default (1) — do not set ENABLE_OPT=0
```

Ship stays:

```bash
source /home/frosty40/turbo/lx/env.sh   # dual_down OFF, tip build-mmadd-decode
./scripts/golden-smoke.sh
./scripts/bench-serial.sh --note "resume check"
```

---

## How to resume infrastructure

```bash
cd /home/frosty40/turbo/lx
# GPU free check
./scripts/gpu-status.sh

# Optional: continuous rebench (20 min idle)
rm -f results/quest-2day-20260731/PAUSED.txt
export QUEST_SLEEP_S=1200
bash scripts/quest-2day-launch.sh
tail -f results/quest-2day-20260731/quest.log

# Source rebuild after edits
source /opt/intel/oneapi/setvars.sh --force
cmake --build /home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-base-control \
  -j"$(nproc)" --target llama-bench llama-perplexity
```

**B70:** never concurrent Level-Zero clients (bench + ppl wedges xe).

---

## Files touched (source tree)

Worktree: `treebeard-base-control-latest`

- `ggml/src/ggml-sycl/ggml-sycl.cpp` — mul_mat reorder chunk; dual_down expert-loop; packed reduce; mm-add fuse
- `ggml/src/ggml-sycl/mmvq.{hpp,cpp}` — row_addend API (earlier)
- `ggml/src/ggml-sycl/fattn.cpp` — FA VEC GQA default (earlier)
- `ggml/src/ggml-sycl/topk-moe.{hpp,cpp}` — fuse wire for mm-add (earlier)

Build: `build-base-control` (research) · Ship binary: `build-mmadd-decode` (unchanged tip)

---

## Afternoon status

| Item | State |
|------|--------|
| Quest loop | **STOPPED** (`PAUSED.txt` set) |
| GPU lock | free |
| Champion claim | tip **1.227** quality-safe |
| Kernel win banked | dual_down PPL root cause + source fix |
| Score win banked | none vs 1.227 yet |

Wind-down complete. Thanks — pick up from this note when ready.
