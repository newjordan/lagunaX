# Post-brownout check-in (2026-07-31)

## System

| check | result |
|-------|--------|
| uptime after reboot | ~53 min at check-in |
| `sycl-ls` B70 | OK (`level_zero:gpu` Arc Pro B70) |
| GPU lock | free |
| foreign GPU procs | none |
| Hydra head | up (`127.0.0.1:17876`) |
| model GGUF | present on `/mnt/data2tb/...` |
| champion lib patch | still applied (`cmp $0x1` @ 0x22952f) |

## Reconfirm champion (binary tip + decode-only mm-add)

| gate | result |
|------|--------|
| golden | **OK** |
| formal `20260731T141436Z` | **pp 1183.3 / tg 139.3 / score 1.227 (+22.74%)** |

Still beats tip-freeze bar (1.209 / tg 136.4). Env: `env.sh` → `build-mmadd-decode`.

## Source restore started (not tip yet)

Landed **decode-only `fuse_mul_mat_add` + MMVQ `row_addend{,2}`** into control source:

- `ggml/src/ggml-sycl/mmvq.{hpp,cpp}` — addend API + reorder epilogue
- `ggml/src/ggml-sycl/ggml-sycl.cpp` — `ggml_sycl_fuse_mul_mat_add` (ne11==1 default)
- `ggml/src/ggml-sycl/topk-moe.{hpp,cpp}` — wire into `ggml_sycl_fuse`

`build-base-control` rebuilds clean. **Golden still mismatches** tip oracle even with
`GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1` — source tree is still missing tip bodies
(softplus/rms/rope/add_add/FA/qkv partial) from the 2026-07-30 strip. Do **not**
point `LX_BIN` at base-control until full tip stack is restored.

## Ops notes restored

- `notes/B70_NO_CONCURRENT_GPU.md` (was empty after brownout)
- `QUEST.md` live tip → quality-safe + decode mm-add (+22.7%), not invalid +63%
- `results/MOUNT_DOOM_STATUS.md` refreshed

## Kernel map (post tip, onednn decode)

Still MoE tiny-N gate/up (`1x512x2048 : 1x2048xN`, N≈9–17) dominate GPU time under
oneDNN path. `results/ktrace-post-brownout-20260731/`.

## Next on the horse

1. Keep scoring on **`build-mmadd-decode`** (binary champion).
2. Finish tip-source restore (rms/rope/softplus/add_add + FA) so source can replace the patch.
3. dual_down still killed (PPL break) — separate fix track.
4. No concurrent GPU jobs.
