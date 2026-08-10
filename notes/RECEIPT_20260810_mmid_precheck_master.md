# RECEIPT 2026-08-10 — MUL_MAT_ID pre-check on clean master (issue #25455)

`test-backend-ops test -b SYCL0 -o MUL_MAT_ID` on upstream master `dd1ea5243`
(build: icpx 2026.0, Release, GGML_SYCL=ON/F16=ON/TARGET=INTEL, worktree
`lx-master-baseline`), Arc Pro B70, xe driver, compute-runtime 26.18:

**Backend SYCL0: OK — all MUL_MAT_ID cases pass. 2/2 backends passed.**

Upstream issue #25455 (MUL_MAT_ID wrong results on Arc Pro B70, reported
against an earlier base) does NOT reproduce on this master/driver/compiler
combo. Consequence for the PR series: PR-D/E MoE claims are not sitting on a
broken op; no standalone fix-PR is needed first.

Procedural note: the first pre-check attempt crashed by loading the champion
libggml-sycl via with-gpu-lock → env.sh LD_LIBRARY_PATH capture (the exact
FINDING_20260810 mechanism, third sighting). Any cross-build tool run must
export LX_BIN first. Receipt logs: scratchpad tasks b1omedt6h (pass),
bxd3xserp (contaminated crash, discarded).
