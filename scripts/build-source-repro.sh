#!/usr/bin/env bash
# Source-level champion reproducibility check.
#
# The shipped champion (score ~1.21-1.23) is a BINARY-patched build
# ("decode-only mm-add" on old 06:38 source) that is NOT reproducible from the
# current worktree source. The worktree now carries 4114 lines of unshipped
# SYCL changes (multitoken/dual_down/topk…). Before any further speed work can
# build on a source-level champion, we must know whether the CURRENT source
# builds to ~champion decode speed (source already contains the mm-add decode
# path) or slower (the binary patch is not in source yet).
#
# This script:
#   1. reuses the cmake config of build-base-control (same flags, so the only
#      delta vs the pinned v0 binary is the worktree source diff)
#   2. builds llama-bench from the current worktree source into build-src-now/
#   3. runs the OFFICIAL serial bench (bench-serial.sh) with the new binary
#   4. writes a sidecar result/<stamp>/source-repro.md comparing vs champion
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_WT="/home/frosty40/turbo/worktrees/treebeard-base-control-latest"
# Build dir must live OUTSIDE the worktree (worktree root is not writable);
# drop it under results/ so it is an addressable benchmark artifact.
BUILD_DIR="${BUILD_DIR:-$ROOT/results/src-repro-$(date -u +%Y%m%dT%H%M%SZ)}"
SRC_BIN="$BUILD_DIR/bin"

if [[ ! -f "$SRC_WT/CMakeLists.txt" ]]; then
  echo "missing source tree: $SRC_WT" >&2
  exit 1
fi
mkdir -p "$BUILD_DIR" "$ROOT/results"
# Read the pinned build's cmake cache to mirror its configure flags.
CACHE="$SRC_WT/build-base-control/CMakeCache.txt"
if [[ ! -f "$CACHE" ]]; then
  echo "missing $CACHE — cannot mirror pinned configure" >&2
  exit 1
fi

# Extract -D entries of interest from the cache (VAR:TYPE=VALUE lines).
pick() { awk -F= -v k="$1" '$1 ~ ("^" k ":") { sub(/^[^=]*=/,""); print }' "$CACHE" | tail -1; }
FLAGS=()
for v in CMAKE_BUILD_TYPE CMAKE_CXX_COMPILER CMAKE_C_COMPILER GGML_SYCL GGML_SYCL_F16 GGML_SYCL_TARGET GGML_SYCL_USE_XMX GGML_NATIVE GGML_OPENMP GGML_LLAMAFILE GGML_CPU_ALL_VARIANTS GGML_CUDA GGML_VULKAN GGML_METAL GGML_CPU; do
  val="$(pick "$v" || true)"
  if [[ -n "$val" && "$val" != "NOTFOUND" && "$val" != "OFF" ]]; then
    FLAGS+=("-D${v}=${val}")
  fi
done
# Guard: SYCL must be on or the whole point is lost.
if ! grep -q "^GGML_SYCL:BOOL=ON" "$CACHE"; then
  echo "pinned cache does not enable GGML_SYCL — abort (flags: ${FLAGS[*]})" >&2
  exit 1
fi

echo "== mirroring pinned cmake flags =="
printf '  %s\n' "${FLAGS[@]}"

# fresh configure+build (keep prior dir on failure for forensics)
if [[ -x "$SRC_BIN/llama-bench" && -z "${LX_REBUILD:-}" ]]; then
  echo "== reusing existing build: $BUILD_DIR =="
else
  rm -rf "$BUILD_DIR"
  source "$ROOT/env.sh" >/dev/null 2>&1 || true
  cmake -S "$SRC_WT" -B "$BUILD_DIR" "${FLAGS[@]}" >"$BUILD_DIR.configure.log" 2>&1 || {
    echo "cmake configure failed:"; tail -40 "$BUILD_DIR.configure.log"; exit 1; }
  cmake --build "$BUILD_DIR" -j"${THREADS:-16}" --target llama-bench >"$BUILD_DIR.build.log" 2>&1 || {
    echo "build failed:"; tail -60 "$BUILD_DIR.build.log"; exit 1; }
fi

test -x "$SRC_BIN/llama-bench" || { echo "llama-bench missing after build" >&2; exit 1; }

# Official bench with the fresh binary. The harness scores vs pinned baseline.
export LX_LLAMA_BENCH="$SRC_BIN/llama-bench"
export LX_LLAMA_CLI="$SRC_BIN/llama-cli"
"$ROOT/scripts/bench-serial.sh" --note "src-repro:current-worktree-source-vs-binary-patched-champion"
