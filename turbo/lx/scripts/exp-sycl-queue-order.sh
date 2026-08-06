#!/usr/bin/env bash
# exp-sycl-queue-order.sh — benchmark the impact of SYCL queue ordering.
#
# New direction: all 16 prior optimization directions modified ggml kernel/fuse
# code. This tests a RUNTIME-CONFIGURATION lever: the SYCL queue submission mode.
#
# Finding #13: expert GEMM stream alternates exec→cache_hit→exec on a
# synchronous host path. If the SYCL queue is in_order, exec submission blocks
# until the GPU accepts it. An out_of_order queue lets the host fire-and-forget.
#
# This script is a harness stub — the actual queue-property change requires a
# ggml-sycl source edit (remove sycl::property::queue::in_order() from the
# queue constructor). This script benchmarks before/after when both binaries
# are available.
set -euo pipefail
source "$(dirname "$0")/../env.sh"

echo "=== SYCL Queue Ordering Experiment ==="
echo "Binary: $LX_LLAMA_BENCH"
echo "Model:  $LX_MODEL"
echo ""

# Baseline (current ship binary — whatever queue mode it uses)
echo "--- Baseline (current binary) ---"
LX_RESULTS="$LX_RESULTS" "$LX_ROOT/scripts/bench-serial.sh" 2>&1 | tee "$LX_RESULTS/exp-queue-baseline.txt"

echo ""
echo "=== Next steps ==="
echo "1. Build a second binary with sycl::property::queue::in_order() REMOVED"
echo "   from all queue constructors in ggml/src/ggml-sycl/"
echo "2. Re-run this script with LX_BIN pointing to the modified binary"
echo "3. Compare decode_tok_s: out_of_order should narrow the max/p50 tail"
echo "   (finding #15: max=12.5x p50 → expected <5x with overlapped submission)"
echo ""
echo "Expected impact: if host cache_hit (1.185 µs/call × 184K calls = 224.8 ms)"
echo "overlaps even 50% with GPU exec, decode gains ~2.9% from overlap alone,"
echo "plus tail-latency reduction from breaking the submission barrier."
