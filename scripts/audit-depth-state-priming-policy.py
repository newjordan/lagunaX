#!/usr/bin/env python3
"""Audit llama-bench depth-state priming policy used by Laguna."""
import json, os, pathlib, re, subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env = subprocess.run(["bash", "-lc", "source ./env.sh >/dev/null 2>&1; printf '%s' \"$LX_LLAMA_BENCH\""], cwd=root, text=True, capture_output=True, check=True).stdout
src = pathlib.Path(env).parents[2] / "tools/llama-bench/llama-bench.cpp"
harness = (root / "scripts/bench-serial.sh").read_text()
text = src.read_text()
probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help'],
    cwd=root, text=True, capture_output=True
)
if probe.returncode not in (0, 1):
    raise RuntimeError(f"llama-bench --help failed ({probe.returncode}): {probe.stderr}")
help_text = probe.stdout + probe.stderr
result = {
    "executable": env,
    "depth_option_supported": "--n-depth" in help_text,
    "depth_default": int(re.search(r"/\* n_depth\s+\*/ \{ (\d+) \}", text).group(1)),
    "harness_passes_depth": bool(re.search(r"(?:^|\s)(?:-d|--n-depth)(?:\s|$)", harness)),
    "depth_expands_context": "n_prompt + n_gen + n_depth" in text,
    "depth_state_is_snapshotted": "llama_state_seq_get_data(ctx, cstate.buf.data()" in text,
    "depth_state_is_cached": 'depth run %d/%d (cached)' in text,
}
out = root / "benchmark/results/depth-state-priming-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
