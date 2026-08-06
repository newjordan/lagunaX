#!/usr/bin/env python3
"""Audit flash-attention configuration and metrics provenance."""
import json, os, pathlib, re, subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
match = re.search(r'export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^}]*)\}"', env)
binary = os.environ.get("LX_LLAMA_BENCH")
if not binary:
    candidates = list((root / "baseline").glob("**/llama-bench"))
    if not candidates:
        raise SystemExit("no llama-bench found")
    binary = str(candidates[0])
help_run = subprocess.run(
    ["bash", "-c", 'source ./env.sh >/dev/null 2>&1; exec "$1" --help', "audit", binary],
    cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
# Some accelerator builds print usable help before failing runtime initialization.
help_text = help_run.stdout
if not help_text:
    raise SystemExit(f"llama-bench --help produced no output (status {help_run.returncode})")
contract = re.search(r'-fa, --flash-attn <([^>]+)>\s+\(default: ([^)]+)\)', help_text)
if not contract:
    raise SystemExit("flash-attn contract absent")
fa_default = re.search(r'export FA="\$\{FA:-(.*?)\}"', env)
passed = bool(re.search(r'\n\s*-fa "\$FA"', bench))
metrics_block = bench[bench.find('payload = {'):]
recorded = bool(re.search(r'["\'](?:fa|flash_attn|flash_attention)["\']\s*:', metrics_block, re.I))
out = {
    "binary": binary,
    "supported_values": contract.group(1).split("|"),
    "executable_default": contract.group(2),
    "laguna_env_default": fa_default.group(1) if fa_default else None,
    "passed_to_benchmark": passed,
    "recorded_in_metrics": recorded,
}
path = root / "results/flash-attention-policy-audit-20260807.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(path)
assert out["supported_values"] == ["on", "off", "auto"]
assert out["executable_default"] == "auto"
assert out["laguna_env_default"] == "-1"
assert passed and recorded
