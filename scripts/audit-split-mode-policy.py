#!/usr/bin/env python3
import json, os, re, subprocess
from pathlib import Path
root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    m = re.search(r'^export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^}]+)\}"', env_text, re.M)
    raw = m.group(1).replace("$LX_BIN", re.search(r'^export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text, re.M).group(1))
    bench = raw
help_text = subprocess.run([bench, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
line = next(x for x in help_text.splitlines() if "--split-mode" in x)
artifact = {
    "executable": bench,
    "help_contract": line.strip(),
    "supported": True,
    "allowed_values": re.search(r'<([^>]+)>', line).group(1).split('|'),
    "default": re.search(r'\(default: ([^)]+)\)', line).group(1),
    "canonical_configuration_present": bool(re.search(r'(^|\s)(-sm|--split-mode)(\s|$)', env_text + "\n" + harness_text)),
    "metrics_provenance_present": bool(re.search(r'["\'](?:split_mode|split-mode)["\']\s*:', harness_text)),
}
out = root / "benchmark/results/split-mode-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["allowed_values"] == ["none", "layer", "row", "tensor"]
assert artifact["default"] == "layer"
assert not artifact["canonical_configuration_present"]
assert not artifact["metrics_provenance_present"]
print(out)
