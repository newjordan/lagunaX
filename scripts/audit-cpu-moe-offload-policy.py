#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text(errors="replace")
bench_text = (root / "scripts/bench-serial.sh").read_text(errors="replace")
bench_bin = os.environ.get("LX_LLAMA_BENCH")
if not bench_bin:
    resolved = subprocess.run(
        ["bash", "-c", 'source "$1" >/dev/null 2>&1; printf "%s" "$LX_LLAMA_BENCH"', "bash", str(root / "env.sh")],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if not resolved:
        raise SystemExit("cannot resolve LX_LLAMA_BENCH from env.sh")
    bench_bin = resolved
help_text = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help', "bash", str(root / "env.sh")],
    check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
).stdout
line = next((x.strip() for x in help_text.splitlines() if "--n-cpu-moe" in x), None)
if line is None:
    raise SystemExit("active llama-bench does not expose --n-cpu-moe")
default = re.search(r"default:\s*([^\)]+)", line)
explicit = bool(re.search(r"(?:-ncmoe|--n-cpu-moe)\b", env_text + "\n" + bench_text))
parsed = 0
mentions = 0
for path in (root / "results").rglob("*.json"):
    if path.name == "cpu-moe-offload-policy-audit-20260807.json":
        continue
    try:
        text = path.read_text(errors="replace")
        json.loads(text)
    except Exception:
        continue
    parsed += 1
    mentions += int(bool(re.search(r'n[_-]?cpu[_-]?moe|cpu[_-]?moe', text, re.I)))
report = {
    "active_help_line": line,
    "executable_default_cpu_moe_layers": int(default.group(1)) if default else None,
    "active_sources_override_cpu_moe": explicit,
    "effective_cpu_moe_layers": None if explicit else int(default.group(1)),
    "parsed_json_artifacts": parsed,
    "artifacts_mentioning_cpu_moe": mentions,
    "quality_result": "not measured by this policy-coverage audit",
}
out = root / "results/cpu-moe-offload-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
