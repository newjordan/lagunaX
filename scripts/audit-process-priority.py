#!/usr/bin/env python3
"""Audit process/thread scheduling-priority control and historical coverage."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
HARNESS = (ROOT / "scripts/bench-serial.sh").read_text()
match = re.search(r'export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^"}]+)', ENV)
assert match, "LX_LLAMA_BENCH not found"
binary = Path(match.group(1).replace("$LX_BIN", re.search(r'export LX_BIN="\$\{LX_BIN:-([^"}]+)', ENV).group(1)))
proc = subprocess.run(
    ["bash", "-lc", f'source "{ROOT / "env.sh"}" && "$LX_LLAMA_BENCH" --help'],
    text=True,
    capture_output=True,
)
help_text = proc.stdout + proc.stderr
assert help_text, f"llama-bench help unavailable (exit {proc.returncode})"
priority_help = re.search(r"--prio\s+<-1\|0\|1\|2\|3>.*default:\s*0", help_text) is not None
harness_explicit = re.search(r"(?:^|\s)--prio(?:\s|$)", HARNESS) is not None
counts = Counter()
parsed = 0
for path in list((ROOT / "results").rglob("*.json")) + list((ROOT / "benchmark/results").rglob("*.json")):
    if path.name == "process-priority-audit-20260807.json":
        continue
    try:
        obj = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    stack = [obj]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in {"prio", "priority", "process_priority", "thread_priority"}:
                    counts[str(value)] += 1
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
report = {
    "binary": str(binary),
    "priority_option_supported": priority_help,
    "supported_values": [-1, 0, 1, 2, 3],
    "default_priority": 0,
    "serial_harness_priority_explicit": harness_explicit,
    "json_artifacts_parsed": parsed,
    "historical_priority_records": sum(counts.values()),
    "historical_priority_values": dict(sorted(counts.items())),
}
assert priority_help
assert not harness_explicit
out = ROOT / "benchmark/results/process-priority-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
