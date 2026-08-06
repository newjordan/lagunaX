#!/usr/bin/env python3
"""Audit whether canonical Laguna trials use llama-bench's in-process parameter sweeps."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
probe = subprocess.run(
    ["bash", "-c", f'source "{ROOT}/env.sh" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help 2>&1'],
    text=True, stdout=subprocess.PIPE, check=True, env=env,
).stdout
bench = (ROOT / "scripts/bench-serial.sh").read_text()
capability = next((line.strip() for line in probe.splitlines() if line.startswith("Multiple values can be given")), None)
range_capability = next((line.strip() for line in probe.splitlines() if line.startswith("or by specifying") or line.startswith("Multiple values")), None)
ranges = next((line.strip() for line in probe.splitlines() if "Ranges can be given" in line), None)
invocations = len(re.findall(r'\$LX_LLAMA_BENCH"\s+"\$\{COMMON\[@\]\}"', bench))
common_block = re.search(r"COMMON=\((.*?)\n\s*\)", bench, re.S)
common = common_block.group(1) if common_block else ""
uses_multi_value = bool(re.search(r'[,]|\b\d+-\d+(?:[+*]\d+)?\b', common))
report = {
    "executable": env.get("LX_LLAMA_BENCH", "resolved through env.sh"),
    "supports_multi_value_parameters": capability is not None,
    "supports_ranges": ranges is not None,
    "help_multi_value_statement": capability,
    "help_range_statement": ranges,
    "canonical_timed_llama_bench_invocations": invocations,
    "canonical_common_arguments_use_multi_value_or_range_sweep": uses_multi_value,
    "canonical_topology": "two independent processes" if invocations == 2 else "unexpected",
    "implication": "candidate parameter sweeps cannot share one model load or one process/runtime state in the canonical harness",
    "quality_result": "not measured by this topology audit",
}
out = ROOT / "benchmark/results/in-process-parameter-sweep-topology-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
