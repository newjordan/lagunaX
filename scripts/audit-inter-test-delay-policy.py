#!/usr/bin/env python3
"""Audit inter-test delay policy and provenance for Laguna's canonical benchmark."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
help_text = subprocess.run(
    ["bash", "-lc", f'source "{root}/env.sh" && "$LX_LLAMA_BENCH" --help 2>&1'],
    text=True,
    capture_output=True,
    check=True,
).stdout

match = re.search(r"--delay\s+<0\.\.\.N> \(seconds\)\s+delay between each test \(default: (\d+)\)", help_text)
source = env_text + "\n" + bench_text
canonical_configures_delay = bool(re.search(r"(^|\s)--delay(?:\s|=)", source))
metrics_record_delay = bool(re.search(r'["\x27](?:delay|inter_test_delay_seconds)["\x27]\s*:', bench_text))

payload = {
    "angle": "inter_test_delay_policy",
    "live_help": {
        "delay_supported": match is not None,
        "delay_unit": "seconds",
        "default_delay_seconds": int(match.group(1)) if match else None,
    },
    "canonical_sources": {
        "configures_delay": canonical_configures_delay,
        "metrics_record_delay": metrics_record_delay,
    },
}
assert match is not None
assert int(match.group(1)) == 0
assert not canonical_configures_delay
assert not metrics_record_delay
out = root / "benchmark/results/inter-test-delay-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
