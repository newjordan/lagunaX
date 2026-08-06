#!/usr/bin/env python3
"""Audit whether Laguna's numeric flash-attention control matches the active CLI contract."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()

shell = subprocess.run(
    ["bash", "-lc", "source ./env.sh; printf '%s\\n' \"$LX_LLAMA_BENCH\"; \"$LX_LLAMA_BENCH\" --help 2>&1"],
    cwd=root, text=True, capture_output=True, check=True,
).stdout.splitlines()
binary, help_text = shell[0], "\n".join(shell[1:])
contract = re.search(r"-fa, --flash-attn <([^>]+)>\s+\(default: ([^)]+)\)", help_text)
configured = re.search(r'export FA="\$\{FA:-(.+?)\}"', env_text)
assert contract and configured
accepted, cli_default = contract.groups()
configured_default = configured.group(1)
passes_fa = bool(re.search(r'-fa\s+"\$FA"', bench_text))
records_integer = '"flash_attn": int("$FA")' in bench_text
result = {
    "binary": binary,
    "live_cli": {"accepted_values": accepted.split("|"), "default": cli_default},
    "laguna": {
        "configured_default": configured_default,
        "passes_configured_value": passes_fa,
        "records_as_integer": records_integer,
        "default_is_documented_cli_value": configured_default in accepted.split("|"),
    },
}
assert result["live_cli"] == {"accepted_values": ["on", "off", "auto"], "default": "auto"}
assert result["laguna"] == {
    "configured_default": "-1", "passes_configured_value": True,
    "records_as_integer": True, "default_is_documented_cli_value": False,
}
out = root / "results" / "flash-attention-cli-contract-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
