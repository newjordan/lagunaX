#!/usr/bin/env python3
"""Audit host/device offload controls exposed by the active llama-bench."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
env.setdefault("LX_ROOT", str(root))
bin_match = re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text)
if not bin_match:
    raise SystemExit("cannot resolve LX_BIN")
binary = Path(env.get("LX_LLAMA_BENCH", str(Path(env.get("LX_BIN", bin_match.group(1))) / "llama-bench")))
probe = f'source "{root / "env.sh"}" >/dev/null; "$LX_LLAMA_BENCH" --help'
help_text = subprocess.run(["bash", "-c", probe], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True, env=env).stdout
controls = {
    "no_kv_offload": {"flag": "--no-kv-offload", "default": "0"},
    "no_op_offload": {"flag": "--no-op-offload", "default": "0"},
    "no_host": {"flag": "--no-host", "default": "0"},
}
for item in controls.values():
    line = next((line.strip() for line in help_text.splitlines() if item["flag"] in line), None)
    if line is None:
        raise SystemExit(f"missing live contract for {item['flag']}")
    item["help"] = line
    item["supported"] = True
    item["live_default_confirmed"] = f"(default: {item['default']})" in line
    item["configured_by_laguna"] = item["flag"] in env_text or item["flag"] in bench_text

artifact = {
    "binary": str(binary),
    "controls": controls,
    "all_supported": all(x["supported"] for x in controls.values()),
    "all_defaults_confirmed": all(x["live_default_confirmed"] for x in controls.values()),
    "laguna_configures_any": any(x["configured_by_laguna"] for x in controls.values()),
    "laguna_records_any": any(k in bench_text for k in ("no_kv_offload", "no_op_offload", "no_host")),
}
out = root / "results" / "host-device-offload-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
