#!/usr/bin/env python3
"""Audit logical and physical batch-size policy in the active serial benchmark."""
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
help_text = subprocess.run(
    ["bash", "-lc", f'source "{ROOT / "env.sh"}"; "$LX_LLAMA_BENCH" --help'],
    cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
).stdout
batch_default = re.search(r"-b, --batch-size <n>\s+\(default: (\d+)\)", help_text)
ubatch_default = re.search(r"-ub, --ubatch-size <n>\s+\(default: (\d+)\)", help_text)
if not (batch_default and ubatch_default):
    raise SystemExit("llama-bench help did not expose expected batch controls")

def shell_default(name: str) -> int:
    match = re.search(rf'^export {name}="\$\{{{name}:-([0-9]+)\}}"', env_text, re.MULTILINE)
    if not match:
        raise SystemExit(f"env.sh does not expose a numeric {name} default")
    return int(match.group(1))

report = {
    "controls": {"batch": "-b/--batch-size", "ubatch": "-ub/--ubatch-size"},
    "executable_defaults": {"batch": int(batch_default.group(1)), "ubatch": int(ubatch_default.group(1))},
    "active_defaults": {"batch": shell_default("BBATCH"), "ubatch": shell_default("UBATCH")},
    "harness_passes_both": bool(re.search(r'-b "\$BBATCH".*-ub "\$UBATCH"', harness_text, re.DOTALL)),
    "process_environment_explicit": {k: k in os.environ for k in ("BBATCH", "UBATCH")},
}
out = ROOT / "results/batch-size-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
