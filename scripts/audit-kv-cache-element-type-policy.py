#!/usr/bin/env python3
"""Audit active KV-cache element types and historical coverage."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()

def env_default(name: str) -> str:
    m = re.search(rf'export {name}="\$\{{{name}:-([^}}]+)\}}"', env_text)
    if not m:
        raise SystemExit(f"missing {name} default")
    return m.group(1)

bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    shell = subprocess.run(
        ["bash", "-lc", "source ./env.sh >/dev/null 2>&1; printf '%s' \"$LX_LLAMA_BENCH\""],
        cwd=root, text=True, capture_output=True, check=True)
    bench = shell.stdout
help_run = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; exec "$LX_LLAMA_BENCH" --help'],
    cwd=root, text=True, capture_output=True)
help_text = help_run.stdout or help_run.stderr
if not help_text:
    raise SystemExit(f"llama-bench --help produced no output (exit {help_run.returncode})")

def help_default(flag: str) -> str:
    m = re.search(rf'{re.escape(flag)}[^\n]*\(default: ([^)]+)\)', help_text)
    if not m:
        raise SystemExit(f"missing help default for {flag}")
    return m.group(1)

artifacts = list((root / "results").rglob("*.json"))
mentions = 0
parsed = 0
for path in artifacts:
    try:
        data = json.loads(path.read_text())
        parsed += 1
    except Exception:
        continue
    text = json.dumps(data)
    if any(token in text for token in ('"ctk"', '"ctv"', 'cache-type-k', 'cache-type-v')):
        mentions += 1

result = {
    "policy": "kv_cache_element_types",
    "executable": bench,
    "executable_defaults": {"ctk": help_default("--cache-type-k"), "ctv": help_default("--cache-type-v")},
    "active": {"ctk": env_default("CTK"), "ctv": env_default("CTV")},
    "harness_passes": {"ctk": '-ctk "$CTK"' in harness_text, "ctv": '-ctv "$CTV"' in harness_text},
    "precision_reduction_active": env_default("CTK") != "f16" or env_default("CTV") != "f16",
    "historical_json": {"parsed": parsed, "with_type_mentions": mentions},
}
out = root / "results/kv-cache-element-type-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
assert result["executable_defaults"] == {"ctk": "f16", "ctv": "f16"}
assert result["active"] == {"ctk": "f16", "ctv": "f16"}
assert result["harness_passes"] == {"ctk": True, "ctv": True}
assert result["precision_reduction_active"] is False
print(out)
