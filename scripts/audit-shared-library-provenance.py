#!/usr/bin/env python3
"""Audit whether benchmark metrics identify dynamically loaded acceleration libraries."""
import json
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env = {}
for line in (root / "env.sh").read_text().splitlines():
    match = re.match(r'export (LX_BIN|LX_LLAMA_BENCH)="\$\{[^:]+:-([^}]*)\}"', line)
    if match:
        env[match.group(1)] = match.group(2).replace("$LX_BIN", env.get("LX_BIN", ""))
binary = pathlib.Path(env["LX_LLAMA_BENCH"])
ldd = subprocess.run(["ldd", str(binary)], text=True, capture_output=True, check=True).stdout
libraries = []
for line in ldd.splitlines():
    match = re.search(r'=>\s+(/\S+)\s+\(', line)
    if match:
        path = pathlib.Path(match.group(1))
        if path.exists():
            libraries.append(str(path))

metric_files = list((root / "results").rglob("*.json")) + list((root / "benchmark" / "results").rglob("*.json"))
records = 0
with_library_identity = 0
keys = re.compile(r"(shared_?librar|dso|soname|ldd|library_?hash)", re.I)
for path in metric_files:
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        continue
    records += 1
    stack = [payload]
    found = False
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            found |= any(keys.search(str(key)) for key in value)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    with_library_identity += int(found)

print(f"binary={binary}")
print(f"resolved_shared_libraries={len(libraries)}")
print(f"json_artifacts_parsed={records}")
print(f"artifacts_with_shared_library_identity={with_library_identity}")
print("shared_library_identity_recorded=" + str(with_library_identity > 0).lower())
assert binary.is_file()
assert libraries
assert records
assert with_library_identity == 0
