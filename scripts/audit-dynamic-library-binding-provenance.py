#!/usr/bin/env python3
"""Audit the active Laguna benchmark's dynamic-library binding provenance."""
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(*args: str) -> str:
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout

binary = pathlib.Path(run("bash", "-lc", "source ./env.sh >/dev/null 2>&1; printf %s \"$LX_LLAMA_BENCH\"").strip())
ldd = run("bash", "-lc", 'source ./env.sh >/dev/null 2>&1; ldd "$LX_LLAMA_BENCH"')
dynamic = run("readelf", "-dW", str(binary))
harness = (ROOT / "scripts/bench-serial.sh").read_text()
resolved = {}
missing = set()
for line in ldd.splitlines():
    if "=> not found" in line:
        missing.add(line.strip().split()[0])
        continue
    match = re.search(r"=>\s+(/\S+)", line) or re.match(r"\s*(/\S+)", line)
    if match:
        path = pathlib.Path(match.group(1))
        if path.is_file():
            resolved[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "audit": "dynamic-library-binding-provenance",
    "binary": str(binary),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "resolved_library_count": len(resolved),
    "resolved_libraries_sha256": resolved,
    "missing_libraries": sorted(missing),
    "elf_has_rpath": "(RPATH)" in dynamic,
    "elf_has_runpath": "(RUNPATH)" in dynamic,
    "harness_records_ld_library_path": "LD_LIBRARY_PATH" in harness,
    "harness_records_resolved_libraries": bool(re.search(r"(?:ldd|readelf.*NEEDED)", harness)),
}
out = ROOT / "benchmark/results/dynamic-library-binding-provenance-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(out)
