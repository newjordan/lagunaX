#!/usr/bin/env python3
"""Audit whether the active benchmark resolves project libraries from one build tree."""
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env = os.environ.copy()
bench = pathlib.Path(env.get("LX_LLAMA_BENCH", ""))
if not bench.is_file():
    raise SystemExit("LX_LLAMA_BENCH must name the active executable")
run = subprocess.run(["ldd", str(bench)], check=True, text=True, capture_output=True)
libs = {}
for line in run.stdout.splitlines():
    match = re.match(r"\s*(lib(?:llama|ggml)[^ ]*) => (\S+)", line)
    if match:
        libs[match.group(1)] = match.group(2)
build_dirs = sorted({str(pathlib.Path(path).parent) for path in libs.values()})
artifact = {
    "schema": "laguna.dynamic-library-provenance.v1",
    "executable": str(bench),
    "project_libraries": libs,
    "project_library_build_dirs": build_dirs,
    "mixed_project_build_provenance": len(build_dirs) > 1,
}
out = pathlib.Path(env.get("OUT", root / "results/dynamic-library-provenance-audit-20260807.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(out)
