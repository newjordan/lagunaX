#!/usr/bin/env python3
"""Audit compiler/build-mode provenance for the canonical benchmark executable."""
import json, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
probe = subprocess.run(
    ["bash", "-lc", f'source "{root}/env.sh" && printf "%s" "$LX_LLAMA_BENCH"'],
    text=True, capture_output=True, check=True,
)
exe = Path(probe.stdout).resolve()
file_out = subprocess.run(["file", str(exe)], text=True, capture_output=True, check=True).stdout.strip()
comment = subprocess.run(["readelf", "-p", ".comment", str(exe)], text=True, capture_output=True, check=True).stdout
sections = subprocess.run(["readelf", "-S", str(exe)], text=True, capture_output=True, check=True).stdout
canonical = env + "\n" + bench
payload = {
    "angle": "compiler_build_mode_provenance",
    "executable": str(exe),
    "binary": {
        "file_identity": file_out,
        "not_stripped": "not stripped" in file_out,
        "has_symbol_table": ".symtab" in sections,
        "compiler_comments": [line.strip() for line in comment.splitlines() if "GCC:" in line or "Compiler" in line],
    },
    "canonical_provenance": {
        "records_build_type": "CMAKE_BUILD_TYPE" in canonical,
        "records_compiler_flags": "CMAKE_CXX_FLAGS" in canonical,
        "records_compiler_identity": "compiler_identity" in bench,
        "records_stripping_policy": "stripping_policy" in bench,
    },
}
assert payload["binary"]["not_stripped"]
assert payload["binary"]["has_symbol_table"]
assert any("GCC:" in x for x in payload["binary"]["compiler_comments"])
assert any("oneAPI" in x for x in payload["binary"]["compiler_comments"])
assert not any(payload["canonical_provenance"].values())
out = root / "results" / "compiler-build-mode-provenance-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
