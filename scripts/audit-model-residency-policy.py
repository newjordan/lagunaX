#!/usr/bin/env python3
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env = os.environ.copy()
binary = pathlib.Path(env.get("LX_LLAMA_BENCH", root / "build/bin/llama-bench"))
bench = root / "scripts/bench-serial.sh"
bench_text = bench.read_text()

run_env = env.copy()
run_env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
help_run = subprocess.run(
    [str(binary), "--help"], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, env=run_env, check=True
)
help_text = help_run.stdout
load_modes_match = re.search(r"--load-mode <([^>]+)>.*?\(default: ([^)]+)\)", help_text)
assert load_modes_match, "binary does not expose --load-mode"
load_modes = load_modes_match.group(1).split("|")
default_load_mode = load_modes_match.group(2)
common = bench_text[bench_text.index("COMMON=("):bench_text.index(")", bench_text.index("COMMON=(")) + 1]
explicit_load_mode = bool(re.search(r"(^|\s)(?:-lm|--load-mode)(\s|$)", common))
explicit_direct_io = bool(re.search(r"(^|\s)(?:-dio|--direct-io)(\s|$)", common))

json_files = list((root / "results").rglob("*.json")) + list((root / "baseline").rglob("*.json"))
parsed = 0
load_mode_records = 0
direct_io_records = 0
for path in json_files:
    try:
        obj = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    blob = json.dumps(obj).lower()
    load_mode_records += int("load_mode" in blob or '"load-mode"' in blob)
    direct_io_records += int("direct_io" in blob or '"direct-io"' in blob)

result = {
    "binary": str(binary),
    "binary_load_modes": load_modes,
    "binary_default_load_mode": default_load_mode,
    "serial_harness_explicit_load_mode": explicit_load_mode,
    "serial_harness_explicit_direct_io": explicit_direct_io,
    "parsed_json_artifacts": parsed,
    "artifacts_recording_load_mode_policy": load_mode_records,
    "artifacts_recording_direct_io_policy": direct_io_records,
}
assert {"none", "mmap", "mlock", "mmap+mlock", "dio"}.issubset(load_modes)
assert default_load_mode == "mmap"
assert not explicit_load_mode and not explicit_direct_io
assert parsed > 0
out = root / "results/model-load-mode-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
