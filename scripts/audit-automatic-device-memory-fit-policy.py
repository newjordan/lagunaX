#!/usr/bin/env python3
"""Audit automatic device-memory fitting policy in Laguna's canonical benchmark."""
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

fit_target = re.search(
    r"-fitt, --fit-target <MiB>\s+fit model to device memory with this margin per device in MiB \(default: (\w+)\)",
    help_text,
)
fit_ctx = re.search(
    r"-fitc, --fit-ctx <n>\s+minimum ctx size for --fit-target \(default: (\d+)\)",
    help_text,
)
source = env_text + "\n" + bench_text
configures_fit_target = bool(re.search(r"(?:^|\s)(?:-fitt|--fit-target)(?:\s|=)", source))
configures_fit_ctx = bool(re.search(r"(?:^|\s)(?:-fitc|--fit-ctx)(?:\s|=)", source))
records_fit_policy = bool(re.search(r'["\x27](?:fit_target|fit_target_mib|fit_ctx)["\x27]\s*:', bench_text))

payload = {
    "angle": "automatic_device_memory_fit_policy",
    "live_help": {
        "fit_target_supported": fit_target is not None,
        "fit_target_default": fit_target.group(1) if fit_target else None,
        "fit_target_unit": "MiB margin per device",
        "fit_ctx_supported": fit_ctx is not None,
        "fit_ctx_default": int(fit_ctx.group(1)) if fit_ctx else None,
    },
    "canonical_sources": {
        "configures_fit_target": configures_fit_target,
        "configures_fit_ctx": configures_fit_ctx,
        "metrics_record_fit_policy": records_fit_policy,
    },
}
assert fit_target is not None and fit_target.group(1) == "off"
assert fit_ctx is not None and int(fit_ctx.group(1)) == 4096
assert not configures_fit_target
assert not configures_fit_ctx
assert not records_fit_policy
out = root / "benchmark/results/automatic-device-memory-fit-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
