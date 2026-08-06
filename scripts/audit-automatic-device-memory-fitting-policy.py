#!/usr/bin/env python3
import json, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
help_text = subprocess.run(
    ["bash", "-lc", f'source "{root}/env.sh" && "$LX_LLAMA_BENCH" --help 2>&1'],
    text=True, capture_output=True, check=True,
).stdout

fit_target = re.search(r"-fitt, --fit-target <MiB>[^\n]*\(default:\s*([^\)]+)\)", help_text)
fit_ctx = re.search(r"-fitc, --fit-ctx <n>[^\n]*\(default:\s*([^\)]+)\)", help_text)
if not fit_target or not fit_ctx:
    raise SystemExit("missing live automatic device-memory fitting help contract")
combined = env_text + "\n" + bench_text
payload = {
    "angle": "automatic_device_memory_fitting_policy",
    "live_help": {
        "fit_target_supported": True,
        "fit_target_default": fit_target.group(1).strip(),
        "fit_ctx_supported": True,
        "fit_ctx_default": fit_ctx.group(1).strip(),
    },
    "canonical_sources": {
        "fit_target_override": bool(re.search(r"(^|\s)(?:-fitt|--fit-target)(?:\s|=)", combined)),
        "fit_ctx_override": bool(re.search(r"(^|\s)(?:-fitc|--fit-ctx)(?:\s|=)", combined)),
        "metrics_record_fit_policy": any(k in bench_text for k in ('"fit_target"', '"fit-target"', '"fit_ctx"', '"fit-ctx"')),
    },
}
assert payload["live_help"] == {
    "fit_target_supported": True,
    "fit_target_default": "off",
    "fit_ctx_supported": True,
    "fit_ctx_default": "4096",
}
assert not any(payload["canonical_sources"].values())
out = root / "benchmark/results/automatic-device-memory-fitting-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
