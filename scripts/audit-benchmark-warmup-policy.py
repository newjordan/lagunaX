#!/usr/bin/env python3
import json, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
cmd = f'source {root}/env.sh && "$LX_LLAMA_BENCH" --help 2>&1'
help_text = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=True).stdout

supported = "--no-warmup" in help_text
help_declares_warmup = bool(re.search(r"--no-warmup\s+skip warmup runs before benchmarking", help_text))
source = env_text + "\n" + bench_text
canonical_disables_warmup = bool(re.search(r"(^|\s)--no-warmup(\s|$)", source))
metrics_record_warmup = bool(re.search(r'["\x27]warmup["\x27]\s*:', bench_text))
process = subprocess.run(["bash", "-lc", "env | grep -Ei '(^|_)(NO_)?WARMUP=' || true"], text=True, capture_output=True, check=True)
active_warmup_environment = [line for line in process.stdout.splitlines() if line]

payload = {
    "angle": "benchmark_warmup_policy",
    "live_help": {
        "no_warmup_supported": supported,
        "warmup_enabled_by_default": help_declares_warmup,
    },
    "canonical_sources": {
        "disables_warmup": canonical_disables_warmup,
        "metrics_record_warmup_policy": metrics_record_warmup,
    },
    "process_environment": {"warmup_controls": active_warmup_environment},
}
assert supported and help_declares_warmup
assert not canonical_disables_warmup
assert not metrics_record_warmup
assert not active_warmup_environment
out = root / "benchmark/results/benchmark-warmup-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
