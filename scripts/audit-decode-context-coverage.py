#!/usr/bin/env python3
"""Audit whether Laguna decode benchmarks exercise occupied-context scaling."""
import json
import re
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()

ctx_default = re.search(r'export CTX="\$\{CTX:-(\d+)\}"', env_text)
if not ctx_default:
    raise SystemExit("CTX default not found")
common = bench_text.split("COMMON=(", 1)[1].split(")", 1)[0]
ctx_passed = bool(re.search(r'(^|\s)(-c|--ctx-size|-ctx)($|\s)', common))
decode_prompt_zero = bool(re.search(r'-p\s+0\s+-n\s+"\$LX_TG"', bench_text))

counts = Counter()
files = 0
for path in (root / "results").rglob("*.json"):
    if path.name == "decode-context-coverage-audit-20260807.json":
        continue
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    files += 1
    stack = [data]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            n_gen = value.get("n_gen", value.get("tg"))
            n_prompt = value.get("n_prompt", value.get("ps"))
            if isinstance(n_gen, (int, float)) and n_gen > 0 and isinstance(n_prompt, (int, float)):
                counts[str(int(n_prompt))] += 1
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)

report = {
    "declared_ctx_default": int(ctx_default.group(1)),
    "ctx_passed_in_serial_common_args": ctx_passed,
    "serial_decode_explicit_prompt_tokens": 0 if decode_prompt_zero else None,
    "json_files_parsed": files,
    "historical_decode_prompt_token_counts": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
    "historical_decode_records_with_nonzero_prompt": sum(v for k, v in counts.items() if int(k) > 0),
}
assert report["declared_ctx_default"] > 0
assert not report["ctx_passed_in_serial_common_args"]
assert report["serial_decode_explicit_prompt_tokens"] == 0
out = root / "results" / "decode-context-coverage-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
