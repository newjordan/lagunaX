#!/usr/bin/env python3
"""Audit whether Laguna's declared context size reaches llama-bench."""
import json, re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
ctx = int(re.search(r'export CTX="\$\{CTX:-(\d+)\}"', env).group(1))
common = bench[bench.index("COMMON=("):bench.index(")", bench.index("COMMON=(")) + 1]
result = {
    "policy": "benchmark-context-size",
    "declared_ctx": ctx,
    "ctx_passed_to_llama_bench": bool(re.search(r'(^|\s)(-c|--ctx-size)($|\s)', common)),
    "ctx_recorded_in_metrics": '"ctx": int("$CTX")' in bench,
    "effective_window": {"prompt_tokens": 512, "generated_tokens": 128},
    "finding": "CTX is declared and recorded but is not passed to either llama-bench invocation; the benchmark is therefore not evidence for an 8192-token runtime context policy.",
}
out = root / "results/context-size-provenance-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
assert result["declared_ctx"] == 8192
assert not result["ctx_passed_to_llama_bench"]
assert result["ctx_recorded_in_metrics"]
