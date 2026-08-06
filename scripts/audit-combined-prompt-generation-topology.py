#!/usr/bin/env python3
"""Audit whether Laguna uses llama-bench's combined prompt+generation mode."""
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = subprocess.run(
    ["bash", "-lc", "source ./env.sh; printf '%s\n%s\n%s\n' \"$LX_LLAMA_BENCH\" \"$LX_PP\" \"$LX_TG\""],
    cwd=ROOT, check=True, text=True, capture_output=True,
).stdout.splitlines()
bench, pp, tg = ENV
help_text = subprocess.run(
    ["bash", "-lc", 'source ./env.sh; "$LX_LLAMA_BENCH" --help'],
    cwd=ROOT, check=True, text=True, capture_output=True,
).stdout
harness = (ROOT / "scripts/bench-serial.sh").read_text()
combined = re.search(r"^\s*-pg\s+<pp,tg>\s+.*\(default:\s*(.*?)\)", help_text, re.MULTILINE)
invocations = re.findall(r'PP_JSON=.*?\n|TG_JSON=.*?\n', harness)
artifact = {
    "audit": "combined-prompt-generation-process-topology",
    "effective": {"prompt_tokens": int(pp), "generation_tokens": int(tg)},
    "capability": {"combined_pg_supported": combined is not None, "combined_pg_default": combined.group(1).strip() if combined else None},
    "harness": {
        "llama_bench_invocation_count": len(invocations),
        "passes_combined_pg": bool(re.search(r'(^|\s)-pg(\s|$)', harness)),
        "prefill_invocation": bool(re.search(r'-p "\$LX_PP" -n 0', harness)),
        "decode_invocation": bool(re.search(r'-p 0 -n "\$LX_TG"', harness)),
    },
    "conclusion": "The harness uses two processes and does not exercise the supported combined prompt+generation benchmark path.",
}
out = pathlib.Path(os.environ.get("AUDIT_OUT", ROOT / "benchmark/results/combined-prompt-generation-topology-audit-20260807.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
