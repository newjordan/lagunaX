#!/usr/bin/env python3
import json, re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = [root/'scripts/golden-smoke.sh', root/'scripts/proof-suite.sh', root/'scripts/treebeard-parity-suite.sh']
rows = []
for p in files:
    text = p.read_text()
    rows.append({
        'path': str(p.relative_to(root)),
        'cache_prompt_false': len(re.findall(r'["\']cache_prompt["\']\s*:\s*False', text)),
        'cache_prompt_true': len(re.findall(r'["\']cache_prompt["\']\s*:\s*True', text)),
    })
report = {
    'direction': 'prompt-prefix cache reuse policy',
    'files': rows,
    'explicit_false_total': sum(r['cache_prompt_false'] for r in rows),
    'explicit_true_total': sum(r['cache_prompt_true'] for r in rows),
    'quality_result': 'not measured; provenance audit only',
}
out = root/'results/prompt-cache-policy-audit-20260807.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(out)
