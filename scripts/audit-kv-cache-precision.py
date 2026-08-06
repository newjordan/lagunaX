#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for path in sorted((root / 'results').rglob('*.json')):
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        continue
    if not isinstance(obj, dict):
        continue
    flags = obj.get('flags')
    if not isinstance(flags, dict):
        continue
    ctk, ctv = flags.get('ctk'), flags.get('ctv')
    if ctk is None and ctv is None:
        continue
    rows.append((str(path.relative_to(root)), str(ctk), str(ctv)))

pairs = Counter((ctk, ctv) for _, ctk, ctv in rows)
baseline = json.loads((root / 'baseline/baseline.json').read_text())
candidate = json.loads((root / 'results/LATEST_SCORE.json').read_text())
score = float(candidate['score'])
target = 2.0
report = {
    'records_with_kv_precision': len(rows),
    'kv_precision_pairs': {f'{k}/{v}': n for (k, v), n in sorted(pairs.items())},
    'baseline_kv_precision': [baseline['flags']['ctk'], baseline['flags']['ctv']],
    'candidate_kv_precision': [candidate['candidate_meta']['flags']['ctk'], candidate['candidate_meta']['flags']['ctv']],
    'baseline_decode_tok_s': baseline['tg128'],
    'candidate_decode_tok_s': candidate['decode_tok_s'],
    'baseline_prefill_tok_s': baseline['pp512'],
    'candidate_prefill_tok_s': candidate['prefill_tok_s'],
    'literal_target_score': target,
    'verified_score': score,
    'literal_target_delta': target - score,
    'quality_result': 'not measured by this provenance audit',
}
out = root / 'results/kv-cache-precision-audit-20260807.json'
out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print(out)
