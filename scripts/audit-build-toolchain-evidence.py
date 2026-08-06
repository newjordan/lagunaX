#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
latest = json.loads((root/'results/LATEST_SCORE.json').read_text())
champ_metrics = json.loads((root/'results/20260731T141436Z/metrics.json').read_text())
champ_score = json.loads((root/'results/20260731T141436Z/score.json').read_text())
latest_metrics = json.loads(Path(latest['candidate_path']).read_text())
provenance_keys = ['source_commit','source_dirty','compiler','compiler_version','cmake_build_type','cmake_cache_sha256','build_flags','linked_libraries']

def missing(d): return [k for k in provenance_keys if k not in d]
champ = float(champ_score['score'])
active = float(latest['score'])
out = {
  'audit': 'build-toolchain-and-active-evidence-reconciliation',
  'champion': {'metrics_path':'results/20260731T141436Z/metrics.json','score_path':'results/20260731T141436Z/score.json','score':champ,'missing_build_provenance':missing(champ_metrics)},
  'active_latest': {'metrics_path':'results/20260731T172351Z/metrics.json','score':active,'missing_build_provenance':missing(latest_metrics)},
  'reconciliation': {
    'latest_is_champion': active == champ,
    'latest_regression_vs_champion_pct': (active/champ-1)*100,
    'literal_target_score': 2.0,
    'champion_absolute_target_gap': 2.0-champ,
    'champion_multiplicative_improvement_needed_pct': (2.0/champ-1)*100,
    'active_absolute_target_gap': 2.0-active,
    'active_multiplicative_improvement_needed_pct': (2.0/active-1)*100,
    'artifact_bound_quality_guard_in_champion_metrics': any(k in champ_metrics for k in ('quality','quality_guard','golden')),
    'artifact_bound_quality_guard_in_latest_metrics': any(k in latest_metrics for k in ('quality','quality_guard','golden')),
  }
}
path=root/'results/build-toolchain-evidence-audit-20260807.json'
path.write_text(json.dumps(out,indent=2)+'\n')
print(path)
