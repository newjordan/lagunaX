#!/usr/bin/env python3
"""lx score: decode_speedup^0.75 * prefill_speedup^0.25 (mlx.fast formula)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DECODE_FLOOR = 0.95
PREFILL_FLOOR = 0.95
DECODE_EXP = 0.75
PREFILL_EXP = 0.25


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def speedup(cand: float, base: float) -> float:
    if base <= 0 or cand <= 0:
        raise ValueError(f"non-positive tok/s: cand={cand} base={base}")
    return cand / base


def score_pair(pp_c: float, tg_c: float, pp_b: float, tg_b: float,
               pp_samples=None, tg_samples=None) -> dict:
    d = speedup(tg_c, tg_b)
    p = speedup(pp_c, pp_b)
    floors_ok = d >= DECODE_FLOOR and p >= PREFILL_FLOOR
    s = (d ** DECODE_EXP) * (p ** PREFILL_EXP) if floors_ok else None
    # Measurement noise: SE of the mean from the per-rep samples llama-bench
    # already logs (stddev/sqrt(n)); propagated RSS through the 0.75/0.25
    # weights. Absent samples → null (legacy artifacts).
    def se_rel(samples):
        if not samples:
            return None
        n = len(samples)
        if n < 2:
            return None
        m = sum(samples) / n
        var = sum((x - m) ** 2 for x in samples) / (n - 1)
        sd = var ** 0.5
        return (sd / m) / (n ** 0.5) if m else None
    rel_tg, rel_pp = se_rel(tg_samples), se_rel(pp_samples)
    score_se_rel = None
    if rel_tg is not None and rel_pp is not None:
        score_se_rel = ((DECODE_EXP * rel_tg) ** 2 + (PREFILL_EXP * rel_pp) ** 2) ** 0.5
    return {
        "decode_tok_s": tg_c,
        "prefill_tok_s": pp_c,
        "baseline_decode_tok_s": tg_b,
        "baseline_prefill_tok_s": pp_b,
        "decode_speedup": d,
        "prefill_speedup": p,
        "decode_floor": DECODE_FLOOR,
        "prefill_floor": PREFILL_FLOOR,
        "floors_ok": floors_ok,
        "score": s,
        "score_se_rel": score_se_rel,
        "score_se": None if (s is None or score_se_rel is None) else s * score_se_rel,
        "gain_distinguishable": None if (s is None or score_se_rel is None) else (abs(s - 1.0) / (s * score_se_rel) if score_se_rel else None),
        "increase_pct": None if s is None else (s - 1.0) * 100.0,
        "formula": "decode_speedup^0.75 * prefill_speedup^0.25",
        "track": "serial",
        "not": "multi-slot aggregate tok/s",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    base = load(args.baseline)
    cand = load(args.candidate)

    try:
        result = score_pair(
            pp_c=float(cand["pp512"]),
            tg_c=float(cand["tg128"]),
            pp_b=float(base["pp512"]),
            tg_b=float(base["tg128"]),
            pp_samples=cand.get("pp_samples", {}).get("samples_ts") if isinstance(cand.get("pp_samples"), dict) else None,
            tg_samples=cand.get("tg_samples", {}).get("samples_ts") if isinstance(cand.get("tg_samples"), dict) else None,
        )
    except (KeyError, ValueError) as e:
        print(f"score error: {e}", file=sys.stderr)
        return 2

    result["baseline_path"] = str(args.baseline)
    result["candidate_path"] = str(args.candidate)
    result["candidate_meta"] = {
        k: cand.get(k)
        for k in ("stamp", "binary", "model", "note", "flags", "env")
        if k in cand
    }
    result["baseline_meta"] = {
        k: base.get(k) for k in ("stamp", "binary", "model", "note") if k in base
    }

    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    sys.stdout.write(text)

    if not result["floors_ok"]:
        print(
            f"FLOORS FAILED: decode={result['decode_speedup']:.4f} "
            f"prefill={result['prefill_speedup']:.4f}",
            file=sys.stderr,
        )
        return 1
    print(
        f"score={result['score']:.6f}  "
        f"increase={result['increase_pct']:+.2f}%  "
        f"dec={result['decode_speedup']:.4f}x  "
        f"pre={result['prefill_speedup']:.4f}x",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
