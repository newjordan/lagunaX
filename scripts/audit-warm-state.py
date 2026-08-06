#!/usr/bin/env python3
"""Audit cold-start versus warm-state generation throughput."""

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text())
    comparable = [
        row for row in rows
        if row.get("ok") and row.get("predicted_n") == 96 and "tg_tps" in row
    ]
    if len(comparable) < 2:
        raise SystemExit("need at least two successful 96-token runs")

    cold = float(comparable[0]["tg_tps"])
    warm_values = [float(row["tg_tps"]) for row in comparable[1:]]
    warm = mean(warm_values)
    payload = {
        "source": str(args.input),
        "comparison": "first successful 96-token run vs subsequent successful 96-token runs",
        "cold_tg_tps": cold,
        "warm_runs": len(warm_values),
        "warm_mean_tg_tps": warm,
        "warm_over_cold_ratio": warm / cold,
        "warm_over_cold_percent": (warm / cold - 1.0) * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
