#!/usr/bin/env python3
"""promote-gate.py — single owner of the score-board promotion decision.

Called by scripts/bench-serial.sh (and any other scorer) after a candidate
run. Policy, from the harness notes (commit 2af8ca5 rationale):

  same measurement geometry -> a plain score win promotes
  changed geometry          -> decode must ALSO not regress
  either way                -> the KLD quality gate must have passed

The board (LATEST_SCORE.json) is monotonic: it is only overwritten when the
candidate passes the gate AND scores >= the current board. Rejected candidates
are recorded at --rejected so the last denial stays inspectable.

Usage:
  scripts/promote-gate.py --candidate <score.json> --board <LATEST_SCORE.json> \
      [--kld <LATEST_KLD.json>] [--rejected <LATEST_SCORE_REJECTED.json>]
"""

import argparse
import json
import pathlib
import shutil
import sys

# Flags that define what the pp/tg speedup terms mean. Binary/model/note are
# deliberately excluded: a new build of the same geometry is a normal candidate.
GEOMETRY_KEYS = (
    "ngl", "ubatch", "bbatch", "ctx", "ctk", "ctv",
    "threads", "threads_batch",
)


def load(path):
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        print(f"promote-gate: unreadable {path}: {e}", file=sys.stderr)
        return None


def geometry(doc):
    flags = (doc or {}).get("candidate_meta", {}).get("flags") or {}
    return {k: flags[k] for k in GEOMETRY_KEYS if k in flags}


def kld_passed(kld_doc):
    if kld_doc is None:
        return False, "no KLD receipt at --kld (run scripts/quality-gate-kld.sh)"
    if not kld_doc.get("passed"):
        return False, f"KLD gate not passed (receipt stamp {kld_doc.get('stamp_utc', '?')})"
    return True, ""


def main():
    ap = argparse.ArgumentParser(description="lx promotion gate: score-board decision owner")
    ap.add_argument("--candidate", required=True, help="candidate score.json from this run")
    ap.add_argument("--board", required=True, help="board LATEST_SCORE.json")
    ap.add_argument("--kld", default=None, help="LATEST_KLD.json quality receipt")
    ap.add_argument("--rejected", default=None, help="where to copy a rejected candidate")
    args = ap.parse_args()

    cand = load(args.candidate)
    if cand is None:
        print("promote-gate: no readable candidate score — NOT promoted", file=sys.stderr)
        return 1
    if not cand.get("floors_ok"):
        print("promote-gate: candidate floors not ok — NOT promoted", file=sys.stderr)
        if args.rejected:
            shutil.copyfile(args.candidate, args.rejected)
        return 1

    ok_kld, why = kld_passed(load(args.kld) if args.kld else None)
    if not ok_kld:
        print(f"promote-gate: {why} — NOT promoted", file=sys.stderr)
        if args.rejected:
            shutil.copyfile(args.candidate, args.rejected)
        return 1

    board = load(args.board)
    if board is None:
        # No board yet: first verified candidate establishes it.
        shutil.copyfile(args.candidate, args.board)
        print(f"promote-gate: board initialized at {cand['score']}")
        return 0

    cand_score = float(cand["score"])
    board_score = float(board.get("score") or 0.0)
    if cand_score < board_score:
        msg = f"promote-gate: {cand_score} < board {board_score} — NOT downgraded"
        verdict = 1
    elif geometry(cand) != geometry(board):
        cand_dec = float(cand["decode_speedup"])
        board_dec = float(board.get("decode_speedup") or 0.0)
        if cand_dec < board_dec:
            msg = (f"promote-gate: geometry changed and decode regressed "
                   f"({cand_dec:.4f} < {board_dec:.4f}) — NOT promoted")
            verdict = 1
        else:
            msg = (f"promote-gate: geometry changed but decode held "
                   f"({cand_dec:.4f} >= {board_dec:.4f}) → promoted at {cand_score}")
            verdict = 0
    else:
        msg = f"promote-gate: {cand_score} >= board {board_score} → promoted"
        verdict = 0

    print(msg)
    if verdict == 0:
        shutil.copyfile(args.candidate, args.board)
    elif args.rejected:
        shutil.copyfile(args.candidate, args.rejected)
    return verdict


if __name__ == "__main__":
    sys.exit(main())
