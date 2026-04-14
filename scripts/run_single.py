#!/usr/bin/env python3

"""
Iterate through every conversation ID defined in data/fashion/conversation/by_tag.json
and invoke run.sh for each one by setting the conversation_ID environment variable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
dataset = 'retail'
RUN_SCRIPT = REPO_ROOT / "scripts/run.sh"
COV_JSON = REPO_ROOT / f"data/{dataset}/conversation/agg/pre_gt.json"

def load_conversation_ids(source: Path) -> list[str]:
    data = json.loads(source.read_text())
    if isinstance(data, dict):
        return list(data.keys())
    if isinstance(data, list):
        ids: list[str] = []
        for item in data:
            if isinstance(item, dict) and "conversation_id" in item:
                ids.append(str(item["conversation_id"]))
        return ids
    raise ValueError(f"Unsupported conversation file format: {source}")


def main() -> None:
    # arg for dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion")
    args = parser.parse_args()
    dataset = args.dataset
    COV_JSON = REPO_ROOT / f"data/{dataset}/conversation/agg/pre_gt.json"
    if not RUN_SCRIPT.exists():
        print(f"run.sh not found at {RUN_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not COV_JSON.exists():
        print(f"conversation file not found at {COV_JSON}", file=sys.stderr)
        sys.exit(1)

    conversation_ids = load_conversation_ids(COV_JSON)
    if not conversation_ids:
        print("No conversation IDs discovered; exiting.", file=sys.stderr)
        sys.exit(1)

    total = len(conversation_ids)
    failures: list[tuple[str, int]] = []

    for idx, qid in enumerate(conversation_ids, start=1):
        # if dataset == 'fashion' and any(c notin qid for c in ['c50', 'c51', 'c52', 'c53', 'c54', 'c55', 'c56', 'c57', 'c58', 'c59']):
        #     continue
        # if idx < 29:
        #     continue
        print(f"[{idx}/{total}] Running pipeline for conversation '{qid}'", flush=True)
        env = os.environ.copy()
        env["QUERY_ID"] = qid
        env["DATASET"] = dataset
        process = subprocess.run(
            ["/bin/bash", str(RUN_SCRIPT)],
            env=env,
            cwd=str(REPO_ROOT),
        )
        if process.returncode != 0:
            print(
                f"[{idx}/{total}] run.sh failed for conversation '{qid}' (exit code {process.returncode}). Continuing.",
                file=sys.stderr,
            )
            failures.append((qid, process.returncode))
            continue
        print(f"[{idx}/{total}] Completed conversation '{qid}'\n", flush=True)

    if failures:
        print("\nFailures:")
        for qid, code in failures:
            print(f"  conversation_id={qid} exit_code={code}")
    print(f"Finished running {total} queries with {len(failures)} failures.")


if __name__ == "__main__":
    main()

