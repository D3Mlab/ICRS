from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import os
from orchestrator.pipeline import ICRSPipeline
from utils.cache_manager import CacheManager


def main() -> None:
    p = argparse.ArgumentParser(description="Run XRIR pipeline and evaluate outputs")
    # Pipeline inputs (optional; auto-fills missing steps)
    p.add_argument("--image")
    p.add_argument("--crops_dir")
    p.add_argument("--segment_rankings")
    p.add_argument("--docs_json")
    p.add_argument("--linker_cfg", default="configs/linker.yaml")
    p.add_argument("--item_recommendation_cfg", default="configs/item_recommendation.yaml")
    p.add_argument("--query", default="", help="Natural language query for relevance filtering and snippet ranking")
    p.add_argument("--api_key", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", default="vit_b")
    p.add_argument("--exhaustive", action="store_true")

    # Linker outputs (optional). If omitted, will default to dataset base_dir under links/ and metadata/
    p.add_argument("--out_links")
    p.add_argument("--out_meta")

    # Evaluation + Step6 snippet ranking
    p.add_argument("--run_eval", action="store_true")
    p.add_argument("--eval_steps", default="2,4", help="comma-separated steps to evaluate, e.g., 1,2,3,4")
    p.add_argument("--data_root", default=".")
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--manifest", default="artifacts/manifest.json")
    p.add_argument("--run_snippet_rank", action="store_true")
    p.add_argument("--step6_config", default="configs/step6.yaml")

    args = p.parse_args()

    # Resolve paths
    image = Path(args.image) if args.image else None
    crops_dir = Path(args.crops_dir) if args.crops_dir else None
    if crops_dir is not None:
        print('Using crops directory: ', crops_dir)
    rankings_json = Path(args.segment_rankings) if args.segment_rankings else None
    if rankings_json is not None:
        print('Using precomputed segment rankings: ', rankings_json)
    docs_json = Path(args.docs_json) if args.docs_json else None
    if docs_json is not None:
        print('Using metadata JSON: ', docs_json)
    out_links = Path(args.out_links) if args.out_links else None
    out_meta = Path(args.out_meta) if args.out_meta else None
    if out_links is not None and out_links.exists():
        print('Using precomputed Metadata links: ', out_links)

    # Resolve query input (string or JSON path). If JSON, extract first query.text for relevance filtering.
    query_arg = args.query or ""
    query_text = query_arg
    try:
        cand = Path(query_arg)
        if query_arg and cand.suffix.lower() == ".json" and cand.exists():
            try:
                qobj = json.loads(cand.read_text())
                if isinstance(qobj, dict) and isinstance(qobj.get("queries"), list) and qobj["queries"]:
                    first = qobj["queries"][0]
                    if isinstance(first, dict) and first.get("text"):
                        query_text = str(first["text"])
            except Exception:
                pass
    except Exception:
        pass

    pipe = ICRSPipeline()
    pipe.run(
        image=image,
        crops_dir=crops_dir,
        item_recommendation_json=rankings_json,
        docs_json=docs_json,
        out_links=out_links,
        out_meta=out_meta,
        api_key=os.environ.get("OPENAI_API_KEY"),
        linker_cfg=Path(args.linker_cfg) if args.linker_cfg else None,
        item_recommendation_cfg=Path(args.item_recommendation_cfg) if args.item_recommendation_cfg else None,
        query=query_text,
        device=args.device,
        model=args.model,
        exhaustive=args.exhaustive,
    )

if __name__ == "__main__":
    main()


