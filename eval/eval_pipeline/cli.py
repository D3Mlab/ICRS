"""
CLI scaffold:
- Subcommands: step1, step2, step3, step4, all
- Common flags: --report_dir, --artifacts_dir, --manifest, --plugins_dir, --extra_metrics, --metrics_cfg, --data_root
- Delegates to orchestrator and StepEvaluators
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .types import EvalConfig
from .orchestrator import run_steps


def main():
    p = argparse.ArgumentParser(description="Evaluation pipeline CLI")
    p.add_argument("cmd", choices=["step1", "step2", "step3", "step4", "step5", "step6", "all"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--artifacts_dir", default="artifacts")
    p.add_argument("--reports_dir", default="reports")
    p.add_argument("--manifest", default="artifacts/manifest.json")
    p.add_argument("--plugins_dir", default="eval_pipeline/metrics_plugins")
    p.add_argument("--extra_metrics", default="")
    p.add_argument("--metrics_cfg", default="configs/metrics.yaml")
    args = p.parse_args()

    metrics_cfg: Dict[str, Any] = {}
    cfg_path = Path(args.metrics_cfg)
    if cfg_path.exists():
        try:
            if cfg_path.suffix.lower() in {".yaml", ".yml"}:
                import yaml  # type: ignore

                metrics_cfg = yaml.safe_load(cfg_path.read_text()) or {}
            else:
                metrics_cfg = json.loads(cfg_path.read_text())
        except Exception:
            metrics_cfg = {}

    cfg = EvalConfig(
        data_root=Path(args.data_root),
        artifacts_dir=Path(args.artifacts_dir),
        reports_dir=Path(args.reports_dir),
        manifest_path=Path(args.manifest),
        plugins_dir=Path(args.plugins_dir) if args.plugins_dir else None,
        extra_metrics=[x for x in args.extra_metrics.split(",") if x],
        metrics_cfg=metrics_cfg,
    )

    if args.cmd == "all":
        res = run_steps(cfg, [1, 2, 3, 4, 5, 6])
    else:
        step = int(args.cmd.replace("step", ""))
        res = run_steps(cfg, [step])


if __name__ == "__main__":
    main()



