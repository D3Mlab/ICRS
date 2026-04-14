"""
Evaluation Orchestrator:
- Runs requested evaluation steps using StepEvaluator subclasses
- Persists per-step CSVs and a scorecard.json aggregating results
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from .types import EvalConfig
from .stores.artifacts import ArtifactStore
from .stores.reporters import write_scorecard
from .steps.step1_eval import Step1Evaluator
from .steps.step2_eval import Step2Evaluator
from .steps.step3_eval import Step3Evaluator
from .steps.step4_eval import Step4Evaluator
from .steps.step5_eval import Step5Evaluator
from .steps.step6_eval import Step6Evaluator


def run_steps(cfg: EvalConfig, steps: List[int]) -> Dict[str, Any]:
    store = ArtifactStore(cfg.artifacts_dir, cfg.manifest_path)
    per_step_csvs = {}
    for s in steps:
        if s == 1:
            ev = Step1Evaluator(cfg, store)
        elif s == 2:
            ev = Step2Evaluator(cfg, store)
        elif s == 3:
            ev = Step3Evaluator(cfg, store)
        elif s == 4:
            ev = Step4Evaluator(cfg, store)
        elif s == 5:
            ev = Step5Evaluator(cfg, store)
        elif s == 6:
            ev = Step6Evaluator(cfg, store)
        else:
            continue
        ev.evaluate()
        per_step_csvs[f"step{s}"] = cfg.reports_dir / f"step{s}_report.csv"

    scorecard_path = cfg.reports_dir / "scorecard.json"
    write_scorecard(scorecard_path, per_step_csvs, plugins={})
    return {"scorecard": str(scorecard_path), "reports": {k: str(v) for k, v in per_step_csvs.items()}}



