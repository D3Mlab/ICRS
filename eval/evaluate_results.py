#!/usr/bin/env python3

"""
Evaluate results from a given result path.

This script takes a result path (e.g., results/fashion/object_ranker/UMBRELLA_LLM)
and automatically determines which evaluation steps to run based on the module name.
It extracts metadata from registry.json or output files and runs the evaluation
using the same logic as main.py.
"""

from __future__ import annotations
import numpy as np
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from eval_pipeline.types import EvalConfig
from eval_pipeline.orchestrator import run_steps

# Cache for ground truth snippet data (keyed by dataset)
_gt_snippet_cache: Dict[str, Dict[str, Any]] = {}

# Mapping from module names to evaluation steps
MODULE_TO_STEPS = {
    "object_ranker": [5],
    "snippet_ranker": [6],
    "linker": [4],
    "segment_describer": [2],
    "segmenter": [1],
}

# Mapping from module names to method/model key prefixes
MODULE_METADATA_KEYS = {
    "object_ranker": ("object_ranker_method", "object_ranker_model"),
    "snippet_ranker": ("snippet_ranker_method", "snippet_ranker_model"),
    "linker": ("linker_method", "linker_model"),
    "segment_describer": ("describer_method", "describer_model"),
    "segmenter": ("segmenter_method", "segmenter_model"),
}


def parse_result_path(result_path: Path) -> tuple[str, str, str, str, str, Path]:
    """
    Parse result path to extract dataset, data, module, model, method, and result directory.
    
    Expected format: results/{dataset}/{data}/{module}/{model}/{method}
    or: {base}/results/{dataset}/{data}/{module}/{model}/{method}
    
    Returns:
        (dataset_name, data_name, module_name, model_name, method_name, result_dir)
    """
    # Normalize path - if it's a file, use its parent
    if result_path.is_file():
        result_dir = result_path.parent
    else:
        result_dir = result_path
    
    parts = [p for p in result_dir.parts if p]
    
    # Find "results" in the path
    try:
        results_idx = next(i for i, p in enumerate(parts) if p == "results")
        # After "results" should be: dataset, data, module, model, method
        if results_idx + 5 <= len(parts):
            dataset = parts[results_idx + 1]
            data = parts[results_idx + 2]
            module = parts[results_idx + 3]
            model = parts[results_idx + 4]
            method = parts[results_idx + 5] if results_idx + 5 < len(parts) else parts[-1]
            return dataset, data, module, model, method, result_dir
        # Fallback: try old format without data/model: results/{dataset}/{module}/{method}
        elif results_idx + 3 <= len(parts):
            dataset = parts[results_idx + 1]
            data = ""  # No data in old format
            module = parts[results_idx + 2]
            model = ""  # No model in old format
            method = parts[results_idx + 3] if results_idx + 3 < len(parts) else parts[-1]
            return dataset, data, module, model, method, result_dir
    except StopIteration:
        pass
    
    # Fallback: assume last parts are dataset, data, module, model, method
    if len(parts) >= 5:
        dataset = parts[-5]
        data = parts[-4]
        module = parts[-3]
        model = parts[-2]
        method = parts[-1]
        return dataset, data, module, model, method, result_dir
    elif len(parts) >= 3:
        # Old format without data/model
        dataset = parts[-3]
        data = ""
        module = parts[-2]
        model = ""
        method = parts[-1]
        return dataset, data, module, model, method, result_dir
    
    raise ValueError(f"Cannot parse result path: {result_path}. Expected format: results/{{dataset}}/{{data}}/{{module}}/{{model}}/{{method}}")


def load_metadata_from_registry(registry_path: Path, method_name: str) -> Dict[str, Any]:
    """Load metadata from registry.json for the given method."""
    if not registry_path.exists():
        return {}
    
    try:
        registry = json.loads(registry_path.read_text())
        methods = registry.get("methods", {})
        method_entry = methods.get(method_name, {})
        
        # Extract metadata from outputs
        outputs = method_entry.get("outputs", {})
        metadata = {}
        
        # Look for metadata in any output
        for output_name, output_data in outputs.items():
            if isinstance(output_data, dict):
                output_meta = output_data.get("metadata", {})
                if output_meta:
                    metadata.update(output_meta)
                    break
        
        # Also get method and model from the entry itself
        if not metadata.get("method"):
            metadata["method"] = method_entry.get("method") or method_name
        if not metadata.get("model"):
            metadata["model"] = method_entry.get("model")
        
        return metadata
    except Exception as e:
        print(f"Warning: Could not load metadata from registry: {e}")
        return {}


def find_output_file(result_dir: Path, module: str, method: str, model: str = "") -> Optional[Path]:
    """Find the main output file for a module.
    
    With new structure: results/{dataset}/{data}/{module}/{model}/{method}
    Files may be in result_dir or in subdirectories within result_dir.
    """
    if module == "object_ranker":
        # Look for rankings files - could be in result_dir or in query-specific subdirectories
        # Try query-specific files first (e.g., c33_b_3_json_rankings.json)
        candidates = []
        # Check for query-specific files in result_dir
        if result_dir.is_dir():
            for item in result_dir.iterdir():
                if item.is_file() and item.name.endswith("_rankings.json"):
                    candidates.append(item)
        # Also check for method-level rankings file
        candidates.extend([
            result_dir / f"{method}_rankings.json",
            result_dir.parent / f"{method}_rankings.json",
        ])
    elif module == "snippet_ranker":
        # Look for {method}_snippets_json.json
        candidates = []
        if result_dir.is_dir():
            for item in result_dir.iterdir():
                if item.is_file() and item.name.endswith("_snippets_json.json"):
                    candidates.append(item)
        candidates.extend([
            result_dir / f"{method}_snippets_json.json",
            result_dir.parent / f"{method}_snippets_json.json",
        ])
    elif module == "linker":
        # Look for {method}_links.json
        candidates = [
            result_dir / f"{method}_links.json",
            result_dir.parent / f"{method}_links.json",
        ]
    elif module == "segment_describer":
        # Look for {method}_entries.json
        candidates = [
            result_dir / f"{method}_entries.json",
            result_dir.parent / f"{method}_entries.json",
        ]
    else:
        return None
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    return None


def load_metadata_from_output(output_path: Path) -> Dict[str, Any]:
    """Try to extract metadata from the output file itself."""
    if not output_path.exists():
        return {}
    
    try:
        data = json.loads(output_path.read_text())
        # Some output files have metadata at the top level
        metadata = {}
        if isinstance(data, dict):
            # Check for common metadata keys
            for key in ["query_id", "query", "method", "model", "meta_json"]:
                if key in data:
                    metadata[key] = data[key]
        return metadata
    except Exception:
        return {}


def infer_dataset_root(result_dir: Path, dataset_name: str, data_root: Path) -> Path:
    """Infer the dataset root directory."""
    # Try to find data/{dataset} directory
    candidates = [
        data_root / "data" / dataset_name,
        result_dir.parent.parent.parent / "data" / dataset_name,
        Path("data") / dataset_name,
    ]
    
    for candidate in candidates:
        if candidate.exists():
            return candidate
    
    # Fallback: use data_root/data/{dataset_name}
    return data_root / "data" / dataset_name


def load_prediction_scores(pred_file: Path, query_id: str) -> Dict[str, float]:
    """Load prediction scores from a ranking JSON file for a specific query_id."""
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    data = json.loads(pred_file.read_text())
    
    # Extract scores - the file structure can vary
    scores: Dict[str, float] = {}
    
    # Check if this is a single query file with scores directly
    if isinstance(data, dict):
        # First try to get scores directly
        scores_dict = data.get("scores", {})
        if isinstance(scores_dict, dict) and scores_dict:
            # File has scores directly
            for doc_id, score in scores_dict.items():
                try:
                    scores[str(doc_id)] = float(score)
                except (TypeError, ValueError):
                    continue
        
        # If no scores found, try rankings list
        if not scores and "rankings" in data:
            rankings = data.get("rankings", [])
            if isinstance(rankings, list):
                for idx, row in enumerate(rankings):
                    if not isinstance(row, dict):
                        continue
                    doc_id = row.get("segment_id") or row.get("doc_id") or row.get("id")
                    if doc_id is None:
                        continue
                    score_val = row.get("score")
                    if score_val is None:
                        score = float(len(rankings) - idx)
                    else:
                        try:
                            score = float(score_val)
                        except (TypeError, ValueError):
                            score = float(len(rankings) - idx)
                    scores[str(doc_id)] = score
    
    return scores


def load_ground_truth_scores(gt_file: Path, query_id: str) -> Dict[str, float]:
    """Load ground truth scores from ground truth JSON file for a specific query_id.
    
    The query_id from filename might be like "c33_b_3_json" but ground truth uses "c33_b_3.json"
    """
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    data = json.loads(gt_file.read_text())
    
    # Ground truth structure: {query_id: {doc_id: score, ...}, ...}
    if isinstance(data, dict):
        # Try different query_id formats
        candidates = [
            query_id,  # Exact match
            f"{query_id}.json",  # Add .json extension
            query_id.replace("_json", ".json"),  # Replace _json with .json
            query_id.replace("_json", ""),  # Remove _json
        ]
        
        for candidate in candidates:
            gt_scores = data.get(candidate)
            if gt_scores and isinstance(gt_scores, dict):
                return {str(doc): float(score) for doc, score in gt_scores.items()}
    
    return {}


def compute_ranking_metrics_direct(
    pred_scores: Dict[str, float],
    gt_scores: Dict[str, float],
    query_id: str,
    k_values: list[int] = np.linspace(1, 20, 10).tolist(),
) -> Dict[str, Any]:
    """Compute ranking metrics directly from loaded scores.
    
    Items with ground truth score of -1 are excluded from evaluation.
    """
    from eval_pipeline.metrics.helpers import ndcg_at_k as _ndcg_at_k, map_at_k as _map_at_k
    from eval_pipeline.metrics.utils import safe_ratio
    
    if not pred_scores:
        raise ValueError("No predicted rankings found")
    if not gt_scores:
        raise ValueError("Ground truth rankings are empty or missing")
    
    # Filter out items with score -1 in ground truth (these should not be involved in ranking)
    excluded_items = {doc_id for doc_id, score in gt_scores.items() if float(score) == -1.0}
    
    # Filter predictions to exclude items with -1 in ground truth
    filtered_pred_scores = {
        doc_id: score 
        for doc_id, score in pred_scores.items() 
        if doc_id not in excluded_items
    }
    
    # Filter ground truth to exclude items with -1
    filtered_gt_scores = {
        doc_id: score 
        for doc_id, score in gt_scores.items() 
        if float(score) != -1.0
    }
    
    if not filtered_pred_scores:
        raise ValueError(f"After filtering out excluded items, no predicted rankings remain for query {query_id}")
    if not filtered_gt_scores:
        raise ValueError(f"After filtering out excluded items, no ground truth rankings remain for query {query_id}")
    
    positive_docs = [doc for doc, score in filtered_gt_scores.items() if score > 0]
    if not positive_docs:
        raise ValueError(f"No positive documents in ground truth for query {query_id} after filtering")
    
    binary_rels: list[int] = []
    gain_rels: list[float] = []
    ranked_docs = sorted(filtered_pred_scores.items(), key=lambda kv: kv[1], reverse=True)
    
    for doc_id, _score in ranked_docs:
        doc_id = str(doc_id)
        gain = max(float(filtered_gt_scores.get(doc_id, 0.0)), 0.0)
        gain_rels.append(gain)
        if gain >= 4:
            binary_rels.append(1)
        else:
            binary_rels.append(0)
    
    if not binary_rels:
        raise ValueError(f"No overlapping documents between predictions and ground truth for query {query_id}")
    
    results: Dict[str, Any] = {"query_id": query_id}
    
    for k_val in k_values:
        top_binary = binary_rels[:k_val]
        top_gain = gain_rels[:k_val]
        if not top_binary:
            continue
        hits_k = sum(top_binary)
        precision_k = safe_ratio(hits_k, len(top_binary))
        recall_k = safe_ratio(hits_k, len(positive_docs))
        map_k = _map_at_k(binary_rels, k_val)
        ndcg_k = _ndcg_at_k(gain_rels, k_val)
        results[f"precision@{k_val}"] = round(precision_k, 4)
        results[f"recall@{k_val}"] = round(recall_k, 4)
        results[f"map@{k_val}"] = round(map_k, 4)
        results[f"ndcg@{k_val}"] = round(ndcg_k, 4)
    
    # Compute correlation metrics
    import numpy as np
    
    ids = []
    preds = []
    trues = []
    for doc_id, score in ranked_docs:
        doc_id = str(doc_id)
        if doc_id not in filtered_gt_scores:
            continue
        ids.append(doc_id)
        preds.append(float(score))
        trues.append(float(filtered_gt_scores[doc_id]))
    
    if len(ids) >= 2:
        # Spearman correlation
        pred_ranks = np.argsort(np.argsort(preds))
        true_ranks = np.argsort(np.argsort(trues))
        pred_ranks = pred_ranks.astype(np.float64) + 1
        true_ranks = true_ranks.astype(np.float64) + 1
        d = pred_ranks - true_ranks
        n = len(pred_ranks)
        numerator = 6 * np.sum(d * d)
        denom = n * (n * n - 1)
        if denom != 0:
            results["spearman"] = float(1 - numerator / denom)
        
        # Kendall tau-b
        concordant = 0
        discordant = 0
        ties_pred = 0
        ties_true = 0
        for i in range(n):
            for j in range(i + 1, n):
                pred_diff = preds[i] - preds[j]
                true_diff = trues[i] - trues[j]
                if pred_diff == 0 and true_diff == 0:
                    continue
                if pred_diff == 0:
                    ties_pred += 1
                    continue
                if true_diff == 0:
                    ties_true += 1
                    continue
                concordant += int(pred_diff * true_diff > 0)
                discordant += int(pred_diff * true_diff < 0)
        denom = np.sqrt((concordant + discordant + ties_pred) * (concordant + discordant + ties_true))
        if denom != 0:
            results["kendall_tau_b"] = float((concordant - discordant) / denom)
    
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate results from a given result path")
    p.add_argument("result_path", type=str, help="Path to result directory (e.g., results/fashion/tag/object_ranker/gpt-5.1/UMBRELLA_LLM)")
    p.add_argument("--data_root", default=".", help="Root directory for data files")
    p.add_argument("--artifacts_dir", default="artifacts", help="Directory for evaluation artifacts")
    p.add_argument("--reports_dir", default="reports", help="Directory for evaluation reports")
    p.add_argument("--manifest", default="artifacts/manifest.json", help="Path to manifest file")
    p.add_argument("--eval_steps", default=None, help="Override evaluation steps (comma-separated, e.g., 5,6)")
    
    args = p.parse_args()
    
    result_path = Path(args.result_path).resolve()
    if not result_path.exists():
        print(f"Error: Result path does not exist: {result_path}")
        return
    
    # Parse the path to get dataset, data, module, model, method, and result directory
    try:
        dataset_name, data_name, module_name, model_name, method_name, result_dir = parse_result_path(result_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Dataset: {dataset_name}")
    print(f"Data: {data_name}")
    print(f"Module: {module_name}")
    print(f"Model: {model_name}")
    print(f"Method: {method_name}")
    
    # Determine which steps to evaluate
    if args.eval_steps:
        steps = [int(s.strip()) for s in args.eval_steps.split(",") if s.strip().isdigit()]
    else:
        steps = MODULE_TO_STEPS.get(module_name, [])
    
    if not steps:
        print(f"Warning: No evaluation steps found for module '{module_name}'. Available modules: {list(MODULE_TO_STEPS.keys())}")
        return
    
    print(f"Evaluation steps: {steps}")
    
    # Find all ranking files in the result directory
    # Files are named like: {query_id}_rankings.json for object_ranker
    # or {query_id}_snippets_json.json for snippet_ranker
    # Extract query_id from filename
    ranking_files = []
    if result_dir.is_dir():
        for item in result_dir.iterdir():
            if module_name == "snippet_ranker" and item.is_file() and item.name.endswith("_snippets_json.json"):
                # For snippet_ranker, files are handled separately in the snippet_ranker section
                # We still add them here to maintain the loop structure, but they'll be processed differently
                query_id_from_file = item.name[:-len("_snippets_json.json")]
                # Try to get query_id from the file itself
                try:
                    file_data = json.loads(item.read_text())
                    if isinstance(file_data, dict) and "query_id" in file_data:
                        query_id_from_file = str(file_data["query_id"])
                except Exception:
                    pass
                ranking_files.append((item, query_id_from_file))
            elif item.is_file() and item.name.endswith("_rankings.json"):
                # Extract query_id from filename: remove _rankings.json
                query_id_from_file = item.name[:-len("_rankings.json")]
                # Also try to get query_id from the file itself (it might be "c33_b_3.json" instead of "c33_b_3_json")
                try:
                    file_data = json.loads(item.read_text())
                    if isinstance(file_data, dict) and "query_id" in file_data:
                        query_id_from_file = str(file_data["query_id"])
                except Exception:
                    pass
                ranking_files.append((item, query_id_from_file))
    
    if not ranking_files:
        print(f"Error: No ranking files found in {result_dir}")
        return
    
    print(f"Found {len(ranking_files)} ranking file(s) to evaluate")
    
    # Get method and model keys for this module
    method_key, model_key = MODULE_METADATA_KEYS.get(module_name, ("", ""))
    
    # Set up data root
    data_root = Path(args.data_root)
    dataset_root = infer_dataset_root(result_dir, dataset_name, data_root)
    query_not_evaluated = []
    # Process each ranking file
    for idx, (ranking_file, query_id) in enumerate(ranking_files, 1):
        print(f"\n[{idx}/{len(ranking_files)}] Evaluating query_id: {query_id}, file: {ranking_file.name}")
        
        # Build metrics_metadata for this query
        metrics_metadata: Dict[str, Any] = {}
        if method_key:
            metrics_metadata[method_key] = method_name
        if model_key:
            metrics_metadata[model_key] = model_name
        if data_name:
            metrics_metadata["data"] = data_name
        
        # Build evaluation config based on module
        metrics_cfg: Dict[str, Any] = {k: v for k, v in metrics_metadata.items() if v}
        metrics_cfg["query_id"] = query_id
        
        # Module-specific configuration
        if module_name == "object_ranker":
            # Step 5: Object ranking
            # Load JSON files directly and compute metrics
            try:
                # Load prediction scores
                pred_scores = load_prediction_scores(ranking_file, query_id)
                
                # Build ground truth variants
                gt_variants: List[Tuple[Path, Optional[str]]] = []
                if dataset_name and "fashion" in dataset_name.lower():
                    base = data_root / "data" / dataset_name / "ground_truth_rating"
                    gt_variants.append((base / "ground_truth_object_ranking_assistant.json", f"{data_name}_assistant" if data_name else "assistant"))
                    gt_variants.append((base / "ground_truth_object_ranking_seeker.json", f"{data_name}_seeker" if data_name else "seeker"))
                else:
                    gt_variants.append((data_root / "data" / dataset_name / "ground_truth_object_ranking.json", data_name))

                from eval_pipeline.stores.reporters import write_step_csv
                step_csv = Path(args.reports_dir) / "step5_report.csv"

                for gt_file, data_tag in gt_variants:
                    gt_scores = load_ground_truth_scores(gt_file, query_id)

                    metric_results = compute_ranking_metrics_direct(
                        pred_scores=pred_scores,
                        gt_scores=gt_scores,
                        query_id=query_id,
                        k_values=[1,2,3,4,5],
                    )

                    metric_row = {
                        "name": "metric",
                        "values": metric_results
                    }
            
                    write_step_csv(
                        step_csv,
                        [metric_row],
                        method_name=method_name,
                        model_name=model_name,
                        data_name=data_tag,
                    )

                print(f"  ✓ Completed evaluation for {query_id} for data {data_tag} and method {method_name} and model {model_name}")
                continue  # Skip the normal evaluation pipeline
                
            except Exception as e:
                print(f"  ✗ Error evaluating {query_id}: {e}")
                # Write error to report
                from eval_pipeline.stores.reporters import write_step_csv
                step_csv = Path(args.reports_dir) / "step5_report.csv"
                write_step_csv(
                    step_csv,
                    [{"name": "metric", "values": {"query_id": query_id, "error": str(e)}}],
                    method_name=method_name,
                    model_name=model_name,
                    data_name=data_name,
                )
                continue
            
        elif module_name == "snippet_ranker":
            # Step 6: Snippet ranking
            # Process the current snippet file
            try:
                # Use the current ranking_file (which is actually a snippet file for snippet_ranker)
                snippet_file = ranking_file
                
                # Check if method contains "REASON" to determine which ground truth to use
                has_reason = "REASON" in method_name.upper()
                
                # Determine which ground truth files to use
                if has_reason:
                    # If REASON in method name, use reason ground truth only
                    gt_configs = [("ground_truth_snippet_by_reason.json", "reason")]
                else:
                    # If no REASON, use both human and questions ground truth
                    gt_configs = [
                        ("ground_truth_snippet_by_human.json", "human"),
                        ("ground_truth_snippet_by_questions.json", "questions")
                    ]
                
                # Load the result file once (used for all GT evaluations)
                result_data = json.loads(snippet_file.read_text())
                query_id_from_file = result_data.get("query_id")
                if not query_id_from_file:
                    # Use the query_id from the loop
                    query_id_from_file = query_id
                
                # Normalize query_id format
                if query_id_from_file.endswith("_json"):
                    query_id_from_file = query_id_from_file[:-5] + ".json"
                elif not query_id_from_file.endswith(".json"):
                    query_id_from_file = query_id_from_file + ".json"
                
                # Get ranked snippets from results (used for all GT evaluations)
                ranked_snippets_by_item = result_data.get("snippets", {})
                if not ranked_snippets_by_item:
                    print(f"  ⚠ No snippets found in results for query {query_id_from_file}, skipping")
                    continue
                
                # Evaluate with each ground truth file
                for gt_filename, gt_label in gt_configs:
                    # Load ground truth (with caching, keyed by dataset and gt type)
                    cache_key = f"{dataset_name}_snippets_{gt_label}"
                    if cache_key not in _gt_snippet_cache:
                        gt_path = data_root / "data" / dataset_name / "groud_truth_snippet" / gt_filename
                        if not gt_path.exists():
                            print(f"  ✗ Ground truth file not found: {gt_path}")
                            continue
                        _gt_snippet_cache[cache_key] = json.loads(gt_path.read_text())
                    gt_data = _gt_snippet_cache[cache_key]
                    
                    # Get ground truth snippets for this query
                    gt_query_snippets = gt_data.get(query_id_from_file, {})
                    if not gt_query_snippets:
                        # Try alternative query_id formats
                        candidates = [
                            query_id_from_file.replace(".json", ""),
                            query_id_from_file.replace("_json", ".json"),
                            query_id.replace("_json", ".json") if query_id.endswith("_json") else query_id + ".json",
                        ]
                        for candidate in candidates:
                            if candidate in gt_data:
                                gt_query_snippets = gt_data[candidate]
                                break
                    
                    if not gt_query_snippets:
                        print(f"  ⚠ No ground truth found for query {query_id_from_file} in {gt_filename}, skipping")
                        continue
                    
                    # Calculate HIT@K and Precision@K for each item, then average
                    hit_at_k_by_item: Dict[int, List[float]] = {k: [] for k in [1, 2, 3, 4, 5]}
                    precision_at_k_by_item: Dict[int, List[float]] = {k: [] for k in [1, 2, 3, 4, 5]}
                    items_evaluated = 0
                    for item_id, ranked_snippets in ranked_snippets_by_item.items():
                        # Get ground truth snippets for this item
                        gt_item_snippets = gt_query_snippets.get(item_id, [])
                        if not gt_item_snippets:
                            if '.json' in item_id:
                                item_id = item_id.replace('.json', '')
                                gt_item_snippets = gt_query_snippets.get(item_id, [])
                            if not gt_item_snippets:
                                continue
                        
                        # Get relevant snippet IDs (relevance == 1)
                        relevant_snippet_ids = set()
                        relevant_snippet_ids_numeric = set()  # For numeric matching (questions/human GT)
                        for gt_snippet in gt_item_snippets:
                            if isinstance(gt_snippet, dict):
                                snippet_id = gt_snippet.get("id")
                                relevance = gt_snippet.get("relevance", 0)
                                if relevance == 1 and snippet_id is not None:
                                    # For questions/human GT: snippet_id is an integer
                                    # For reason GT: snippet_id is a string like "item_2.json#catalogue#0_12"
                                    snippet_id_str = str(snippet_id)
                                    relevant_snippet_ids.add(snippet_id_str)
                                    relevant_snippet_ids_numeric.add(str(int(snippet_id)))
                        
                        if not relevant_snippet_ids and not relevant_snippet_ids_numeric:
                            # No relevant snippets for this item, skip
                            continue
                        
                        # Get ranked snippet IDs from results (ordered by relevance score, descending)
                        # Collect snippet IDs and their relevance scores
                        snippet_score_pairs = []
                        for snippet in ranked_snippets:
                            if isinstance(snippet, dict):
                                snippet_id = snippet.get("id")
                                score = snippet.get("relevance", 0)
                                # Ensure score is numeric for proper sorting
                                try:
                                    score = float(score) if score is not None else 0.0
                                except (ValueError, TypeError):
                                    score = 0.0
                                if snippet_id is not None:
                                    snippet_score_pairs.append((str(snippet_id), score))
                        # Sort from highest to lowest relevance score
                        ranked_snippet_ids = sorted(snippet_score_pairs, key=lambda x: x[1], reverse=True)
                        ranked_snippet_ids = [sid for sid, _ in ranked_snippet_ids]
                        if not ranked_snippet_ids:
                            continue
                        # Calculate HIT@K and Precision@K for this item
                        for k in [1, 2, 3, 4, 5]:
                            top_k_ids = ranked_snippet_ids[:k]
                            # HIT@K = 1 if at least one relevant snippet in top K, else 0
                            # Try exact match first, then numeric match
                            hit = 0.0
                            if any(sid in relevant_snippet_ids for sid in top_k_ids):
                                hit = 1.0
                            elif relevant_snippet_ids_numeric and any(sid in relevant_snippet_ids_numeric for sid in top_k_ids):
                                hit = 1.0
                            hit_at_k_by_item[k].append(hit)
                            
                            # Precision@K = (number of relevant snippets in top K) / K
                            relevant_count = 0
                            for sid in top_k_ids:
                                if sid in relevant_snippet_ids:
                                    relevant_count += 1
                                elif relevant_snippet_ids_numeric and sid in relevant_snippet_ids_numeric:
                                    relevant_count += 1
                            precision = relevant_count / k if k > 0 else 0.0
                            precision_at_k_by_item[k].append(precision)
                        
                        items_evaluated += 1
                    
                    if items_evaluated == 0:
                        print(f"  !!! No items with relevant snippets found for query {query_id_from_file} in {gt_filename}, skipping")
                        if gt_label == gt_configs[-1][1]:  # Only add to query_not_evaluated for the last GT file
                            query_not_evaluated.append(query_id_from_file)
                        continue
                    
                    # Calculate mean HIT@K and Precision@K over all items
                    metric_results = {
                        "query_id": query_id_from_file,
                        "gt": gt_label  # Changed from "reason" to "gt" with values "human", "questions", or "reason"
                    }
                    for k in [1, 2, 3, 4, 5]:
                        if hit_at_k_by_item[k]:
                            mean_hit = sum(hit_at_k_by_item[k]) / len(hit_at_k_by_item[k])
                            metric_results[f"hit@{k}"] = round(mean_hit, 4)
                        else:
                            metric_results[f"hit@{k}"] = 0.0
                        
                        if precision_at_k_by_item[k]:
                            mean_precision = sum(precision_at_k_by_item[k]) / len(precision_at_k_by_item[k])
                            metric_results[f"precision@{k}"] = round(mean_precision, 4)
                        else:
                            metric_results[f"precision@{k}"] = 0.0
                    
                    from eval_pipeline.stores.reporters import write_step_csv
                    step_csv = Path(args.reports_dir) / "step6_report.csv"
                    
                    metric_row = {
                        "name": "metric",
                        "values": metric_results
                    }
                    
                    write_step_csv(
                        step_csv,
                        [metric_row],
                        method_name=method_name,
                        model_name=model_name,
                        data_name=data_name,
                    )
                    
                    print(f"  ✓ Completed evaluation for query {query_id_from_file} ({items_evaluated} items) with {gt_label} ground truth")
                
                continue  # Skip the normal evaluation pipeline
                
            except Exception as e:
                print(f"  ✗ Error evaluating snippet file {query_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        elif module_name == "linker":
            # Step 4: Linking
            # Linker outputs are typically in segment_links.json format
            # We may need to copy it to data/preds/segment_links.json
            links_files = []
            if result_dir.is_dir():
                for item in result_dir.iterdir():
                    if item.is_file() and (item.name.endswith("_links.json") or item.name == "segment_links.json"):
                        links_files.append(item)
            
            if links_files:
                # Use first links file found
                links_file = links_files[0]
                links_out = data_root / "data/preds/segment_links.json"
                links_out.parent.mkdir(parents=True, exist_ok=True)
                links_out.write_text(links_file.read_text())
        
        elif module_name == "segment_describer":
            # Step 2: Description
            # Descriptions are typically in {method}_entries.json
            desc_files = []
            if result_dir.is_dir():
                for item in result_dir.iterdir():
                    if item.is_file() and item.name.endswith("_entries.json"):
                        desc_files.append(item)
            
            if desc_files:
                # Use first description file found
                desc_file = desc_files[0]
                descriptions_out = data_root / "data/preds/descriptions_pred.json"
                descriptions_out.parent.mkdir(parents=True, exist_ok=True)
                descriptions_out.write_text(desc_file.read_text())
        
        # Map step keys
        step_key_map = {
            "segmenter_method": "step1_method",
            "segmenter_model": "step1_model",
            "describer_method": "step2_method",
            "describer_model": "step2_model",
            "linker_method": "step4_method",
            "linker_model": "step4_model",
            "object_ranker_method": "step5_method",
            "object_ranker_model": "step5_model",
            "snippet_ranker_method": "step6_method",
            "snippet_ranker_model": "step6_model",
        }
        for src, dst in step_key_map.items():
            value = metrics_metadata.get(src)
            if value:
                metrics_cfg[dst] = value
        
        # Create evaluation config
        cfg = EvalConfig(
            data_root=data_root,
            artifacts_dir=Path(args.artifacts_dir),
            reports_dir=Path(args.reports_dir),
            manifest_path=Path(args.manifest),
            plugins_dir=None,
            extra_metrics=[],
            metrics_cfg=metrics_cfg,
        )
        
        # Run evaluation for this query
        try:
            eval_res = run_steps(cfg, steps)
            print(f"  ✓ Completed evaluation for {query_id}")
        except Exception as e:
            print(f"  ✗ Error evaluating {query_id}: {e}")
            continue
    print('query_not_evaluated', query_not_evaluated)
    print(f"\nEvaluation complete. Processed {len(ranking_files)} file(s). Reports written to: {args.reports_dir}")


if __name__ == "__main__":
    main()

