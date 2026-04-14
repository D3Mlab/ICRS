#!/usr/bin/env python3

"""
Evaluate snippet_ranker results to find false positives in top 3 and label them.

This script:
1. Reads snippet_ranker result files
2. Loads ground truth snippet data
3. Identifies false positives in top 3 ranks
4. Uses Gemini 2.5 Flash to label each false positive into one of three categories:
   - IP (Incorrect Proactive Information)
   - SR (Surface-level Request)
   - VI (Latent Awareness)
5. Saves tagged results in tags/ folder with same structure as input
"""

from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys

# Add parent directory to path to import llm_base
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_base import LLMBase

# Cache for ground truth snippet data (keyed by dataset)
_gt_snippet_cache: Dict[str, Dict[str, Any]] = {}

# Cache for query data (keyed by dataset)
_query_cache: Dict[str, Dict[str, str]] = {}

# Cache for seeker questions (keyed by dataset)
_seeker_questions_cache: Dict[str, Dict[str, List[str]]] = {}


def parse_result_path(result_path: Path) -> tuple[str, str, str, str, str, Path, Path]:
    """
    Parse result path to extract dataset, data, module, model, method, result directory, and data root.
    
    Expected format: .../results/{dataset}/{data}/{module}/{model}/{method}
    or absolute path: /path/to/results/{dataset}/{data}/{module}/{model}/{method}
    
    Returns:
        (dataset_name, data_name, module_name, model_name, method_name, result_dir, data_root)
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
            
            # Infer data_root: everything up to (but not including) "results"
            # For absolute paths, reconstruct from parts
            if result_path.is_absolute():
                # Reconstruct absolute path up to "results" directory
                # parts[:results_idx] gives all parts before "results"
                data_root = Path("/").joinpath(*parts[:results_idx])
            else:
                # For relative paths, use current working directory
                data_root = Path.cwd()
                for i in range(results_idx):
                    data_root = data_root / parts[i]
            
            return dataset, data, module, model, method, result_dir, data_root
    except StopIteration:
        pass
    
    raise ValueError(f"Cannot parse result path: {result_path}. Expected format: .../results/{{dataset}}/{{data}}/{{module}}/{{model}}/{{method}}")


def load_ground_truth_snippets(data_root: Path, dataset_name: str, method_name: str) -> Dict[str, Any]:
    """Load ground truth snippet data based on method name."""
    # Check if method contains "REASON" to determine which ground truth to use
    has_reason = "REASON" in method_name.upper()
    
    if has_reason:
        gt_filename = "ground_truth_snippet_by_reason.json"
        gt_label = "reason"
    else:
        # For non-REASON methods, use human ground truth
        gt_filename = "ground_truth_snippet_by_human.json"
        gt_label = "questions"
    
    cache_key = f"{dataset_name}_snippets_{gt_label}"
    if cache_key not in _gt_snippet_cache:
        gt_path = data_root / "data" / dataset_name / "groud_truth_snippet" / gt_filename
        if not gt_path.exists():
            # Try alternative path
            gt_path = data_root / "data" / dataset_name / "ground_truth_snippet" / gt_filename
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        _gt_snippet_cache[cache_key] = json.loads(gt_path.read_text())
    
    return _gt_snippet_cache[cache_key]


def load_query_data(data_root: Path, dataset_name: str, data_name: str) -> Dict[str, str]:
    """Load query data for the dataset."""
    cache_key = f"{dataset_name}_{data_name}"
    if cache_key not in _query_cache:
        # Try different possible query file paths
        query_paths = [
            data_root / "data" / dataset_name / "query" / "by_pre_recommend.json"
        ]
        
        query_path = None
        for path in query_paths:
            if path.exists():
                query_path = path
                break
        
        if query_path is None:
            raise FileNotFoundError(f"Query file not found. Tried: {[str(p) for p in query_paths]}")
        
        _query_cache[cache_key] = json.loads(query_path.read_text())
    
    return _query_cache[cache_key]


def load_seeker_questions(data_root: Path, dataset_name: str) -> Dict[str, List[str]]:
    """Load seeker questions (ground truth needs) for the dataset."""
    if dataset_name not in _seeker_questions_cache:
        # Try different possible seeker questions file paths
        seeker_questions_paths = [
            data_root / "data" / dataset_name / "utterance" / "seeker_questions_generated.json",
        ]
        
        seeker_questions_path = None
        for path in seeker_questions_paths:
            if path.exists():
                seeker_questions_path = path
                break
        
        if seeker_questions_path is None:
            # Return empty dict if file doesn't exist (not all datasets may have this)
            print(f"  ⚠ Seeker questions file not found for {dataset_name}, using empty ground truth needs")
            _seeker_questions_cache[dataset_name] = {}
        else:
            _seeker_questions_cache[dataset_name] = json.loads(seeker_questions_path.read_text())
    
    return _seeker_questions_cache[dataset_name]


def find_false_positives_in_top3(
    ranked_snippets: List[Dict[str, Any]],
    gt_snippet_ids: set,
    gt_snippet_ids_numeric: set
) -> List[Dict[str, Any]]:
    """Find false positives in top 3 ranked snippets."""
    false_positives = []
    
    # Get top 3 snippets
    top_3 = ranked_snippets[:3]
    
    for snippet in top_3:
        snippet_id = str(snippet.get("id", ""))
        # Check if this snippet is NOT in ground truth
        is_relevant = (
            snippet_id in gt_snippet_ids or
            snippet_id in gt_snippet_ids_numeric
        )
        
        if not is_relevant:
            false_positives.append(snippet)
    
    return false_positives


def _read_image_b64(path: Path) -> str:
    """Read image file and return as base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _get_image_path(data_root: Path, dataset_name: str, item_id: str) -> Optional[Path]:
    """Get image path for an item_id from segments/crops directory."""
    # Try different possible image file extensions and formats
    crops_dir = data_root / "data" / dataset_name / "segments" / "crops"
    
    # Clean item_id (remove .json extension if present)
    clean_item_id = item_id.replace(".json", "")
    
    # Try different formats: {item_id}.jpg, {item_id}.png, segment_{item_id}.jpg, etc.
    possible_paths = [
        crops_dir / f"{clean_item_id}.jpg",
        crops_dir / f"{clean_item_id}.png",
        crops_dir / f"{clean_item_id}.jpeg",
        crops_dir / f"segment_{clean_item_id}.jpg",
        crops_dir / f"segment_{clean_item_id}.png",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    print(f"  ⚠ No image found for {clean_item_id}")
    return None


def label_false_positive(
    llm: LLMBase,
    query_text: str,
    snippet: Dict[str, Any],
    item_id: str,
    ground_truth_needs: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
    dataset_name: Optional[str] = None
) -> str:
    """Use LLM to label a false positive snippet."""
    snippet_text = snippet.get("text", "")
    snippet_id = snippet.get("id", "")
    
    # Format ground truth needs with question numbers
    if ground_truth_needs:
        numbered_needs = [f"Question {i+1}: {need}" for i, need in enumerate(ground_truth_needs)]
        gt_needs_str = "; ".join(numbered_needs)
    else:
        gt_needs_str = "Not available"
    
    # Try to load image for this item
    image_b64 = None
    image_path = None
    image_mime_type = "image/jpeg"
    if data_root and dataset_name:
        image_path = _get_image_path(data_root, dataset_name, item_id)
        if image_path:
            try:
                image_b64 = _read_image_b64(image_path)
                # Detect MIME type from file extension
                if image_path.suffix.lower() == ".png":
                    image_mime_type = "image/png"
                elif image_path.suffix.lower() in [".jpg", ".jpeg"]:
                    image_mime_type = "image/jpeg"
            except Exception as e:
                print(f"  ⚠ Could not load image from {image_path}: {e}")
                image_b64 = None
    
    prompt = f"""You are analyzing a false-positive snippet selection in a recommendation system.

Context:
- Conversation: {query_text}
- Snippet Text: {snippet_text}
- Ground Truth Questions (asked later by the seeker for IP only): {gt_needs_str}
{f"- Item Image: [Image of the item/movie poster is provided below]" if image_b64 else ""}

Background:
The conversation consists of a recommendation scenario between a seeker and a recommender. The seeker expresses (i) preferences/constraints used for recommendation and (ii) some explicit questions about the item/movie.

The snippet is a text passage associate with an item (like metadata) selected by the system as *proactive information*—i.e., information the seeker might want next but has not explicitly asked for in the current conversation. However, this snippet is a **false positive**: it should not have been selected.

A valid proactive snippet should:
- NOT be inferable from the visual appearance of the associate item itself / movie poster (e.g., color, shape, size, visible material, name/title, style), and
- NOT directly answer any explicit question already asked in the conversation, and
- NOT merely restate the rationale for recommending the item based on the seeker’s stated preferences.

Task:
Classify the reason this snippet is a false positive into **exactly one** of the following categories:

1. **VI (Visual Inferable)**: The snippet content maybe can be inferred from the associate item’s visual appearance (via attached item image) (e.g., visible attributes, style, material, color, title/name, style, or anything you think reasonable) assuming you are physically next to the item.
2. **RR (Reason Recommendation)**: The snippet mainly provides a direct reason why the item/movie was recommended, based on the seeker’s explicitly stated requests/preferences/constraints in the conversation. (do not use Ground Truth Questions as the basis for this category)
3. **ER (Explicit Request)**: The snippet answers an explicit question the seeker asked in the conversation (do not use Ground Truth Questions as the basis for this category).
4. **IP (Incorrect Proactive Information)**: The snippet does **not** answer or address **any** of the Ground Truth Questions that was asked later by the seeker listed above.
   - **IP-O (Off-Target)**: Item-related and topically relevant, but addresses the wrong aspect (does not match any ground-truth need).
   - **IP-G (Generic / Low-Utility)**: Overly generic/boilerplate, lacks specificity, and does not resolve any ground-truth need.
5. **Other**: Does not fit any category above.

Priority rule (choose the first applicable):
1) VI  
2) RR or ER  
4) IP-O or IP-G (pick the better fit)  
5) Other

Output:
Respond with ONLY ONE code: ER, VI, RR, IP-O, IP-G, or Other.
"""
    try:
        # Build message content - include image if available
        if image_b64 and llm.provider == "google":
            # Google API format for images
            content = [
                {"type": "text", "text": prompt},
                {"type": "inline_data", "inline_data": {"mime_type": image_mime_type, "data": image_b64}},
            ]
        elif image_b64:
            # OpenAI/OpenRouter format for images
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{image_mime_type};base64,{image_b64}", "detail": "high"}},
            ]
        else:
            print(f"  ⚠ No image found for {item_id}")
            content = prompt
        
        # Use generate_chat_completion directly to avoid any caching
        resp = llm.generate_chat_completion(
            messages=[{"role": "user", "content": content}],
            temperature=0.0
        )
        response = resp.choices[0].message.content
        # Extract the label (should be IP, SR, VI, or Other)
        response = response.strip().upper()
        if response in ["IP-O", "IP-G", "ER", "VI", "RR", "OTHER"]:
            return "Other" if response == "OTHER" else response
        # Try to extract from response if it contains the label
        for label in ["IP-O", "IP-G", "ER", "VI", "RR", "OTHER"]:
            if label in response:
                return "Other" if label == "OTHER" else label
        # Default to Other if can't parse
        print(f"  ⚠ Could not parse label from response: {response}, defaulting to Other")
        return "Other"
    except Exception as e:
        print(f"  ✗ Error labeling snippet {snippet_id}: {e}")
        return "Other"  # Default label


def process_snippet_file(
    snippet_file: Path,
    data_root: Path,
    dataset_name: str,
    data_name: str,
    method_name: str,
    model_name: str,
    llm: LLMBase,
    output_base: Path,
    reports_dir: Path
) -> None:
    """Process a single snippet result file."""
    print(f"Processing: {snippet_file.name}")
    
    # Load result file
    result_data = json.loads(snippet_file.read_text())
    query_id_from_file = result_data.get("query_id")
    
    # Normalize query_id format
    if query_id_from_file.endswith("_json"):
        query_id_from_file = query_id_from_file[:-5] + ".json"
    elif not query_id_from_file.endswith(".json"):
        query_id_from_file = query_id_from_file + ".json"
    
    # Load ground truth
    try:
        gt_data = load_ground_truth_snippets(data_root, dataset_name, method_name)
    except FileNotFoundError as e:
        print(f"  ✗ {e}, skipping")
        return
    
    # Load query data
    try:
        query_data = load_query_data(data_root, dataset_name, data_name)
    except FileNotFoundError as e:
        print(f"  ✗ {e}, skipping")
        return
    
    # Load seeker questions (ground truth needs)
    seeker_questions = load_seeker_questions(data_root, dataset_name)
    
    # Get query text
    query_text = query_data.get(query_id_from_file, "")
    if not query_text:
        # Try alternative formats
        candidates = [
            query_id_from_file.replace(".json", ""),
            query_id_from_file.replace("_json", ".json"),
        ]
        for candidate in candidates:
            if candidate in query_data:
                query_text = query_data[candidate]
                break
    
    if not query_text:
        print(f"  ⚠ No query text found for {query_id_from_file}, skipping")
        return
    
    # Get ground truth needs (seeker questions) for this query
    ground_truth_needs = seeker_questions.get(query_id_from_file, [])
    if not ground_truth_needs:
        # Try alternative query_id formats
        candidates = [
            query_id_from_file.replace(".json", ""),
            query_id_from_file.replace("_json", ".json"),
        ]
        for candidate in candidates:
            if candidate in seeker_questions:
                ground_truth_needs = seeker_questions[candidate]
                break
    
    # Get ranked snippets from results
    ranked_snippets_by_item = result_data.get("snippets", {})
    if not ranked_snippets_by_item:
        print(f"  ⚠ No snippets found in results for query {query_id_from_file}, skipping")
        return
    
    # Get ground truth snippets for this query
    gt_query_snippets = gt_data.get(query_id_from_file, {})
    if not gt_query_snippets:
        # Try alternative query_id formats
        candidates = [
            query_id_from_file.replace(".json", ""),
            query_id_from_file.replace("_json", ".json"),
        ]
        for candidate in candidates:
            if candidate in gt_data:
                gt_query_snippets = gt_data[candidate]
                break
    
    if not gt_query_snippets:
        print(f"  ⚠ No ground truth found for query {query_id_from_file}, skipping")
        return
    
    # Determine GT label based on method
    has_reason = "REASON" in method_name.upper()
    gt_label = "reason" if has_reason else "questions"
    
    # Process each item
    tagged_result = {
        "query": result_data.get("query", ""),
        "query_id": query_id_from_file,
        "method": result_data.get("method", ""),
        "model": result_data.get("model", ""),
        "snippets": {}
    }
    
    has_false_positives = False
    # Track tag counts for this query
    tag_counts = {"IP-O": 0, "IP-G": 0, "ER": 0, "VI": 0, "RR": 0, "Other": 0}
    
    for item_id, ranked_snippets in ranked_snippets_by_item.items():
        # Get ground truth snippets for this item
        gt_item_snippets = gt_query_snippets.get(item_id, [])
        if not gt_item_snippets:
            # Try without .json extension
            item_id_alt = item_id.replace('.json', '')
            gt_item_snippets = gt_query_snippets.get(item_id_alt, [])
            if not gt_item_snippets:
                # Copy snippets as-is if no ground truth
                tagged_result["snippets"][item_id] = ranked_snippets
                continue
        
        # Get relevant snippet IDs (relevance == 1)
        relevant_snippet_ids = set()
        relevant_snippet_ids_numeric = set()
        for gt_snippet in gt_item_snippets:
            if isinstance(gt_snippet, dict):
                snippet_id = gt_snippet.get("id")
                relevance = gt_snippet.get("relevance", 0)
                if relevance == 1 and snippet_id is not None:
                    snippet_id_str = str(snippet_id)
                    relevant_snippet_ids.add(snippet_id_str)
                    try:
                        relevant_snippet_ids_numeric.add(str(int(snippet_id)))
                    except (ValueError, TypeError):
                        pass
        
        if not relevant_snippet_ids and not relevant_snippet_ids_numeric:
            # No relevant snippets for this item, copy as-is
            tagged_result["snippets"][item_id] = ranked_snippets
            continue
        
        # Sort ranked snippets by relevance score to find top 3
        snippet_score_pairs = []
        for snippet in ranked_snippets:
            if isinstance(snippet, dict):
                snippet_id = snippet.get("id")
                score = snippet.get("relevance", 0)
                try:
                    score = float(score) if score is not None else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                if snippet_id is not None:
                    snippet_score_pairs.append((snippet, score))
        
        # Sort from highest to lowest relevance score
        sorted_snippets = sorted(snippet_score_pairs, key=lambda x: x[1], reverse=True)
        ranked_snippets_sorted = [snippet for snippet, _ in sorted_snippets]
        
        # Find false positives in top 3
        false_positives = find_false_positives_in_top3(
            ranked_snippets_sorted,
            relevant_snippet_ids,
            relevant_snippet_ids_numeric
        )
        
        # Create a set of false positive IDs for quick lookup
        fp_ids = {str(fp.get("id", "")) for fp in false_positives}
        
        # Create a mapping from snippet ID to rank in top 3
        top3_rank_map = {}
        for idx, snippet in enumerate(ranked_snippets_sorted[:3]):
            snippet_id = str(snippet.get("id", ""))
            top3_rank_map[snippet_id] = idx + 1
        
        # Tag false positives (preserving original order)
        tagged_snippets = []
        for snippet in ranked_snippets:
            snippet_id = str(snippet.get("id", ""))
            tagged_snippet = snippet.copy()
            
            # Check if this is a false positive in top 3
            if snippet_id in fp_ids and snippet_id in top3_rank_map:
                # Label this false positive
                rank = top3_rank_map[snippet_id]
                label = label_false_positive(llm, query_text, snippet, item_id, ground_truth_needs, data_root, dataset_name)
                tagged_snippet["false_positive_tag"] = label
                has_false_positives = True
                # Normalize label for counting (handle case variations)
                label_upper = label.upper()
                if label_upper == "OTHER":
                    label_normalized = "Other"
                elif label_upper in ["IP-O", "IP-G", "ER", "VI", 'RR']:
                    label_normalized = label_upper
                else:
                    label_normalized = "Other"
                tag_counts[label_normalized] = tag_counts.get(label_normalized, 0) + 1
                print(f"    Tagged snippet {snippet_id} (rank {rank}) as {label}")
            
            tagged_snippets.append(tagged_snippet)
        
        tagged_result["snippets"][item_id] = tagged_snippets
    
    # Save tagged result if there are false positives
    if has_false_positives:
        # Create output path with same structure as input
        # Input: results/{dataset}/{data}/snippet_ranker/{model}/{method}/{file}
        # Output: tags/{dataset}/{data}/snippet_ranker/{model}/{method}/{file}
        # Try to compute relative path from data_root/results
        try:
            results_dir = data_root / "results"
            if snippet_file.is_relative_to(results_dir):
                relative_path = snippet_file.relative_to(results_dir)
            else:
                # If absolute path, extract the path after "results"
                parts = snippet_file.parts
                try:
                    results_idx = next(i for i, p in enumerate(parts) if p == "results")
                    relative_path = Path(*parts[results_idx + 1:])
                except StopIteration:
                    # Fallback: use filename only
                    relative_path = Path(snippet_file.name)
        except (ValueError, AttributeError):
            # Fallback: extract path components after "results"
            parts = snippet_file.parts
            try:
                results_idx = next(i for i, p in enumerate(parts) if p == "results")
                relative_path = Path(*parts[results_idx + 1:])
            except StopIteration:
                # Last resort: use filename only
                relative_path = Path(snippet_file.name)
        
        output_path = output_base / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write tagged result
        with open(output_path, 'w') as f:
            json.dump(tagged_result, f, indent=2)
        
        print(f"  ✓ Saved tagged results to {output_path}")
    else:
        print(f"  ℹ No false positives found in top 3 for this file")
    
    # Write CSV report with tag counts for this query (even if no false positives)
    from eval_pipeline.stores.reporters import write_step_csv
    
    csv_report_path = reports_dir / "fp_analysis.csv"
    metric_row = {
        "name": "fp_analysis",
        "values": {
            "query_id": query_id_from_file,
            "gt": gt_label,
            "count_IP-O": tag_counts["IP-O"],
            "count_IP-G": tag_counts["IP-G"],
            "count_ER": tag_counts["ER"],
            "count_VI": tag_counts["VI"],
            "count_RR": tag_counts["RR"],
            "count_Other": tag_counts["Other"],
            "total_fp": sum(tag_counts.values())
        }
    }
    
    write_step_csv(
        csv_report_path,
        [metric_row],
        method_name=method_name,
        model_name=model_name,
        data_name=data_name,
    )
    
    print(f"  ✓ Updated CSV report: {csv_report_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate snippet_ranker results and label false positives")
    p.add_argument("result_path", type=str, help="Path to result directory or file (e.g., results/movie/.../snippet_ranker/...)")
    p.add_argument("--data_root", default=".", help="Root directory for data files")
    p.add_argument("--output_base", default="reports/tags", help="Base directory for output tagged files")
    p.add_argument("--reports_dir", default="reports", help="Directory for CSV reports")
    p.add_argument("--gemini_api_key", default=None, help="Google API key (or set GOOGLE_API_KEY env var)")
    p.add_argument("--gemini_model", default="gemini-2.5-flash", help="Model to use for labeling")
    
    args = p.parse_args()
    
    result_path = Path(args.result_path).resolve()
    if not result_path.exists():
        print(f"Error: Result path does not exist: {result_path}")
        return
    
    # Parse the path to get dataset, data, module, model, method, and inferred data_root
    try:
        dataset_name, data_name, module_name, model_name, method_name, result_dir, inferred_data_root = parse_result_path(result_path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    if module_name != "snippet_ranker":
        print(f"Error: This script is only for snippet_ranker module, got: {module_name}")
        return
    
    print(f"Dataset: {dataset_name}")
    print(f"Data: {data_name}")
    print(f"Module: {module_name}")
    print(f"Model: {model_name}")
    print(f"Method: {method_name}")
    
    # Set up data root - use inferred from path if not explicitly provided
    if args.data_root == ".":
        data_root = inferred_data_root
        print(f"Inferred data root: {data_root}")
    else:
        data_root = Path(args.data_root)
    
    output_base = Path(args.output_base)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM (Gemini 2.5 Flash Lite)
    api_key = args.gemini_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Google API key required. Set GOOGLE_API_KEY env var or use --gemini_api_key")
        return
    
    llm = LLMBase(
        provider="google",
        api_key=api_key,
        model=args.gemini_model,
        temperature=0.0
    )
    
    # Find all snippet files in the result directory
    snippet_files = []
    if result_dir.is_dir():
        for item in result_dir.iterdir():
            if item.is_file() and item.name.endswith("_snippets_json.json"):
                snippet_files.append(item)
    elif result_path.is_file() and result_path.name.endswith("_snippets_json.json"):
        snippet_files.append(result_path)
    
    if not snippet_files:
        print(f"Error: No snippet files found in {result_dir}")
        return
    
    print(f"Found {len(snippet_files)} snippet file(s) to process")
    
    # Process each snippet file
    for idx, snippet_file in enumerate(snippet_files, 1):
        print(f"\n[{idx}/{len(snippet_files)}] Processing: {snippet_file.name}")
        try:
            process_snippet_file(
                snippet_file,
                data_root,
                dataset_name,
                data_name,
                method_name,
                model_name,
                llm,
                output_base,
                reports_dir
            )
        except Exception as e:
            print(f"  ✗ Error processing {snippet_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Processing complete. Tagged files saved to: {output_base}")


if __name__ == "__main__":
    main()

