ICRS: Immersive Conversational Recommendation System
====================================================

This repository implements an end-to-end immersive conversational recommendation System (ICRS) and retrieval pipeline, along with evaluation utilities and pre-computed experiment outputs for academic reproducibility.

IMPORTANT: due to space constraints of anyomous repo, you have to first download the data and unzip the data.zip from [Zenodo](https://zenodo.org/records/18158834?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc0ZDNjZmUwLWU0M2EtNDIzZS04MDFiLWQ5ZGY3ZWE0YzUzYSIsImRhdGEiOnt9LCJyYW5kb20iOiI2MTViNmY3OTZiZTBmOGI0N2MxNzUxOWM2M2IwY2UzOSJ9.dGBTJt5cDlY5gYKH1pI0VX3YAczPHOzA-V2Og4lLpMSx6Jsev8Cc_odjAqHsAlh5lGUp2uw9jdH25eCWNiisTQ) into this folder to use the repo

Repository Layout
-----------------
- `main.py`: Entry point that wires the pipeline together (linker + item recommendation + optional eval).
- `orchestrator/`: Pipeline driver (`ICRSPipeline`) that sequences major stages.
- `item_segmentation/`: SAM-based segmentation.
- `linker/`: Attribute Lookup Components of ICRS
- `item_recommendation/`: Item recommendation methods (BM25, dense, fusion, rerank, umbrella LLM/VLM variants).
- `label_selection/`: abel selection methods and atomic attributes decomposition.
- `eval/`: Evaluation pipeline, metrics, and CLI for offline analysis.
- `configs/`: YAML configs for pipeline components (`default.yaml`, `item_recommendation.yaml`, `label_selection.yaml`, `linker.yaml`).
- `scripts/`: Convenience runners for sweeping methods or iterating over conversations.
- `data/`: Three datasets (`fashion`, `movie`, `retail`) with attributes, conversations, labels, and segment crops.
- `pre_run_results/`: Precomputed outputs (JSON) for experiments, grouped by dataset and stage.
- `artifacts/` & `log/`: Generated artifacts and logs (may be created at runtime).

Data Layout (per dataset under `data/{fashion|movie|retail}`)
------------------------------------------------------------
- `attributes/`
  - `raw/`: Individual item JSON files.
  - `atomic_attributes.json`: decomposted attribute dictionary.
  - `ground_truth_mapping.json/segment_links.json`: One to one mapping between attributes and segments
- `conversation/`
  - `raw/`: Conversation transcripts.
  - `agg/`: Aggregated conversations 
    - `pre_gt.json` utterance before first ground truth item mentioned in the conversation
    - `full.json` full conversation
- `groud_truth_label/`: Ground-truth labels by intent/tag.
    - `utterance_by_intent_tag` taged utterances by intent tags (Sec 3.3 of the paper)
- `segments/crops/`: Cropped segment (We assume perfect segment in this paper).
- `ground_truth_object_ranking.json`: Ground truth Item ranking annotations.

Pre-run Results Layout (`pre_run_results/`)

IMPORTANT: due to space constraints of anyomous repo, you need to download pre_run_results.zip from [Zenodo](https://zenodo.org/records/18158834?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc0ZDNjZmUwLWU0M2EtNDIzZS04MDFiLWQ5ZGY3ZWE0YzUzYSIsImRhdGEiOnt9LCJyYW5kb20iOiI2MTViNmY3OTZiZTBmOGI0N2MxNzUxOWM2M2IwY2UzOSJ9.dGBTJt5cDlY5gYKH1pI0VX3YAczPHOzA-V2Og4lLpMSx6Jsev8Cc_odjAqHsAlh5lGUp2uw9jdH25eCWNiisTQ) and unzip it
-------------------------------------------
Precomputed outputs reported in the manuscript
```
pre_run_results/
  {dataset}/
    full/
      item_recommendation/    # Item rec with full conversation
    pre_gt/
      item_recommendation/    # Item rec before ground-truth metioned
      label_selection/        # Snippet selection outputs
```
- Filenames follow the query/conversation id (e.g., `test_1.json`) and may include method labels.
- Typical label-selection file shape:
  - `query`: full dialog text
  - `query_id`: identifier matching conversation file
  - `method` / `model`: method metadata
  - `snippets`: map of `{document_id: [ {id, text, relevance}, ... ]}`
- Item-recommendation files mirror the pattern with ranked items/snippets per query.
- LLM/MLLM based methods are further grouped by model used, experiment trial with task description for IN or EIS

Environment & Requirements
--------------------------
- Python ≥ 3.10.
- Core deps are declared in `pyproject.toml` / `requirements.txt` (PyTorch, torchvision, opencv-python, numpy, scikit-learn, tqdm, segment-anything, openai).
- Install (recommended, from repo root):
  ```
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  pip install pyyaml  # required by the sweep scripts
  ```
- Set API keys as env vars before running (do not hardcode secrets):
  - `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACE_API_KEY`, `COHERE_API_KEY`.

Running the Pipeline (single query)
-----------------------------------
`scripts/run.sh` wraps `main.py` for a single query. Required env vars:
- `DATASET` in `{fashion,movie,retail}`
- `QUERY_ID` matching a conversation id (e.g., `c6_c_6.json`)
- Optional: override `OPENAI_API_KEY` et al. in the shell.

Example:
```
export DATASET=retail
export QUERY_ID=test_1.json
source .venv/bin/activate
bash scripts/run.sh
```
The script selects dataset-specific inputs (`--image`, `--docs_json`, `--crops_dir`, `--out_links`, `--out_meta`) and forwards them to `main.py`.

You can also direct modify `run.sh` to feed different dataset/segments

Batch Over Conversations
------------------------
`scripts/run_single.py` iterates over every conversation id in `data/{dataset}/conversation/agg/pre_gt.json`, invoking `run.sh` for each.
```
python scripts/run_single.py --dataset retail
```
Progress and failures are printed per conversation.

Item-Recommendation Sweeps
--------------------------
`scripts/run_item_recommendation_experiment.py` edits `configs/item_recommendation.yaml` per method, backs it up, and runs `scripts/run_single.py`.
- Default methods: `UMBRELLA_LLM`, `UMBRELLA_VLM`, `VISON`, `UMBRELLA_VLM_LIST`, `VISON_LIST`, `UMBRELLA_LLM_LIST`, `BM25`, `DENSE`, `RERANK`, `FUSION_DENSE`.
- Usage:
```
python scripts/run_item_recommendation_experiment.py \
  --dataset fashion \
  --methods BM25 DENSE RERANK
```
Logs are written to `log/object_ranker_*.log`; the YAML config is restored after each run.

Label-Selection Sweeps
----------------------
`scripts/run_label_selection_experiment.py` edits `configs/label_selection.yaml` and runs `scripts/run_single.py` across method/`require_reason`/`use_image` combinations.
- Defaults: 
   - methods pointwise llm (`llm`) and listwise llm (`llm_list`); 
   - EIS task description `{ --require-reasons  true,false}` (False mean using IN); 
   - using item apperance (V+T) `{-- use_image true,false}`.
- CLIP methods automatically test fusion strategies; LLM methods iterate over predefined provider/model pairs.
- Usage:
```
python scripts/run_label_selection_experiment.py \
  --dataset retail \
  --methods llm llm_list\
  --require-reasons true false \
  --use-images true
```
Experiment logs are under `log/snippet_ranker_*.log`; configs are backed up and restored.

Configs
-------
- `configs/default.yaml`: default weights and thresholds for ranking (text/image weights, k, m, thresholds).
- `configs/item_recommendation.yaml`: method selection and hyperparameters for item rec (mutated by the sweep script).
- `configs/label_selection.yaml`: method and LLM/CLIP options for snippet ranking (mutated by the sweep script).
- `configs/linker.yaml`: attribute lookup-specific settings (embedding, indexing, reranking).

