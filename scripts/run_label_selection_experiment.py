#!/usr/bin/env python3

"""
Iterate through different method, require_reason, and use_image combinations,
modify snippet_ranker.yaml config, and run run_all_queries.py for each combination.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "label_selection.yaml"
RUN_ALL_QUERIES_SCRIPT = REPO_ROOT / "scripts" / "run_single.py"
LOG_DIR = REPO_ROOT / "log"


class TeeLogger:
    """A logger that writes to both file and console."""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(log_file, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def write(self, message: str) -> None:
        """Write message to both console and file."""
        self.original_stdout.write(message)
        self.file_handle.write(message)
        self.file_handle.flush()
    
    def flush(self) -> None:
        """Flush both outputs."""
        self.original_stdout.flush()
        self.file_handle.flush()
    
    def close(self) -> None:
        """Close the file handle."""
        if self.file_handle:
            self.file_handle.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    try:
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        print("Error: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)


def save_yaml_config(config_path: Path, config: Dict[str, Any]) -> None:
    """Save YAML config file."""
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    except Exception as e:
        print(f"Error saving config: {e}", file=sys.stderr)
        sys.exit(1)


def update_config(
    config: Dict[str, Any], 
    method_name: str, 
    require_reason: bool, 
    use_image: bool, 
    fusion_method: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Update config with method name, require_reason, use_image, provider, model, and optionally fusion_method for CLIP."""
    config = config.copy()  # Don't modify original
    
    # Update method name
    if "method" not in config:
        config["method"] = {}
    config["method"]["name"] = method_name
    
    # Update provider and model if provided
    if provider and model:
        # Map provider to api_key_env
        api_key_env_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        api_key_env = api_key_env_map.get(provider, "OPENAI_API_KEY")
        
        # Update llm section
        if "llm" not in config["method"]:
            config["method"]["llm"] = {}
        config["method"]["llm"]["provider"] = provider
        config["method"]["llm"]["model"] = model
        config["method"]["llm"]["api_key_env"] = api_key_env
        
        # Also update in method-specific sections if they exist (llm_list, llm_expansion)
        if method_name == "llm_list":
            if "llm_list" not in config["method"]:
                config["method"]["llm_list"] = {}
            config["method"]["llm_list"]["provider"] = provider
            config["method"]["llm_list"]["model"] = model
            config["method"]["llm_list"]["api_key_env"] = api_key_env
        elif method_name == "llm_expansion":
            if "llm_expansion" not in config["method"]:
                config["method"]["llm_expansion"] = {}
            config["method"]["llm_expansion"]["provider"] = provider
            config["method"]["llm_expansion"]["model"] = model
            config["method"]["llm_expansion"]["api_key_env"] = api_key_env
    
    # Update require_reason and use_image in llm section
    if "llm" not in config["method"]:
        config["method"]["llm"] = {}
    config["method"]["llm"]["require_reason"] = require_reason
    config["method"]["llm"]["use_image"] = use_image
    
    # Also update in method-specific sections if they exist (llm_list, llm_expansion)
    if method_name == "llm_list":
        if "llm_list" not in config["method"]:
            config["method"]["llm_list"] = {}
        config["method"]["llm_list"]["require_reason"] = require_reason
        config["method"]["llm_list"]["use_image"] = use_image
    elif method_name == "llm_expansion":
        if "llm_expansion" not in config["method"]:
            config["method"]["llm_expansion"] = {}
        config["method"]["llm_expansion"]["require_reason"] = require_reason
        config["method"]["llm_expansion"]["use_image"] = use_image
    
    # Update fusion_method for CLIP methods
    if method_name == "clip" and fusion_method:
        if "clip" not in config.get("method", {}):
            config["method"]["clip"] = {}
        config["method"]["clip"]["fusion_method"] = fusion_method
    
    return config


def run_experiment(
    method_name: str, 
    require_reason: bool, 
    use_image: bool, 
    dataset: str, 
    config_path: Path,
    experiment_log_file: Optional[Path] = None,
    fusion_method: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> bool:
    """Run a single experiment with given method, require_reason, use_image, provider, model, and optionally fusion_method."""
    exp_desc = f"method={method_name}, require_reason={require_reason}, use_image={use_image}"
    if fusion_method:
        exp_desc += f", fusion_method={fusion_method}"
    if provider and model:
        exp_desc += f", provider={provider}, model={model}"
    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_desc}")
    print(f"{'='*80}\n")
    
    # Load current config
    original_config = load_yaml_config(config_path)
    
    # Create backup
    backup_path = config_path.with_suffix('.yaml.backup')
    try:
        shutil.copy2(config_path, backup_path)
    except Exception as e:
        print(f"Warning: Could not create backup: {e}", file=sys.stderr)
    
    try:
        # Update config
        updated_config = update_config(original_config, method_name, require_reason, use_image, fusion_method, provider, model)
        save_yaml_config(config_path, updated_config)
        
        # Run run_all_queries.py with output captured
        print(f"Running: python {RUN_ALL_QUERIES_SCRIPT} --dataset {dataset}")
        
        # Open log file for this experiment if provided
        if experiment_log_file:
            experiment_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(experiment_log_file, 'w', encoding='utf-8') as exp_log:
                exp_desc = f"method={method_name}, require_reason={require_reason}, use_image={use_image}"
                if fusion_method:
                    exp_desc += f", fusion_method={fusion_method}"
                if provider and model:
                    exp_desc += f", provider={provider}, model={model}"
                exp_log.write(f"{'='*80}\n")
                exp_log.write(f"Experiment: {exp_desc}, dataset={dataset}\n")
                exp_log.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
                exp_log.write(f"{'='*80}\n\n")
                exp_log.write(f"Command: python {RUN_ALL_QUERIES_SCRIPT} --dataset {dataset}\n\n")
                exp_log.flush()
                
                # Run subprocess and capture output
                process = subprocess.Popen(
                    [sys.executable, str(RUN_ALL_QUERIES_SCRIPT), "--dataset", dataset],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                
                # Stream output to both console and log file
                for line in process.stdout:
                    print(line, end='', flush=True)
                    exp_log.write(line)
                    exp_log.flush()
                
                process.wait()
                returncode = process.returncode
        else:
            # Fallback: run without experiment log file
            process = subprocess.run(
                [sys.executable, str(RUN_ALL_QUERIES_SCRIPT), "--dataset", dataset],
                cwd=str(REPO_ROOT),
            )
            returncode = process.returncode
        
        if returncode != 0:
            error_msg = f"Error: run_all_queries.py failed with exit code {returncode}"
            print(error_msg, file=sys.stderr)
            if experiment_log_file:
                with open(experiment_log_file, 'a', encoding='utf-8') as exp_log:
                    exp_log.write(f"\n{error_msg}\n")
            return False
        
        exp_desc = f"method={method_name}, require_reason={require_reason}, use_image={use_image}"
        if fusion_method:
            exp_desc += f", fusion_method={fusion_method}"
        if provider and model:
            exp_desc += f", provider={provider}, model={model}"
        success_msg = f"\n✓ Completed: {exp_desc}\n"
        print(success_msg)
        if experiment_log_file:
            with open(experiment_log_file, 'a', encoding='utf-8') as exp_log:
                exp_log.write(f"\n{success_msg}")
                exp_log.write(f"Completed at: {datetime.datetime.now().isoformat()}\n")
        return True
        
    except Exception as e:
        error_msg = f"Error during experiment: {e}"
        print(error_msg, file=sys.stderr)
        if experiment_log_file:
            with open(experiment_log_file, 'a', encoding='utf-8') as exp_log:
                exp_log.write(f"\n{error_msg}\n")
        return False
    finally:
        # Restore original config
        try:
            if backup_path.exists():
                shutil.copy2(backup_path, config_path)
                backup_path.unlink()
        except Exception as e:
            warning_msg = f"Warning: Could not restore config: {e}"
            print(warning_msg, file=sys.stderr)
            if experiment_log_file:
                with open(experiment_log_file, 'a', encoding='utf-8') as exp_log:
                    exp_log.write(f"{warning_msg}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run snippet ranker experiments with different method, require_reason, and use_image combinations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fashion",
        help="Dataset name (default: fashion)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["llm", "llm_list"],
        help="Methods to test (default: llm llm_list). For CLIP, use 'clip' and fusion methods will be tested automatically.",
    )
    parser.add_argument(
        "--require-reasons",
        nargs="+",
        type=str,
        default=["true", "false"],
        help="require_reason values to test (default: true false)",
    )
    parser.add_argument(
        "--use-images",
        nargs="+",
        type=str,
        default=["true", "false"],
        help="use_image values to test (default: true false)",
    )
    args = parser.parse_args()
    
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    
    if not RUN_ALL_QUERIES_SCRIPT.exists():
        print(f"Error: run_all_queries.py not found at {RUN_ALL_QUERIES_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    
    # Parse require_reason values
    require_reason_values = []
    for val in args.require_reasons:
        val_lower = val.lower()
        if val_lower in ["true", "1", "yes"]:
            require_reason_values.append(True)
        elif val_lower in ["false", "0", "no"]:
            require_reason_values.append(False)
        else:
            print(f"Warning: Invalid require_reason value '{val}', skipping", file=sys.stderr)
    
    if not require_reason_values:
        print("Error: No valid require_reason values provided", file=sys.stderr)
        sys.exit(1)
    
    # Parse use_image values
    use_image_values = []
    for val in args.use_images:
        val_lower = val.lower()
        if val_lower in ["true", "1", "yes"]:
            use_image_values.append(True)
        elif val_lower in ["false", "0", "no"]:
            use_image_values.append(False)
        else:
            print(f"Warning: Invalid use_image value '{val}', skipping", file=sys.stderr)
    
    if not use_image_values:
        print("Error: No valid use_image values provided", file=sys.stderr)
        sys.exit(1)
    
    methods = args.methods
    # For CLIP methods, expand to include both linear and late fusion
    # Define model/provider combinations
    model_provider_combos = [
        ("openrouter", "qwen/qwen3-vl-30b-a3b-instruct"),
        ("google", "gemini-2.5-pro"),
        ("openai", "gpt-5.1"),
    ]
    
    expanded_methods = []
    fusion_methods = ["linear", "late"]
    for method_name in methods:
        if method_name == "clip":
            # Add both fusion methods for CLIP
            for fusion_method in fusion_methods:
                expanded_methods.append((method_name, fusion_method, None, None))
        else:
            # For LLM methods, iterate through model/provider combinations
            if method_name in ["llm", "llm_list", "llm_expansion"]:
                for provider, model in model_provider_combos:
                    expanded_methods.append((method_name, None, provider, model))
            else:
                # Non-LLM methods don't have fusion_method or model/provider
                expanded_methods.append((method_name, None, None, None))
    
    total_experiments = len(expanded_methods) * len(require_reason_values) * len(use_image_values)
    completed = 0
    failed = 0
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create main log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = LOG_DIR / f"snippet_ranker_experiments_{timestamp}.log"
    
    # Set up main logging to both console and file
    with TeeLogger(main_log_file) as main_logger:
        # Redirect stdout and stderr to TeeLogger
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = main_logger
        sys.stderr = main_logger
        
        try:
            print(f"\nStarting experiments:")
            print(f"  Methods: {methods}")
            if "clip" in methods:
                print(f"  CLIP fusion methods: {fusion_methods}")
            if any(m in methods for m in ["llm", "llm_list", "llm_expansion"]):
                print(f"  Model/Provider combinations: {model_provider_combos}")
            print(f"  require_reason values: {require_reason_values}")
            print(f"  use_image values: {use_image_values}")
            print(f"  Dataset: {args.dataset}")
            print(f"  Total experiments: {total_experiments}")
            print(f"  Main log file: {main_log_file}")
            print(f"  Started at: {datetime.datetime.now().isoformat()}\n")
            
            experiment_num = 0
            for method_name, fusion_method, provider, model in expanded_methods:
                for require_reason in require_reason_values:
                    for use_image in use_image_values:
                        # if method_name == "llm":
                        #     continue
                        # if method_name == "llm_list" and require_reason == True and use_image == True:
                        #     continue
                        experiment_num += 1
                        # Create experiment-specific log file
                        exp_name_parts = [method_name]
                        if fusion_method:
                            exp_name_parts.append(fusion_method)
                        if provider and model:
                            # Sanitize model name for filename (replace / with _)
                            model_safe = model.replace("/", "_")
                            exp_name_parts.append(f"{provider}_{model_safe}")
                        exp_name_parts.append(f"req{require_reason}_img{use_image}")
                        exp_name = "_".join(exp_name_parts)
                        experiment_log_file = LOG_DIR / f"snippet_ranker_{exp_name}_{args.dataset}_{timestamp}.log"
                        print(f"\n[{experiment_num}/{total_experiments}] Experiment log: {experiment_log_file}")
                        
                        success = run_experiment(
                            method_name, 
                            require_reason, 
                            use_image, 
                            args.dataset, 
                            CONFIG_PATH,
                            experiment_log_file,
                            fusion_method,
                            provider,
                            model
                        )
                        if success:
                            completed += 1
                        else:
                            failed += 1
            
            print(f"\n{'='*80}")
            print(f"Summary: {completed} completed, {failed} failed out of {total_experiments} experiments")
            print(f"  Completed at: {datetime.datetime.now().isoformat()}")
            print(f"  Main log file: {main_log_file}")
            print(f"{'='*80}\n")
            
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

