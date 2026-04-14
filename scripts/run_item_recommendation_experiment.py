#!/usr/bin/env python3

"""
Iterate through different item_recommendation methods,
modify item_recommendation.yaml config, and run run_all_queries.py for each method.
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
CONFIG_PATH = REPO_ROOT / "configs" / "item_recommendation.yaml"
RUN_ALL_QUERIES_SCRIPT = REPO_ROOT / "scripts" / "run_single.py"
LOG_DIR = REPO_ROOT / "log"

# Default methods to iterate through
DEFAULT_METHODS = [
    "UMBRELLA_LLM",
    "UMBRELLA_VLM",
    "VISON",
    "UMBRELLA_VLM_LIST",
    "VISON_LIST",
    "UMBRELLA_LLM_LIST",
    'BM25',
    'DENSE',
    'RERANK',
    'FUSION_DENSE',
]


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


def update_config(config: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """Update config with method name."""
    config = config.copy()  # Don't modify original
    
    # Update method name (it's a simple string at the top level)
    config["method"] = method_name
    
    return config


def run_experiment(
    method_name: str, 
    dataset: str, 
    config_path: Path, 
    experiment_log_file: Optional[Path] = None
) -> bool:
    """Run a single experiment with given method."""
    print(f"\n{'='*80}")
    print(f"Running experiment: method={method_name}")
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
        updated_config = update_config(original_config, method_name)
        save_yaml_config(config_path, updated_config)
        
        # Run run_all_queries.py with output captured
        print(f"Running: python {RUN_ALL_QUERIES_SCRIPT} --dataset {dataset}")
        
        # Open log file for this experiment if provided
        if experiment_log_file:
            experiment_log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(experiment_log_file, 'w', encoding='utf-8') as exp_log:
                exp_log.write(f"{'='*80}\n")
                exp_log.write(f"Experiment: method={method_name}, dataset={dataset}\n")
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
        
        success_msg = f"\n✓ Completed: method={method_name}\n"
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
        description="Run object ranker experiments with different methods"
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
        default=DEFAULT_METHODS,
        help=f"Methods to test (default: {' '.join(DEFAULT_METHODS)})",
    )
    args = parser.parse_args()
    
    if not CONFIG_PATH.exists():
        print(f"Error: Config file not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    
    if not RUN_ALL_QUERIES_SCRIPT.exists():
        print(f"Error: run_all_queries.py not found at {RUN_ALL_QUERIES_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    
    # Create log directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create main log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_file = LOG_DIR / f"object_ranker_experiments_{timestamp}.log"
    
    methods = args.methods
    total_experiments = len(methods)
    completed = 0
    failed = 0
    
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
            print(f"  Dataset: {args.dataset}")
            print(f"  Total experiments: {total_experiments}")
            print(f"  Main log file: {main_log_file}")
            print(f"  Started at: {datetime.datetime.now().isoformat()}\n")
            
            for idx, method_name in enumerate(methods, 1):
                # Create experiment-specific log file
                experiment_log_file = LOG_DIR / f"object_ranker_{method_name}_{args.dataset}_{timestamp}.log"
                print(f"\n[{idx}/{total_experiments}] Experiment log: {experiment_log_file}")
                
                success = run_experiment(method_name, args.dataset, CONFIG_PATH, experiment_log_file)
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

