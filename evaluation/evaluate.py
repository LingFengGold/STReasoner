import argparse
import json
import os
from typing import Any, Dict

from evaluation.evaluate_qa import (
    load_jsonl_dataset,
    load_prediction_files,
    evaluate_predictions_for_task,
)


def extract_token_stats(exp_dir: str, pattern: str = "generated_answer") -> Dict[str, Any]:
    """
    Extract token usage statistics from generated_answer.json files.
    
    Returns:
        Dictionary with total_tokens, avg_tokens, and num_samples_with_tokens
    """
    total_tokens = 0
    num_samples_with_tokens = 0
    
    if not os.path.isdir(exp_dir):
        return {}
    
    for name in os.listdir(exp_dir):
        if not name.endswith(".json") or pattern not in name:
            continue
        file_path = os.path.join(exp_dir, name)
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        
        # Support new format with metadata wrapper
        if isinstance(entries, dict) and "results" in entries:
            entries = entries["results"]
        
        for entry in entries:
            num_tokens = entry.get("num_tokens")
            if num_tokens is not None and isinstance(num_tokens, (int, float)):
                total_tokens += int(num_tokens)
                num_samples_with_tokens += 1
    
    if num_samples_with_tokens == 0:
        return {}
    
    return {
        "total_input_tokens": total_tokens,
        "avg_input_tokens": round(total_tokens / num_samples_with_tokens, 2),
        "samples_with_token_info": num_samples_with_tokens,
    }


DEFAULT_TASK_CONFIG: Dict[str, Dict[str, str]] = {
    "alignment": {
        "dataset": os.path.join("data", "alignment", "alignment_test.jsonl"),
        "task": "alignment",
    },
    "reasoning_forecasting": {
        "dataset": os.path.join("data", "reasoning", "forecasting_test.jsonl"),
        "task": "reasoning_forecasting",
    },
    "reasoning_entity": {
        "dataset": os.path.join("data", "reasoning", "entity_test.jsonl"),
        "task": "reasoning_entity",
    },
    "reasoning_etiological": {
        "dataset": os.path.join("data", "reasoning", "etiological_test.jsonl"),
        "task": "reasoning_etiological",
    },
    "reasoning_correlation": {
        "dataset": os.path.join("data", "reasoning", "correlation_test.jsonl"),
        "task": "reasoning_correlation",
    },
    "reasoning_causal": {
        "dataset": os.path.join("data", "reasoning", "causal.jsonl"),
        "task": "reasoning_causal",
    },
}


def resolve_path(path: str, repo_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n--- Evaluation Metrics ---")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("--------------------------\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QA model outputs")
    parser.add_argument(
        "--exp_path",
        type=str,
        required=True,
        help="Full path to the experiment directory (e.g., /path/to/exp/reasoning_entity-Qwen3-8B).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the evaluation dataset (JSONL). Defaults depend on --task.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task type: alignment | reasoning_forecasting | reasoning_entity | reasoning_etiological | reasoning_correlation.",
    )
    parser.add_argument(
        "--pred_pattern",
        type=str,
        default="generated_answer",
        help="Substring to match prediction files within the experiment directory.",
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=None,
        help="Project root directory. If not set, defaults to the parent of this script's directory.",
    )
    args = parser.parse_args()

    if args.repo_root is not None:
        repo_root = os.path.abspath(args.repo_root)
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use exp_path directly (resolve relative paths if needed)
    if os.path.isabs(args.exp_path):
        exp_dir = args.exp_path
    else:
        exp_dir = os.path.join(repo_root, args.exp_path)
    
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    task_key = args.task.lower()
    config = DEFAULT_TASK_CONFIG.get(task_key, {})
    dataset_path = args.dataset or config.get("dataset")
    task_type = config.get("task", task_key)

    if not dataset_path:
        raise ValueError("Dataset path must be provided via --dataset or default configuration.")

    dataset_path = resolve_path(dataset_path, repo_root)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_jsonl_dataset(dataset_path)
    predictions = load_prediction_files(exp_dir, pattern=args.pred_pattern)

    metrics = evaluate_predictions_for_task(dataset, predictions, task_type)
    
    # Add token usage statistics if available
    token_stats = extract_token_stats(exp_dir, pattern=args.pred_pattern)
    if token_stats:
        metrics.update(token_stats)
    
    print_metrics(metrics)

    output_path = os.path.join(exp_dir, "evaluation_metrics.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
