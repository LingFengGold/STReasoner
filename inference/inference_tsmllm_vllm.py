# Copyright 2025 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Note: You have to install `vllm==0.8.5`.
# Note: This is a beta version, which may change in the future.
# Note: `chatts.vllm.chatts_vllm` has to be imported here first as it will register the custom ChatTS module and the multimodal processor.
# Note: Usage: `python3 -m chatts.utils.inference_tsmllm_vllm`

# [Important Note] This script is still under development and may not work as expected.

import argparse
import inference.vllm.chatts_vllm
from vllm import SamplingParams
from inference.llm_utils import LLMClient
import os
import json
from loguru import logger
import numpy as np
import multiprocessing
from typing import Any, Dict, List, Optional
import math

from transformers import AutoTokenizer
from inference.prompt_utils import get_prompt_suffix

# Time series patch size (each patch becomes one token)
TS_PATCH_SIZE = 8


# CONFIG
DEFAULT_TASK_CONFIG = {
    "alignment": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Align", "alignment_test.jsonl"),
    },
    "reasoning_forecasting": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Test", "forecasting_test.jsonl"),
    },
    "reasoning_entity": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Test", "entity_test.jsonl"),
    },
    "reasoning_etiological": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Test", "etiological_test.jsonl"),
    },
    "reasoning_correlation": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Test", "correlation_test.jsonl"),
    },
    "reasoning_causal": {
        "dataset": os.path.join("data", "ST-Bench", "ST-Causal", "causal.jsonl"),
    },
}


# Sampling parameters
sampling_params = SamplingParams(
    max_tokens=512,
    temperature=0.2
)

def answer_question_list(
    question_list: List[str],
    ts_list: List[List[np.ndarray]],
    model_path: str,
    num_gpus: int,
    num_gpus_per_process: int,
    sampling_params: SamplingParams,
) -> Dict[int, Dict[str, str]]:
    answer_dict: Dict[int, Dict[str, str]] = {}
    llm_client = LLMClient(
        model_path=model_path,
        engine="vllm-ts",
        num_gpus=num_gpus,
        gpus_per_model=num_gpus_per_process,
    )
    try:
        llm_client.wait_for_ready()
        answer_list = llm_client.llm_batch_generate(
            question_list, ts_list, sampling_params=sampling_params
        )
    finally:
        llm_client.kill()

    for idx, answer in enumerate(answer_list):
        answer_dict[idx] = {"response": answer}
    return answer_dict


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    dataset: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            dataset.append(json.loads(line))
    return dataset


def calculate_ts_tokens(ts_series: List[List[float]], patch_size: int = TS_PATCH_SIZE) -> int:
    """
    Calculate the number of tokens for time series data.
    Each time series is divided into patches, and each patch becomes one token.
    """
    total_tokens = 0
    for series in ts_series:
        # Number of patches = ceil(length / patch_size)
        ts_length = len(series)
        num_patches = math.ceil(ts_length / patch_size) if ts_length > 0 else 0
        total_tokens += num_patches
    return total_tokens


def prepare_batches(dataset: List[Dict[str, Any]], max_samples: Optional[int] = None):
    question_list: List[str] = []
    ts_list: List[List[List[float]]] = []

    def _to_float_list(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return _to_float_list(value.tolist())
        if isinstance(value, (list, tuple)):
            return [_to_float_list(v) for v in value]
        if value is None:
            return None
        return float(value)

    total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(total):
        sample = dataset[idx]
        question_list.append(sample["input"])
        ts_series: List[List[float]] = []
        for item in sample.get("timeseries", []):
            converted = _to_float_list(item)
            if converted is None:
                continue
            if not isinstance(converted, list):
                converted = [converted]
            ts_series.append(converted)
        ts_list.append(ts_series)
    return question_list, ts_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for temporal-spatial models.")
    parser.add_argument(
        "--task",
        type=str,
        default="alignment",
        choices=list(DEFAULT_TASK_CONFIG.keys()),
        help="Task name to run inference on.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSONL file. Overrides task default.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="Experiment directory name under exp/. Defaults to task name.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Total number of GPUs available for inference.",
    )
    parser.add_argument(
        "--num_gpus_per_process",
        type=int,
        default=2,
        help="Number of GPUs allocated per model replica.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to generate (useful for debugging).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="generated_answer.json",
        help="Filename for saved predictions inside experiment directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    multiprocessing.set_start_method("spawn", force=True)

    task = args.task.lower()
    config = DEFAULT_TASK_CONFIG.get(task, {})
    dataset_path = args.dataset or config.get("dataset")
    if not dataset_path:
        raise ValueError("Dataset path must be provided via --dataset or task defaults.")
    
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Use model_path as provided (supports both local path and HuggingFace model ID)
    model_path = args.model_path
    is_local_path = os.path.isdir(model_path)
    is_hf_model_id = "/" in model_path and not os.path.exists(model_path.split("/")[0])
    
    if not is_local_path and not is_hf_model_id:
        raise FileNotFoundError(f"Model path not found: {model_path}. Provide a local directory or HuggingFace model ID (e.g., 'Time-HD-Anonymous/STReasoner-8B').")

    # Set exp name
    if args.exp:
        exp_name = args.exp
    else:
        # Extract model name from path or HuggingFace model ID
        if is_hf_model_id:
            # For HuggingFace model ID like "Time-HD-Anonymous/STReasoner-8B"
            model_name = model_path.split("/")[-1]  # STReasoner-8B
        else:
            # For local path
            model_path_normalized = os.path.normpath(model_path)
            model_name = os.path.basename(model_path_normalized)
            
            if model_name == "huggingface":
                # Go up to find a more meaningful name
                # e.g., ".../qwen3_8b_grpo_stage1+2+3_w_spatial/global_step_51/actor/huggingface"
                # -> use "qwen3_8b_grpo_stage1+2+3_w_spatial"
                parent_parts = model_path_normalized.split(os.sep)
                # Find checkpoint name (usually 3 levels up from huggingface)
                if len(parent_parts) >= 4:
                    model_name = parent_parts[-4]  # qwen3_8b_grpo_stage1+2+3_w_spatial
                else:
                    model_name = parent_parts[-2] if len(parent_parts) >= 2 else "model"
        
        exp_name = f"{task}-{model_name}"
    exp_dir = os.path.join("exp", exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")

    dataset = load_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples from {dataset_path}")

    prompt_suffix = get_prompt_suffix(task)

    question_list, ts_list = prepare_batches(dataset, max_samples=args.max_samples)
    if prompt_suffix:
        question_list = [f"{question.rstrip()}\n\n{prompt_suffix}" for question in question_list]
    logger.info(f"Prepared {len(question_list)} prompts for generation")

    # Load tokenizer for counting text tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Calculate input tokens for each sample: text_tokens + ts_tokens
    input_token_counts: List[int] = []
    for idx in range(len(question_list)):
        # Text tokens
        text_tokens = len(tokenizer.encode(question_list[idx], add_special_tokens=False))
        # Time series tokens: total_length / patch_size (ceil)
        ts_tokens = calculate_ts_tokens(ts_list[idx])
        total_tokens = text_tokens + ts_tokens
        input_token_counts.append(total_tokens)
    
    total_input_tokens = sum(input_token_counts)
    avg_input_tokens = total_input_tokens / len(input_token_counts) if input_token_counts else 0
    logger.info(f"Total input tokens: {total_input_tokens}, Average: {avg_input_tokens:.1f}")

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)

    logger.info("Starting generation with vLLM...")
    answers = answer_question_list(
        question_list,
        ts_list,
        model_path=model_path,
        num_gpus=args.num_gpus,
        num_gpus_per_process=args.num_gpus_per_process,
        sampling_params=sampling_params,
    )

    generated_answer = []
    for idx, ans in answers.items():
        generated_answer.append(
            {
                "idx": idx,
                "question_text": question_list[idx],
                "response": ans["response"],
                "num_tokens": input_token_counts[idx],
            }
        )

    output_file = os.path.join(exp_dir, args.output_name)
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(generated_answer, fh, ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Token stats - Total: {total_input_tokens}, Average: {avg_input_tokens:.1f}")


if __name__ == "__main__":
    main()
