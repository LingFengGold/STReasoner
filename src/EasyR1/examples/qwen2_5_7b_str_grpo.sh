#!/bin/bash

set -x

MODEL_PATH=../base_model/Qwen2.5-7B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=../data/reasoning/etiological_rl.jsonl \
    data.val_files=../data/reasoning/etiological_test.jsonl \
    data.prompt_key=input \
    data.ts_key=timeseries \
    data.answer_key=output \
    data.rollout_batch_size=128 \
    data.val_batch_size=128 \
    data.format_prompt=./examples/format_prompt/str.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.reward.reward_function=./examples/reward_function/str.py:compute_score \
    trainer.experiment_name=qwen2_5_7b_str_grpo \
    trainer.n_gpus_per_node=8