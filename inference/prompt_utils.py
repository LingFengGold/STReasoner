import json
import os
from typing import Dict, List, Optional

_PROMPT_CONFIG_CACHE: Optional[Dict[str, Dict[str, str]]] = None

PROMPT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prompt.json"
)


def _load_prompt_config() -> Dict[str, Dict[str, str]]:
    global _PROMPT_CONFIG_CACHE
    if _PROMPT_CONFIG_CACHE is None:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        if not isinstance(config, dict):
            raise ValueError("prompt.json must contain a JSON object at the top level.")
        _PROMPT_CONFIG_CACHE = config
    return _PROMPT_CONFIG_CACHE


def available_prompt_tasks() -> List[str]:
    return sorted(_load_prompt_config().keys())


def get_prompt_suffix(task: str) -> str:
    config = _load_prompt_config()
    entry = config.get(task)
    if not entry:
        available = ", ".join(available_prompt_tasks())
        raise ValueError(f"Unknown task '{task}'. Available tasks: {available}")
    suffix = entry.get("prompt")
    if not isinstance(suffix, str):
        raise ValueError(
            f"Prompt configuration for task '{task}' must include a string value under 'prompt'."
        )
    return suffix.strip()


AVAILABLE_PROMPT_TASKS = tuple(available_prompt_tasks())


