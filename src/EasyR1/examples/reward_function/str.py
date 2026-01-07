import re
from typing import Any, Iterable, List, Optional

from mathruler.grader import grade_answer


# Metadata
REWARD_NAME = "str"
REWARD_TYPE = "sequential"


def format_reward(response: str) -> float:
    pattern = re.compile(r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def _extract_answer(text: str) -> str:
    """Extract content inside <answer> tags; fallback to full text."""
    tag_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return tag_match.group(1).strip() if tag_match else text.strip()


def _normalize_number(token: str) -> Optional[float]:
    """Convert a numeric token to float; return None on failure."""
    try:
        return float(token.replace(",", ""))
    except Exception:
        return None


def _parse_number_list(text: str) -> List[float]:
    """Extract all numeric tokens (including scientific notation) into a list of floats."""
    number_tokens = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", text, flags=re.IGNORECASE)
    numbers: List[float] = []
    for tok in number_tokens:
        num = _normalize_number(tok)
        if num is not None:
            numbers.append(num)
    return numbers


def _relative_list_reward(pred_list: Iterable[float], gt_list: Iterable[float]) -> float:
    """Compute mean (1 - relative error) with padding/truncation; bonus for exact length match."""
    pred = list(pred_list)
    gt = list(gt_list)
    if len(gt) == 0 or len(pred) == 0:
        return 0.0

    original_len_match = len(pred) == len(gt)

    # Pad or truncate predictions to match GT length
    if len(pred) < len(gt):
        pred.extend([pred[-1]] * (len(gt) - len(pred)))
    elif len(pred) > len(gt):
        pred = pred[: len(gt)]

    scores = []
    eps = 1e-9
    for p, g in zip(pred, gt):
        rel_diff = (abs(p - g) + eps) / (abs(g) + eps)
        rel_diff = min(1.0, max(0.0, rel_diff))
        scores.append(1.0 - rel_diff)  # higher is better

    base = sum(scores) / len(scores)
    if original_len_match:
        base = min(1.0, base + 0.1)  # small bonus for correct length
    return base


def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        given_answer = _extract_answer(response)
        gt_answer = _extract_answer(ground_truth)

        # Case 1: single-letter ground truth -> fall back to original grading.
        if re.fullmatch(r"[A-Za-z]", gt_answer):
            if grade_answer(given_answer, gt_answer):
                return 1.0
            return 0.0

        # Case 2: numeric list ground truth -> relative-error reward.
        # We require predictions to present numbers inside <answer>...</answer>; otherwise 0.
        answer_tag_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        given_numbers = _parse_number_list(answer_tag_match.group(1)) if answer_tag_match else []
        gt_numbers = _parse_number_list(gt_answer)
        if len(gt_numbers) >= 1:
            if len(given_numbers) == 0:
                return 0.0  # invalid format for numeric targets
            return _relative_list_reward(given_numbers, gt_numbers)

        # Fallback: original exact/semantic grading.
        if grade_answer(given_answer, gt_answer):
            return 1.0

    except Exception:
        pass

    return 0.0

def compute_score(reward_input: dict[str, Any], format_weight: float = 0.5) -> dict[str, float]:
    format_score = format_reward(reward_input["response"])
    accuracy_score = accuracy_reward(reward_input["response"], reward_input["ground_truth"])
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
