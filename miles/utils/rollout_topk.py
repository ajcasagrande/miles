"""Sampling constraints for recording rollout top-k candidates."""

import math
from collections.abc import Mapping
from typing import Any


def rollout_topk_width(args: Any) -> int:
    """Number of candidate log-probs recorded per generated token; 0 when recording is off."""
    return getattr(args, "rollout_top_logprobs_num", 0)


def validate_rollout_topk_args(args: Any) -> None:
    """Reject rollout top-k settings whose recorded candidates cannot match the sampler."""
    k = rollout_topk_width(args)
    mode = getattr(args, "rollout_sampling_logprobs_mode", "selected")
    filtered = args.rollout_top_p < 1.0 or args.rollout_top_k > 0
    if k < 0:
        raise ValueError(f"--rollout-top-logprobs-num must be non-negative, got {k}")
    if mode == "support":
        if not filtered:
            raise ValueError("--rollout-sampling-logprobs-mode support requires filtered rollout sampling")
        if k < args.rollout_top_k:
            raise ValueError(
                "--rollout-sampling-logprobs-mode support requires --rollout-top-logprobs-num >= --rollout-top-k "
                "so the recorded candidates can hold the whole sampling support"
            )
    elif k and filtered:
        raise ValueError(
            "Recording --rollout-top-logprobs-num candidates under filtered rollout sampling requires "
            "--rollout-sampling-logprobs-mode support; selected mode records pre-filter candidates"
        )
    opd_student_top_k = getattr(args, "use_opd", False) and (getattr(args, "opd_log_prob_top_k", 0) or 0) > 0
    if k and opd_student_top_k and getattr(args, "opd_top_k_strategy", "only-student") != "only-teacher":
        raise ValueError("--rollout-top-logprobs-num cannot be combined with OPD student top-k log-probs")


def validate_rollout_topk_sampling(sampling: Mapping[str, Any], *, temperature: float, candidate_count: int) -> None:
    """Require a bounded support covered by the recorded candidates.

    Filtered generation requests SGLang's post-filter support probabilities.
    The candidate count must cover the complete realized support.
    """
    if sampling.get("temperature", temperature) != temperature or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Rollout top-k collection requires the same positive rollout temperature on every generation call"
        )
    top_p = sampling.get("top_p", 1.0)
    top_k = sampling.get("top_k", -1)
    if not 0 < top_p <= 1 or (top_k != -1 and not 0 < top_k <= candidate_count):
        raise ValueError("Rollout top-k collection requires top_p in (0, 1] and top_k=-1 or 1..k")
    if top_p < 1 and top_k == -1:
        raise ValueError("Rollout top-k collection requires positive top_k with top_p filtering to bound the support")
    if sampling.get("min_p", 0.0) != 0.0:
        raise ValueError("Rollout top-k collection requires min_p=0.0")
    for key in ("json_schema", "regex", "ebnf", "structural_tag", "custom_logit_processor", "logit_bias"):
        if sampling.get(key):
            raise ValueError(f"Rollout top-k collection does not support constrained/custom sampling ({key})")
    response_format = sampling.get("response_format")
    if response_format and (not isinstance(response_format, Mapping) or response_format.get("type", "text") != "text"):
        raise ValueError("Rollout top-k collection does not support constrained response_format")
    tool_choice = sampling.get("tool_choice", "auto")
    if tool_choice not in (None, "auto", "none"):
        raise ValueError("Rollout top-k collection does not support constrained tool_choice")
    if tool_choice != "none" and any(tool.get("function", {}).get("strict") for tool in sampling.get("tools") or []):
        raise ValueError("Rollout top-k collection does not support strict tool schemas")
