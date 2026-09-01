from __future__ import annotations

from typing import Any


def post_calibration_enabled(config: dict[str, Any]) -> bool:
    evaluation = config.get("EveNetEvaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("EveNetEvaluation must be a mapping.")

    enabled = evaluation.get("post_calibration", True)
    if not isinstance(enabled, bool):
        raise ValueError("EveNetEvaluation.post_calibration must be true or false.")
    return enabled


def qi_region_to_signals(
    config: dict[str, Any],
    fallback_regions: list[str] | None = None,
) -> dict[str, list[str]]:
    evaluation = config.get("QIAnalysis") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("QIAnalysis must be a mapping.")

    raw_mapping = evaluation.get("region_to_signals")
    if raw_mapping is None and fallback_regions:
        return {str(region): [str(region)] for region in fallback_regions}
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("QIAnalysis.region_to_signals must be a non-empty mapping.")

    mapping: dict[str, list[str]] = {}
    for raw_region, raw_signals in raw_mapping.items():
        region = str(raw_region)
        if not region.startswith("Ztautau_"):
            raise ValueError(f"Unsupported QI region '{region}'.")
        if not isinstance(raw_signals, list) or not raw_signals:
            raise ValueError(f"QI region '{region}' must contain a non-empty signal list.")
        signals = [str(signal) for signal in raw_signals]
        if any(not signal.startswith("Ztautau_") for signal in signals):
            raise ValueError(f"QI region '{region}' contains an unsupported signal label.")
        if len(signals) != len(set(signals)):
            raise ValueError(f"QI region '{region}' contains duplicate signal labels.")
        mapping[region] = signals
    return mapping
