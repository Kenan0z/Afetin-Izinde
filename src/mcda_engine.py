"""MCDA (Multi-Criteria Decision Analysis) risk calculation engine for Afetin İzinde.

Implements the Multi-Hazard Risk Index (MHRI) mathematical model:
    MHRI_i = clamp(alpha * Sum_{k=1}^{5} (w_k * x_{i,k}), 0.0, 1.0)
where:
    - x_{i,k} in [0.0, 1.0] for k in {S_p, G_s, T_e, H_m, E_d}
    - Sum(w_k) = 1.0
    - alpha in [1.05, 1.15] when >= 2 parameters are >= 0.70, else 1.0
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.config import (
    ALPHA_MAX,
    ALPHA_MIN,
    CASCADING_MIN_PARAMS,
    CASCADING_THRESHOLD,
    DEFAULT_WEIGHTS,
    PARAM_KEYS,
    RISK_BANDS,
)


@dataclass(frozen=True)
class RiskClassification:
    """Risk classification band result."""

    level_en: str
    level_tr: str
    color_hex: str


@dataclass(frozen=True)
class MHRResult:
    """Complete output of an MHRI calculation for a coordinate or grid cell."""

    parameters: Dict[str, float]
    weights: Dict[str, float]
    contributions: Dict[str, float]
    base_mhri: float
    alpha: float
    cascading_triggered: bool
    triggered_parameters: List[str]
    mhri: float
    risk_level_en: str
    risk_level_tr: str
    color_hex: str


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a numeric value to the inclusive interval [low, high]."""
    return float(max(low, min(high, value)))


def normalize_min_max(
    value: float,
    min_val: float,
    max_val: float,
    invert: bool = False,
) -> float:
    """Normalize a physical value into [0.0, 1.0] using min-max scaling.

    Args:
        value: Raw measurement.
        min_val: Lower bound corresponding to 0.0 (or 1.0 if inverted).
        max_val: Upper bound corresponding to 1.0 (or 0.0 if inverted).
        invert: If True, returns 1.0 - normalized (useful for distance-to-hazard).
    """
    if max_val <= min_val:
        raise ValueError("max_val must be strictly greater than min_val")
    ratio = (float(value) - float(min_val)) / (float(max_val) - float(min_val))
    normalized = clamp(ratio, 0.0, 1.0)
    return round(1.0 - normalized if invert else normalized, 4)


def validate_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    """Validate and return a clean weight dictionary summing to 1.0."""
    if weights is None:
        return dict(DEFAULT_WEIGHTS)

    missing = [k for k in PARAM_KEYS if k not in weights]
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")

    cleaned: Dict[str, float] = {}
    for key in PARAM_KEYS:
        val = float(weights[key])
        if val < 0.0:
            raise ValueError(f"Weight for '{key}' cannot be negative: {val}")
        cleaned[key] = val

    total = sum(cleaned.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0 (got {total:.6f})")

    return cleaned


def validate_parameters(params: Mapping[str, float]) -> Dict[str, float]:
    """Validate that all 5 MCDA parameters exist and clamp them to [0.0, 1.0]."""
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"Missing parameter keys: {missing}")

    return {key: round(clamp(float(params[key]), 0.0, 1.0), 4) for key in PARAM_KEYS}


def compute_cascading_multiplier(
    params: Mapping[str, float],
    threshold: float = CASCADING_THRESHOLD,
    min_params: int = CASCADING_MIN_PARAMS,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> Tuple[float, List[str]]:
    """Compute the cascading interaction multiplier (alpha) and triggered parameters.

    When two or more normalized parameters meet or exceed `threshold` (default 0.70),
    alpha scales linearly within [alpha_min, alpha_max] ([1.05, 1.15]) based on the
    number of triggered parameters. Otherwise, alpha is 1.0.
    """
    triggered = [key for key in PARAM_KEYS if float(params[key]) >= threshold]
    count = len(triggered)

    if count < min_params:
        return 1.0, triggered

    max_possible = len(PARAM_KEYS)
    if max_possible <= min_params:
        return alpha_max, triggered

    ratio = (count - min_params) / (max_possible - min_params)
    alpha = alpha_min + ratio * (alpha_max - alpha_min)
    return round(clamp(alpha, alpha_min, alpha_max), 4), triggered


def classify_risk(mhri: float) -> RiskClassification:
    """Classify a normalized MHRI score in [0.0, 1.0] into a discrete risk band."""
    score = clamp(mhri, 0.0, 1.0)
    for idx, (low, high, label_en, label_tr, color) in enumerate(RISK_BANDS):
        if idx == 0 and score <= high:
            return RiskClassification(label_en, label_tr, color)
        if low < score <= high:
            return RiskClassification(label_en, label_tr, color)
    # Fallback for boundary 1.0
    _, _, label_en, label_tr, color = RISK_BANDS[-1]
    return RiskClassification(label_en, label_tr, color)


def calculate_mhri(
    params: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
) -> MHRResult:
    """Calculate the composite Multi-Hazard Risk Index (MHRI) for a single cell/point."""
    clean_params = validate_parameters(params)
    clean_weights = validate_weights(weights)

    contributions = {
        key: round(clean_weights[key] * clean_params[key], 4)
        for key in PARAM_KEYS
    }
    base_mhri = round(float(sum(contributions.values())), 4)

    alpha, triggered = compute_cascading_multiplier(clean_params)
    final_mhri = round(clamp(base_mhri * alpha, 0.0, 1.0), 4)
    classification = classify_risk(final_mhri)

    return MHRResult(
        parameters=clean_params,
        weights=clean_weights,
        contributions=contributions,
        base_mhri=base_mhri,
        alpha=alpha,
        cascading_triggered=len(triggered) >= CASCADING_MIN_PARAMS,
        triggered_parameters=triggered,
        mhri=final_mhri,
        risk_level_en=classification.level_en,
        risk_level_tr=classification.level_tr,
        color_hex=classification.color_hex,
    )


def calculate_grid_mhri(
    grid_params: Sequence[Mapping[str, float]],
    weights: Optional[Mapping[str, float]] = None,
) -> List[MHRResult]:
    """Compute MHRI results for a sequence of grid cells."""
    clean_weights = validate_weights(weights)
    return [calculate_mhri(cell, clean_weights) for cell in grid_params]
