"""Unit tests for the MCDA risk calculation engine."""

import pytest

from src.config import ALPHA_MAX, ALPHA_MIN, DEFAULT_WEIGHTS
from src.mcda_engine import (
    calculate_grid_mhri,
    calculate_mhri,
    clamp,
    classify_risk,
    compute_cascading_multiplier,
    normalize_min_max,
    validate_weights,
)


def test_default_weights_sum_to_one() -> None:
    weights = validate_weights(None)
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
    assert weights == DEFAULT_WEIGHTS


def test_invalid_weights_raise_value_error() -> None:
    bad_sum = {"S_p": 0.5, "G_s": 0.5, "T_e": 0.1, "H_m": 0.1, "E_d": 0.1}
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_weights(bad_sum)

    negative = {"S_p": -0.1, "G_s": 0.5, "T_e": 0.2, "H_m": 0.2, "E_d": 0.2}
    with pytest.raises(ValueError, match="negative"):
        validate_weights(negative)

    missing = {"S_p": 0.5, "G_s": 0.5}
    with pytest.raises(ValueError, match="Missing"):
        validate_weights(missing)


def test_normalize_min_max_standard_and_inverted() -> None:
    assert normalize_min_max(0.0, 0.0, 100.0) == 0.0
    assert normalize_min_max(50.0, 0.0, 100.0) == 0.5
    assert normalize_min_max(150.0, 0.0, 100.0) == 1.0
    assert normalize_min_max(-10.0, 0.0, 100.0) == 0.0

    # Inverted: distance to fault (0 km -> 1.0 risk, >= 150 km -> 0.0 risk)
    assert normalize_min_max(0.0, 0.0, 150.0, invert=True) == 1.0
    assert normalize_min_max(150.0, 0.0, 150.0, invert=True) == 0.0


def test_cascading_multiplier_not_triggered_under_two_params() -> None:
    # Only 1 parameter >= 0.70 -> alpha must be 1.0
    params = {"S_p": 0.85, "G_s": 0.69, "T_e": 0.40, "H_m": 0.30, "E_d": 0.20}
    alpha, triggered = compute_cascading_multiplier(params)
    assert alpha == 1.0
    assert triggered == ["S_p"]


def test_cascading_multiplier_triggered_scales_between_alpha_min_and_max() -> None:
    # 2 parameters >= 0.70 -> ALPHA_MIN (1.05)
    params_two = {"S_p": 0.70, "G_s": 0.75, "T_e": 0.20, "H_m": 0.20, "E_d": 0.20}
    alpha_2, trig_2 = compute_cascading_multiplier(params_two)
    assert alpha_2 == ALPHA_MIN
    assert trig_2 == ["S_p", "G_s"]

    # All 5 parameters >= 0.70 -> ALPHA_MAX (1.15)
    params_five = {"S_p": 0.90, "G_s": 0.85, "T_e": 0.80, "H_m": 0.75, "E_d": 0.70}
    alpha_5, trig_5 = compute_cascading_multiplier(params_five)
    assert alpha_5 == ALPHA_MAX
    assert len(trig_5) == 5


def test_calculate_mhri_clamped_at_one_and_zero() -> None:
    extreme_high = {"S_p": 1.0, "G_s": 1.0, "T_e": 1.0, "H_m": 1.0, "E_d": 1.0}
    res_high = calculate_mhri(extreme_high)
    assert res_high.base_mhri == 1.0
    assert res_high.alpha == ALPHA_MAX
    assert res_high.mhri == 1.0  # Clamped to 1.0
    assert res_high.risk_level_en == "Critical Risk"
    assert res_high.risk_level_tr == "Kritik Risk"

    extreme_low = {"S_p": 0.0, "G_s": 0.0, "T_e": 0.0, "H_m": 0.0, "E_d": 0.0}
    res_low = calculate_mhri(extreme_low)
    assert res_low.mhri == 0.0
    assert res_low.alpha == 1.0
    assert res_low.risk_level_en == "Low Risk"
    assert res_low.risk_level_tr == "Düşük Risk"


def test_risk_classification_boundaries() -> None:
    assert classify_risk(0.00).level_en == "Low Risk"
    assert classify_risk(0.25).level_en == "Low Risk"
    assert classify_risk(0.26).level_en == "Moderate Risk"
    assert classify_risk(0.50).level_en == "Moderate Risk"
    assert classify_risk(0.51).level_en == "High Risk"
    assert classify_risk(0.75).level_en == "High Risk"
    assert classify_risk(0.76).level_en == "Critical Risk"
    assert classify_risk(1.00).level_en == "Critical Risk"


def test_calculate_grid_mhri() -> None:
    grid = [
        {"S_p": 0.2, "G_s": 0.2, "T_e": 0.1, "H_m": 0.1, "E_d": 0.2},
        {"S_p": 0.9, "G_s": 0.8, "T_e": 0.4, "H_m": 0.5, "E_d": 0.7},
    ]
    results = calculate_grid_mhri(grid)
    assert len(results) == 2
    assert results[0].risk_level_en == "Low Risk"
    assert results[1].cascading_triggered is True
    assert results[1].mhri > results[0].mhri


def test_clamp_helper() -> None:
    assert clamp(-0.5) == 0.0
    assert clamp(1.5) == 1.0
    assert clamp(0.42) == 0.42
