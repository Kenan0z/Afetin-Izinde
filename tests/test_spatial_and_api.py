"""Integration tests for spatial analysis, recommendations, and FastAPI endpoints."""

import json

from app.main import (
    AnalyzeRequest,
    SimulateRequest,
    analyze_post,
    get_assembly_areas,
    get_faults_geojson,
    get_grid_heatmap,
    get_presets,
    get_zones,
    health_check,
    live_disasters_endpoint,
    simulate_what_if,
)
from src.recommendations import generate_recommendations
from src.spatial_analyzer import analyze_coordinate, haversine_km


def test_haversine_known_distance() -> None:
    # Ankara (39.9208, 32.8541) to Istanbul (41.0082, 28.9784) ~ 350 km
    dist = haversine_km(39.9208, 32.8541, 41.0082, 28.9784)
    assert 340.0 < dist < 365.0


def test_analyze_turkish_reference_cities_offline() -> None:
    ankara = analyze_coordinate(39.9208, 32.8541, offline_only=True)
    istanbul = analyze_coordinate(40.9792, 28.7214, offline_only=True)
    hatay = analyze_coordinate(36.2021, 36.1606, offline_only=True)
    rize = analyze_coordinate(41.0201, 40.5234, offline_only=True)

    assert ankara.mcda.risk_level_en == "Low Risk"
    assert istanbul.mcda.risk_level_en in ("High Risk", "Critical Risk")
    assert hatay.mcda.risk_level_en == "Critical Risk"
    assert rize.mcda.cascading_triggered is True  # Steep slope + hydro-meteorological load
    assert hatay.mcda.mhri > ankara.mcda.mhri


def test_recommendations_for_critical_city() -> None:
    hatay = analyze_coordinate(36.2021, 36.1606, offline_only=True)
    plan = generate_recommendations(hatay.mcda)
    assert len(plan.citizen_actions) >= 2
    assert len(plan.municipality_actions) >= 2
    assert any("Zincirleme" in item for item in plan.citizen_actions)


def test_fastapi_endpoints() -> None:
    health = health_check()
    assert health["status"] == "ok"

    presets = get_presets()
    assert len(presets) >= 4

    faults_resp = get_faults_geojson()
    assert faults_resp.status_code == 200
    faults_data = json.loads(faults_resp.body.decode("utf-8"))
    assert faults_data["type"] == "FeatureCollection"

    analyze_body = analyze_post(
        AnalyzeRequest(lat=36.2021, lon=36.1606, offline_only=True)
    )
    assert analyze_body["mcda"]["risk_level_tr"] == "Kritik Risk"
    assert len(analyze_body["recommendations"]["citizen_actions"]) >= 1
    assert "ml_prediction" in analyze_body
    assert analyze_body["ml_prediction"]["train_r2"] >= 0.85
    assert len(analyze_body["assembly_areas"]) == 3

    zones = get_zones()
    assert len(zones) == 81

    grid = get_grid_heatmap(lat=40.9792, lon=28.7214, radius_km=10.0, grid_size=5)
    assert len(grid["cells"]) == 25

    areas = get_assembly_areas(lat=40.9792, lon=28.7214, limit=2)
    assert len(areas) == 2

    sim = simulate_what_if(
        SimulateRequest(
            parameters={"S_p": 0.9, "G_s": 0.8, "T_e": 0.5, "H_m": 0.6, "E_d": 0.7}
        )
    )
    assert sim["mcda"]["cascading_triggered"] is True
    assert "ml_prediction" in sim

    live = live_disasters_endpoint(min_magnitude=2.5, offline_only=True)
    assert live["source"] == "offline_fallback"
    assert live["count"] >= 1
    first_ev = live["events"][0]
    assert len(first_ev["affected_regions"]) >= 1
    assert len(first_ev["potential_impacts"]) >= 1
