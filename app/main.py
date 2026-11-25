"""FastAPI backend server and REST API endpoints for Afetin İzinde."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import __doi__, __version__
from src.config import DEFAULT_WEIGHTS, PROJECT_NAME, RAW_DATA_DIR
from src.data_loader import load_geotechnical_zones
from src.mcda_engine import calculate_mhri
from src.ml_model import get_local_ml_model
from src.recommendations import generate_recommendations
from src.spatial_analyzer import (
    analyze_coordinate,
    analyze_grid,
    find_nearest_assembly_areas,
)

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title=PROJECT_NAME,
    description="Open-source Multi-Hazard Disaster Risk Analysis and Spatial Decision Support System",
    version=__version__,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AnalyzeRequest(BaseModel):
    """Request body for coordinate-based multi-hazard risk analysis."""

    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Optional custom MCDA weights summing to 1.0",
    )
    offline_only: bool = Field(
        default=False,
        description="Force local offline fallback datasets instead of open-data APIs",
    )


PRESET_LOCATIONS: List[Dict[str, Any]] = [
    {"name": "Ankara (Çankaya)", "lat": 39.9208, "lon": 32.8541},
    {"name": "İstanbul (Avcılar)", "lat": 40.9792, "lon": 28.7214},
    {"name": "Hatay (Antakya)", "lat": 36.2021, "lon": 36.1606},
    {"name": "Rize (Merkez)", "lat": 41.0201, "lon": 40.5234},
    {"name": "İzmir (Bayraklı)", "lat": 38.4622, "lon": 27.1672},
    {"name": "Erzincan (Merkez)", "lat": 39.7500, "lon": 39.5000},
]


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the interactive Leaflet.js map interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_path)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Return service health and academic metadata."""
    return {
        "status": "ok",
        "project": PROJECT_NAME,
        "version": __version__,
        "doi": __doi__,
        "default_weights": DEFAULT_WEIGHTS,
    }


@app.get("/api/presets")
def get_presets() -> List[Dict[str, Any]]:
    """Return preset Turkish coordinates for rapid testing and exploration."""
    return PRESET_LOCATIONS


@app.get("/api/faults")
def get_faults_geojson() -> JSONResponse:
    """Return active fault GeoJSON layer for map rendering."""
    fault_file = RAW_DATA_DIR / "active_faults.geojson"
    if not fault_file.exists():
        raise HTTPException(status_code=404, detail="Fault GeoJSON dataset not found")
    with fault_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return JSONResponse(content=data)


@app.post("/api/analyze")
def analyze_post(payload: AnalyzeRequest) -> Dict[str, Any]:
    """Run MCDA multi-hazard analysis and return risk report + action plan."""
    try:
        report = analyze_coordinate(
            lat=payload.lat,
            lon=payload.lon,
            weights=payload.weights,
            offline_only=payload.offline_only,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    actions = generate_recommendations(report.mcda)
    ml_pred = get_local_ml_model().predict(report.mcda.parameters, report.mcda.mhri)
    assembly_areas = find_nearest_assembly_areas(payload.lat, payload.lon, limit=3)

    return {
        "metadata": asdict(report.metadata),
        "mcda": asdict(report.mcda),
        "ml_prediction": asdict(ml_pred),
        "assembly_areas": assembly_areas,
        "recommendations": asdict(actions),
    }


class SimulateRequest(BaseModel):
    """Request body for What-If scenario simulation."""

    parameters: Dict[str, float]
    weights: Optional[Dict[str, float]] = None


@app.get("/api/zones")
def get_zones() -> List[Dict[str, Any]]:
    """Return all 81 Turkish province reference zones."""
    return [
        {"name": z.region, "lat": z.lat, "lon": z.lon, "soil_class": z.soil_class}
        for z in load_geotechnical_zones()
    ]


@app.get("/api/grid")
def get_grid_heatmap(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    radius_km: float = Query(default=15.0, ge=2.0, le=60.0),
    grid_size: int = Query(default=7, ge=3, le=15),
) -> Dict[str, Any]:
    """Compute an N x N spatial risk heatmap grid around the target coordinate."""
    cells = analyze_grid(lat, lon, radius_km=radius_km, grid_size=grid_size)
    return {"center": {"lat": lat, "lon": lon}, "grid_size": grid_size, "cells": cells}


@app.get("/api/assembly-areas")
def get_assembly_areas(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    limit: int = Query(default=3, ge=1, le=10),
) -> List[Dict[str, Any]]:
    """Return nearest emergency assembly areas."""
    return find_nearest_assembly_areas(lat, lon, limit=limit)


@app.post("/api/simulate")
def simulate_what_if(payload: SimulateRequest) -> Dict[str, Any]:
    """Run What-If simulation with custom parameters and/or weights."""
    try:
        mcda = calculate_mhri(payload.parameters, weights=payload.weights)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ml_pred = get_local_ml_model().predict(mcda.parameters, mcda.mhri)
    actions = generate_recommendations(mcda)
    return {
        "mcda": asdict(mcda),
        "ml_prediction": asdict(ml_pred),
        "recommendations": asdict(actions),
    }


@app.get("/api/analyze")
def analyze_get(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    offline_only: bool = Query(default=False),
) -> Dict[str, Any]:
    """GET convenience endpoint for coordinate-based risk analysis."""
    return analyze_post(
        AnalyzeRequest(lat=lat, lon=lon, offline_only=offline_only)
    )
