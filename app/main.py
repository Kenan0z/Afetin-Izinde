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
from src.recommendations import generate_recommendations
from src.spatial_analyzer import analyze_coordinate

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
    return {
        "metadata": asdict(report.metadata),
        "mcda": asdict(report.mcda),
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
