"""Open-source keyless data loader with automatic offline fallback.

Data sources:
- Local GeoJSON/CSV datasets in `data/raw/` (always available offline)
- Open-Meteo Forecast & Elevation APIs (keyless, free)
- USGS FDSN Earthquake Catalog API (keyless, free)
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from src.config import ENDPOINTS, RAW_DATA_DIR


@dataclass(frozen=True)
class FaultSegment:
    """Active fault polyline segment loaded from GeoJSON."""

    name: str
    fault_system: str
    slip_rate_mm_yr: float
    historical_pga_g: float
    coordinates: List[Tuple[float, float]]  # (lat, lon) pairs


@dataclass(frozen=True)
class GeotechnicalZone:
    """Reference geotechnical, topographic, hydro-meteorological, and exposure zone."""

    region: str
    lat: float
    lon: float
    radius_km: float
    soil_class: str
    liquefaction_potential: float
    slope_deg: float
    drainage_density: float
    avg_precip_7d_mm: float
    building_density: float
    building_age_score: float


@dataclass(frozen=True)
class EarthquakeRecord:
    """Earthquake catalog entry from USGS/AFAD or local fallback CSV."""

    event_id: str
    date: str
    lat: float
    lon: float
    magnitude: float
    depth_km: float
    pga_g: float
    location: str


@dataclass(frozen=True)
class AssemblyArea:
    """Emergency assembly / safe evacuation open area."""

    name: str
    city: str
    lat: float
    lon: float
    capacity: int
    area_type: str


def load_active_faults(path: Optional[Path] = None) -> List[FaultSegment]:
    """Load active fault segments from local GeoJSON."""
    target = path or (RAW_DATA_DIR / "active_faults.geojson")
    with target.open("r", encoding="utf-8") as fh:
        geojson = json.load(fh)

    segments: List[FaultSegment] = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        geom = feature.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        # GeoJSON coordinates are [lon, lat]; convert to (lat, lon)
        coords = [(float(pt[1]), float(pt[0])) for pt in geom.get("coordinates", [])]
        if len(coords) < 2:
            continue
        segments.append(
            FaultSegment(
                name=str(props.get("name", "Unknown Fault")),
                fault_system=str(props.get("fault_system", "UNKNOWN")),
                slip_rate_mm_yr=float(props.get("slip_rate_mm_yr", 5.0)),
                historical_pga_g=float(props.get("historical_pga_g", 0.25)),
                coordinates=coords,
            )
        )
    return segments


def load_geotechnical_zones(path: Optional[Path] = None) -> List[GeotechnicalZone]:
    """Load reference geotechnical and exposure zones from local CSV."""
    target = path or (RAW_DATA_DIR / "geotechnical_zones.csv")
    zones: List[GeotechnicalZone] = []
    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            zones.append(
                GeotechnicalZone(
                    region=str(row["region"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    radius_km=float(row["radius_km"]),
                    soil_class=str(row["soil_class"]),
                    liquefaction_potential=float(row["liquefaction_potential"]),
                    slope_deg=float(row["slope_deg"]),
                    drainage_density=float(row["drainage_density"]),
                    avg_precip_7d_mm=float(row["avg_precip_7d_mm"]),
                    building_density=float(row["building_density"]),
                    building_age_score=float(row["building_age_score"]),
                )
            )
    return zones


def load_historical_earthquakes(path: Optional[Path] = None) -> List[EarthquakeRecord]:
    """Load historical earthquake catalog from local CSV."""
    target = path or (RAW_DATA_DIR / "historical_earthquakes.csv")
    records: List[EarthquakeRecord] = []
    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(
                EarthquakeRecord(
                    event_id=str(row["event_id"]),
                    date=str(row["date"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    magnitude=float(row["magnitude"]),
                    depth_km=float(row["depth_km"]),
                    pga_g=float(row["pga_g"]),
                    location=str(row["location"]),
                )
            )
    return records


def load_assembly_areas(path: Optional[Path] = None) -> List[AssemblyArea]:
    """Load emergency assembly areas from local CSV."""
    target = path or (RAW_DATA_DIR / "assembly_areas.csv")
    areas: List[AssemblyArea] = []
    with target.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            areas.append(
                AssemblyArea(
                    name=str(row["name"]),
                    city=str(row["city"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    capacity=int(row["capacity"]),
                    area_type=str(row["type"]),
                )
            )
    return areas


def fetch_open_meteo_precip_7d(
    lat: float,
    lon: float,
    offline_only: bool = False,
) -> Tuple[Optional[float], str]:
    """Fetch 7-day precipitation sum (mm) from Open-Meteo without an API key.

    Returns (precip_mm, source_tag). On failure or offline mode, returns (None, 'offline_fallback').
    """
    if offline_only:
        return None, "offline_fallback"

    params = {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "daily": "precipitation_sum",
        "past_days": 5,
        "forecast_days": 2,
        "timezone": "UTC",
    }
    try:
        with httpx.Client(timeout=ENDPOINTS.http_timeout_seconds) as client:
            resp = client.get(ENDPOINTS.open_meteo_forecast, params=params)
            resp.raise_for_status()
            data = resp.json()
            daily_precip = data.get("daily", {}).get("precipitation_sum", [])
            valid = [float(x) for x in daily_precip if x is not None]
            if valid:
                return round(sum(valid), 2), "open-meteo"
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return None, "offline_fallback"


def fetch_open_meteo_slope_deg(
    lat: float,
    lon: float,
    offline_only: bool = False,
) -> Tuple[Optional[float], Optional[float], str]:
    """Estimate local terrain slope (degrees) using Open-Meteo Elevation API.

    Samples center + 4 cardinal points (~300m offset). Returns (slope_deg, elevation_m, source).
    """
    if offline_only:
        return None, None, "offline_fallback"

    delta = 0.0027  # ~300 m in latitude
    lats = [lat, lat + delta, lat - delta, lat, lat]
    lons = [lon, lon, lon, lon + delta, lon - delta]
    params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats),
        "longitude": ",".join(f"{x:.5f}" for x in lons),
    }
    try:
        with httpx.Client(timeout=ENDPOINTS.http_timeout_seconds) as client:
            resp = client.get(ENDPOINTS.open_meteo_elevation, params=params)
            resp.raise_for_status()
            elevs = resp.json().get("elevation", [])
            if len(elevs) == 5:
                center_elev = float(elevs[0])
                dz_ns = abs(float(elevs[1]) - float(elevs[2])) / 600.0
                dz_ew = abs(float(elevs[3]) - float(elevs[4])) / 600.0
                import math

                slope_rad = math.atan(math.hypot(dz_ns, dz_ew))
                slope_deg = round(math.degrees(slope_rad), 2)
                return slope_deg, round(center_elev, 1), "open-meteo-dem"
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        pass
    return None, None, "offline_fallback"


def fetch_usgs_recent_earthquakes(
    lat: float,
    lon: float,
    radius_km: float = 150.0,
    min_magnitude: float = 4.0,
    offline_only: bool = False,
) -> Tuple[List[EarthquakeRecord], str]:
    """Query USGS FDSN earthquake catalog (keyless) with offline CSV fallback."""
    if offline_only:
        return load_historical_earthquakes(), "offline_fallback"

    params = {
        "format": "geojson",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "maxradiuskm": radius_km,
        "minmagnitude": min_magnitude,
        "limit": 20,
    }
    try:
        with httpx.Client(timeout=ENDPOINTS.http_timeout_seconds) as client:
            resp = client.get(ENDPOINTS.usgs_earthquake_api, params=params)
            resp.raise_for_status()
            features = resp.json().get("features", [])
            records: List[EarthquakeRecord] = []
            for feat in features:
                props = feat.get("properties", {})
                coords = feat.get("geometry", {}).get("coordinates", [0.0, 0.0, 10.0])
                mag = float(props.get("mag") or 0.0)
                # Approximate PGA proxy from magnitude if not provided
                pga_proxy = round(min(0.8, 0.02 * (2.0 ** (mag - 4.0))), 3)
                records.append(
                    EarthquakeRecord(
                        event_id=str(feat.get("id", "usgs-event")),
                        date=str(props.get("time", "")),
                        lat=float(coords[1]),
                        lon=float(coords[0]),
                        magnitude=mag,
                        depth_km=float(coords[2]) if len(coords) > 2 else 10.0,
                        pga_g=pga_proxy,
                        location=str(props.get("place", "USGS Event")),
                    )
                )
            return records, "usgs-fdsn"
    except (httpx.HTTPError, ValueError, KeyError):
        return load_historical_earthquakes(), "offline_fallback"
