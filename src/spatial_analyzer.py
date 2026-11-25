"""Spatial analysis module: Haversine distance, fault proximity, and coordinate risk query."""

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from src.config import MAX_FAULT_DISTANCE_KM, MAX_PRECIP_MM_7D, MAX_SLOPE_DEG
from src.data_loader import (
    FaultSegment,
    GeotechnicalZone,
    fetch_open_meteo_precip_7d,
    fetch_open_meteo_slope_deg,
    fetch_usgs_recent_earthquakes,
    load_active_faults,
    load_geotechnical_zones,
)
from src.mcda_engine import MHRResult, calculate_mhri, clamp, normalize_min_max

EARTH_RADIUS_KM: float = 6371.0088

SOIL_CLASS_RISK: Dict[str, float] = {
    "ZA": 0.10,
    "ZB": 0.22,
    "ZC": 0.45,
    "ZD": 0.72,
    "ZE": 0.90,
}


@dataclass(frozen=True)
class SpatialMetadata:
    """Contextual geospatial and environmental details for an analyzed coordinate."""

    lat: float
    lon: float
    nearest_fault_name: str
    nearest_fault_system: str
    nearest_fault_distance_km: float
    nearest_zone_region: str
    soil_class: str
    elevation_m: Optional[float]
    slope_deg: float
    precip_7d_mm: float
    recent_eq_count_150km: int
    data_sources: Dict[str, str]


@dataclass(frozen=True)
class SpatialRiskReport:
    """Combined spatial context and MCDA risk index result."""

    metadata: SpatialMetadata
    mcda: MHRResult


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two (lat, lon) coordinates in kilometers."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(d_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_KM * c


def point_to_segment_distance_km(
    lat: float,
    lon: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Approximate shortest distance (km) from a point to a fault line segment."""
    # Equirectangular projection centered at query latitude for local accuracy
    cos_lat = math.cos(math.radians(lat))
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(0.01, cos_lat)

    px, py = lon * km_per_deg_lon, lat * km_per_deg_lat
    ax, ay = lon1 * km_per_deg_lon, lat1 * km_per_deg_lat
    bx, by = lon2 * km_per_deg_lon, lat2 * km_per_deg_lat

    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return haversine_km(lat, lon, lat1, lon1)

    t = clamp(((px - ax) * dx + (py - ay) * dy) / seg_len_sq, 0.0, 1.0)
    proj_lat = lat1 + t * (lat2 - lat1)
    proj_lon = lon1 + t * (lon2 - lon1)
    return haversine_km(lat, lon, proj_lat, proj_lon)


def find_nearest_fault(
    lat: float,
    lon: float,
    faults: Sequence[FaultSegment],
) -> Tuple[FaultSegment, float]:
    """Find the closest fault segment and distance in km."""
    if not faults:
        raise ValueError("Fault dataset cannot be empty")

    best_fault = faults[0]
    best_dist = float("inf")

    for fault in faults:
        coords = fault.coordinates
        for idx in range(len(coords) - 1):
            lat1, lon1 = coords[idx]
            lat2, lon2 = coords[idx + 1]
            dist = point_to_segment_distance_km(lat, lon, lat1, lon1, lat2, lon2)
            if dist < best_dist:
                best_dist = dist
                best_fault = fault

    return best_fault, round(best_dist, 2)


def _interpolate_zone_values(
    lat: float,
    lon: float,
    zones: Sequence[GeotechnicalZone],
) -> Tuple[GeotechnicalZone, Dict[str, float]]:
    """Interpolate local geotechnical/exposure values using Inverse Distance Weighting."""
    if not zones:
        raise ValueError("Geotechnical zones dataset cannot be empty")

    distances = [(zone, haversine_km(lat, lon, zone.lat, zone.lon)) for zone in zones]
    nearest_zone, min_dist = min(distances, key=lambda item: item[1])

    # If very close to a reference station, use it directly
    if min_dist <= 5.0:
        return nearest_zone, {
            "soil_risk": SOIL_CLASS_RISK.get(nearest_zone.soil_class, 0.45),
            "liquefaction": nearest_zone.liquefaction_potential,
            "slope_deg": nearest_zone.slope_deg,
            "drainage_density": nearest_zone.drainage_density,
            "precip_7d_mm": nearest_zone.avg_precip_7d_mm,
            "building_density": nearest_zone.building_density,
            "building_age_score": nearest_zone.building_age_score,
        }

    weights: List[float] = []
    for _, dist in distances:
        weights.append(1.0 / ((dist + 1.0) ** 2))
    total_w = sum(weights)

    def idw(attr: str) -> float:
        return sum(w * float(getattr(z, attr)) for (z, _), w in zip(distances, weights)) / total_w

    soil_risk = (
        sum(
            w * SOIL_CLASS_RISK.get(z.soil_class, 0.45)
            for (z, _), w in zip(distances, weights)
        )
        / total_w
    )

    return nearest_zone, {
        "soil_risk": soil_risk,
        "liquefaction": idw("liquefaction_potential"),
        "slope_deg": idw("slope_deg"),
        "drainage_density": idw("drainage_density"),
        "precip_7d_mm": idw("avg_precip_7d_mm"),
        "building_density": idw("building_density"),
        "building_age_score": idw("building_age_score"),
    }


def analyze_coordinate(
    lat: float,
    lon: float,
    weights: Optional[Mapping[str, float]] = None,
    offline_only: bool = False,
) -> SpatialRiskReport:
    """Compute full MCDA Multi-Hazard Risk Index for a geographic coordinate."""
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid coordinates: lat={lat}, lon={lon}")

    faults = load_active_faults()
    zones = load_geotechnical_zones()

    nearest_fault, fault_dist_km = find_nearest_fault(lat, lon, faults)
    nearest_zone, zone_vals = _interpolate_zone_values(lat, lon, zones)

    # Open data queries (automatically fall back if offline or unreachable)
    live_precip, precip_src = fetch_open_meteo_precip_7d(lat, lon, offline_only=offline_only)
    live_slope, elevation_m, dem_src = fetch_open_meteo_slope_deg(lat, lon, offline_only=offline_only)
    eq_records, eq_src = fetch_usgs_recent_earthquakes(
        lat, lon, radius_km=150.0, offline_only=offline_only
    )

    # Count nearby earthquakes within 150 km
    nearby_eq = [
        eq for eq in eq_records if haversine_km(lat, lon, eq.lat, eq.lon) <= 150.0
    ]
    max_nearby_pga = max([eq.pga_g for eq in nearby_eq], default=0.0)

    # 1. S_p: Seismic Proximity (fault distance + historical/catalog PGA)
    fault_prox_score = normalize_min_max(fault_dist_km, 0.0, MAX_FAULT_DISTANCE_KM, invert=True)
    effective_pga = max(nearest_fault.historical_pga_g * fault_prox_score, max_nearby_pga * 0.8)
    pga_score = normalize_min_max(effective_pga, 0.05, 0.60)
    s_p = round(clamp(0.65 * fault_prox_score + 0.35 * pga_score), 4)

    # 2. G_s: Geotechnical Stability (soil class + liquefaction potential)
    g_s = round(clamp(0.40 * zone_vals["soil_risk"] + 0.60 * zone_vals["liquefaction"]), 4)

    # 3. T_e: Topographical Relief (DEM slope)
    slope_deg = live_slope if live_slope is not None else round(zone_vals["slope_deg"], 2)
    t_e = normalize_min_max(slope_deg, 0.0, MAX_SLOPE_DEG)

    # 4. H_m: Hydro-Meteorological Load (drainage density + 7-day precipitation)
    precip_mm = live_precip if live_precip is not None else round(zone_vals["precip_7d_mm"], 2)
    precip_norm = normalize_min_max(precip_mm, 0.0, MAX_PRECIP_MM_7D)
    h_m = round(clamp(0.45 * zone_vals["drainage_density"] + 0.55 * precip_norm), 4)

    # 5. E_d: Structural Exposure (building density + building age)
    e_d = round(clamp(0.50 * zone_vals["building_density"] + 0.50 * zone_vals["building_age_score"]), 4)

    params = {
        "S_p": s_p,
        "G_s": g_s,
        "T_e": t_e,
        "H_m": h_m,
        "E_d": e_d,
    }
    mcda_result = calculate_mhri(params, weights=weights)

    metadata = SpatialMetadata(
        lat=round(lat, 5),
        lon=round(lon, 5),
        nearest_fault_name=nearest_fault.name,
        nearest_fault_system=nearest_fault.fault_system,
        nearest_fault_distance_km=fault_dist_km,
        nearest_zone_region=nearest_zone.region,
        soil_class=nearest_zone.soil_class,
        elevation_m=elevation_m,
        slope_deg=slope_deg,
        precip_7d_mm=precip_mm,
        recent_eq_count_150km=len(nearby_eq),
        data_sources={
            "faults": "local_geojson",
            "geotechnical": "local_csv_idw",
            "precipitation": precip_src,
            "topography": dem_src,
            "earthquakes": eq_src,
        },
    )

    return SpatialRiskReport(metadata=metadata, mcda=mcda_result)
