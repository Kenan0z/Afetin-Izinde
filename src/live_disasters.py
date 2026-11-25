"""Real-time disaster feed and magnitude-based impact interpreter.

Fetches keyless open-source disaster events (USGS FDSN Earthquake API for Turkey)
with automatic offline fallback, identifies affected Turkish provinces from the
81-province dataset, and generates severity-based impact interpretations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import httpx

from src.config import ENDPOINTS
from src.data_loader import EarthquakeRecord, load_geotechnical_zones, load_historical_earthquakes
from src.spatial_analyzer import haversine_km


@dataclass(frozen=True)
class LiveDisasterAlert:
    """Enriched real-time disaster alert with affected areas and impact analysis."""

    event_id: str
    hazard_type: str
    title: str
    magnitude: float
    depth_km: float
    lat: float
    lon: float
    time_utc: str
    impact_radius_km: float
    severity_level: str
    severity_color: str
    affected_regions: List[str]
    potential_impacts: List[str]
    summary_tr: str


def _estimate_impact_radius_km(magnitude: float, depth_km: float) -> float:
    """Estimate felt/impact radius (km) from magnitude and focal depth."""
    if magnitude >= 6.5:
        base = 240.0
    elif magnitude >= 6.0:
        base = 170.0
    elif magnitude >= 5.0:
        base = 100.0
    elif magnitude >= 4.0:
        base = 55.0
    else:
        base = 25.0
    # Shallow earthquakes (< 12 km) produce stronger localized surface shaking
    depth_factor = 1.15 if depth_km < 12.0 else 1.0
    return round(base * depth_factor, 1)


def _classify_severity(magnitude: float) -> Tuple[str, str]:
    """Return (severity_label_tr, color_hex) for an earthquake magnitude."""
    if magnitude >= 6.0:
        return "Kritik / Yıkıcı", "#ef4444"
    if magnitude >= 5.0:
        return "Yüksek / Şiddetli", "#f97316"
    if magnitude >= 4.0:
        return "Orta / Hissedilir", "#eab308"
    return "Düşük / Mikro", "#38bdf8"


def _build_magnitude_impacts(
    magnitude: float,
    depth_km: float,
    affected_zones: List[Any],
) -> Tuple[List[str], str]:
    """Generate specific physical/structural impact warnings based on magnitude & local soil."""
    impacts: List[str] = []

    if magnitude >= 6.0:
        impacts.append("Geniş alanda ciddi yapısal hasar ve özellikle 2000 öncesi yapılarda göçme riski.")
        impacts.append("Enerji, su, doğalgaz ve ulaşım altyapısında kesintiler beklenebilir.")
        impacts.append("Güçlü artçı sarsıntı zinciri nedeniyle hasarlı binalara kesinlikle girilmemelidir.")
    elif magnitude >= 5.0:
        impacts.append("Zayıf zeminli ve yaşlı yapılarda duvar çatlakları ve yapısal olmayan elemanlarda devrilme riski.")
        impacts.append("Kısa süreli iletişim yoğunluğu ve lokal elektrik kesintileri yaşanabilir.")
    elif magnitude >= 4.0:
        impacts.append("Merkez üssüne yakın bölgelerde belirgin sarsıntı ve sabitlenmemiş eşyalarda düşme riski.")
        impacts.append("Yapısal hasar beklenmez ancak hasarlı binalarda kontrol önerilir.")
    else:
        impacts.append("Hafif sarsıntı; genel olarak yapısal bir sorun oluşturmaz.")

    if depth_km < 12.0 and magnitude >= 4.0:
        impacts.append(f"Sığ odaklı ({depth_km:.1f} km) olduğu için yüzeyde beklenenden daha sert hissedilebilir.")

    # Check secondary cascading hazards in affected provinces
    high_liq = [z.region for z in affected_zones if z.liquefaction_potential >= 0.70]
    steep = [z.region for z in affected_zones if z.slope_deg >= 20.0]

    if magnitude >= 4.8 and high_liq:
        impacts.append(
            f"Sıvılaşma riski: Alüvyon/yumuşak zeminli {', '.join(high_liq[:3])} hattında zemin oturması tetiklenebilir."
        )
    if magnitude >= 4.5 and steep:
        impacts.append(
            f"Heyelan / kaya düşmesi riski: Dik eğimli {', '.join(steep[:3])} kesimlerinde yamaç duraysızlığı oluşabilir."
        )

    region_text = ", ".join(z.region for z in affected_zones[:3]) if affected_zones else "Merkez üssü çevresi"
    summary_tr = (
        f"M{magnitude:.1f} büyüklüğündeki sarsıntı ({depth_km:.1f} km derinlik) "
        f"başta {region_text} olmak üzere ~{int(_estimate_impact_radius_km(magnitude, depth_km))} km yarıçapında etki alanına sahiptir."
    )
    return impacts, summary_tr


def fetch_turkey_live_earthquakes(
    min_magnitude: float = 2.5,
    days_back: int = 7,
    offline_only: bool = False,
) -> Tuple[List[EarthquakeRecord], str]:
    """Fetch recent earthquakes across Turkey from USGS FDSN (keyless) with offline fallback."""
    if offline_only:
        return load_historical_earthquakes(), "offline_fallback"

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days_back)

    params = {
        "format": "geojson",
        "starttime": start_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "minlatitude": 35.5,
        "maxlatitude": 42.5,
        "minlongitude": 25.5,
        "maxlongitude": 45.0,
        "minmagnitude": min_magnitude,
        "orderby": "time",
        "limit": 15,
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
                ts_ms = props.get("time")
                time_str = (
                    datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    if ts_ms
                    else ""
                )
                pga_proxy = round(min(0.8, 0.02 * (2.0 ** max(0.0, mag - 4.0))), 3)
                records.append(
                    EarthquakeRecord(
                        event_id=str(feat.get("id", "usgs-live")),
                        date=time_str,
                        lat=float(coords[1]),
                        lon=float(coords[0]),
                        magnitude=mag,
                        depth_km=float(coords[2]) if len(coords) > 2 else 10.0,
                        pga_g=pga_proxy,
                        location=str(props.get("place", "Türkiye")),
                    )
                )
            return records, "usgs-live"
    except (httpx.HTTPError, ValueError, KeyError):
        return load_historical_earthquakes(), "offline_fallback"


def get_live_disaster_alerts(
    min_magnitude: float = 2.5,
    offline_only: bool = False,
) -> Dict[str, Any]:
    """Return enriched live disaster alerts with affected regions and magnitude-based impacts."""
    records, source = fetch_turkey_live_earthquakes(
        min_magnitude=min_magnitude,
        offline_only=offline_only,
    )
    zones = load_geotechnical_zones()
    alerts: List[Dict[str, Any]] = []

    for rec in records:
        radius_km = _estimate_impact_radius_km(rec.magnitude, rec.depth_km)
        severity_label, severity_color = _classify_severity(rec.magnitude)

        nearby_zones = sorted(
            [
                (z, haversine_km(rec.lat, rec.lon, z.lat, z.lon))
                for z in zones
                if haversine_km(rec.lat, rec.lon, z.lat, z.lon) <= radius_km
            ],
            key=lambda item: item[1],
        )
        affected_zone_objs = [z for z, _ in nearby_zones]
        affected_names = [f"{z.region} ({dist:.0f} km)" for z, dist in nearby_zones[:5]]
        if not affected_names:
            closest = min(zones, key=lambda z: haversine_km(rec.lat, rec.lon, z.lat, z.lon))
            closest_dist = haversine_km(rec.lat, rec.lon, closest.lat, closest.lon)
            affected_names = [f"{closest.region} ({closest_dist:.0f} km)"]
            affected_zone_objs = [closest]

        impacts, summary_tr = _build_magnitude_impacts(rec.magnitude, rec.depth_km, affected_zone_objs)

        alert = LiveDisasterAlert(
            event_id=rec.event_id,
            hazard_type="Deprem",
            title=f"M{rec.magnitude:.1f} — {rec.location}",
            magnitude=round(rec.magnitude, 1),
            depth_km=round(rec.depth_km, 1),
            lat=round(rec.lat, 4),
            lon=round(rec.lon, 4),
            time_utc=rec.date,
            impact_radius_km=radius_km,
            severity_level=severity_label,
            severity_color=severity_color,
            affected_regions=affected_names,
            potential_impacts=impacts,
            summary_tr=summary_tr,
        )
        alerts.append(alert.__dict__)

    return {
        "source": source,
        "count": len(alerts),
        "events": alerts,
    }
