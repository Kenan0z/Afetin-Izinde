"""Configuration constants, MCDA weights, thresholds, and data source settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Project directory layout
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# Academic & project metadata
PROJECT_NAME: str = "Afetin İzinde"
PROJECT_DOI: str = "10.5281/zenodo.24514467"
PROJECT_ORCID: str = "0009-0005-6710-3009"

# Parameter keys (5 core MCDA dimensions)
PARAM_KEYS: Tuple[str, ...] = ("S_p", "G_s", "T_e", "H_m", "E_d")

# Default expert weights (must sum to 1.0)
DEFAULT_WEIGHTS: Dict[str, float] = {
    "S_p": 0.30,  # Seismic Proximity
    "G_s": 0.25,  # Geotechnical Stability
    "T_e": 0.15,  # Topographical Relief
    "H_m": 0.15,  # Hydro-Meteorological Load
    "E_d": 0.15,  # Structural Exposure
}

# Cascading Interaction Multiplier (alpha) configuration
CASCADING_THRESHOLD: float = 0.70
CASCADING_MIN_PARAMS: int = 2
ALPHA_MIN: float = 1.05
ALPHA_MAX: float = 1.15

# Risk classification bands: (min_inclusive, max_inclusive, en_label, tr_label, color_hex)
RISK_BANDS: Tuple[Tuple[float, float, str, str, str], ...] = (
    (0.00, 0.25, "Low Risk", "Düşük Risk", "#22c55e"),
    (0.25, 0.50, "Moderate Risk", "Orta Risk", "#eab308"),
    (0.50, 0.75, "High Risk", "Yüksek Risk", "#f97316"),
    (0.75, 1.00, "Critical Risk", "Kritik Risk", "#ef4444"),
)


@dataclass(frozen=True)
class OpenDataEndpoints:
    """Keyless, free public APIs used by the data loader."""

    open_meteo_forecast: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_elevation: str = "https://api.open-meteo.com/v1/elevation"
    usgs_earthquake_api: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    afad_earthquake_api: str = "https://deprem.afad.gov.tr/apiv2/event/filter"
    http_timeout_seconds: float = 6.0


ENDPOINTS = OpenDataEndpoints()

# Normalization bounds for physical measurements
MAX_FAULT_DISTANCE_KM: float = 150.0  # >= 150 km -> S_p proximity component ~ 0
MAX_SLOPE_DEG: float = 45.0           # >= 45 deg -> T_e ~ 1.0
MAX_PRECIP_MM_7D: float = 150.0       # >= 150 mm 7-day precip -> H_m ~ 1.0
