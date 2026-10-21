"""
ML integration: Usage Prediction & Smart Resource Allocation.

Utilizes Chronos-2 foundation time-series models to forecast next-hour device charging
demand, charging-port availability, and power-bank preparation for SmartHub kiosks,
with safe fallback to historical baselines when sufficient data is available.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)

# --------------------------------------------------
# CONFIGURATION & KIOSK HARDWARE PARAMETERS
# --------------------------------------------------

DEFAULT_KIOSK_CHARGING_PORTS = 16      # Physical device charging ports on kiosk
DEFAULT_KIOSK_POWER_BANKS = 12         # Internal power-bank storage / dispenser capacity

# Kiosk Capacity Thresholds (% of port capacity utilized)
HIGH_PORT_UTILIZATION_THRESHOLD = 85.0    # >= 85% port demand -> High demand
MODERATE_PORT_UTILIZATION_THRESHOLD = 50.0 # 50% - 84% port demand -> Moderate demand

# Data Paths
KIOSK_DATA_PATH = BASE_DIR / "data" / "dataset" / "charge_usage.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "hub_595_hourly.csv"

# Foundation Model ID
MODEL_NAME = "autogluon/chronos-2-small"

# Target Definitions
DEFAULT_KIOSK_TARGET = "charge_sessions"
DEFAULT_KIOSK_UNITS = "sessions"
PROXY_TARGET = "volume_kwh"
PROXY_UNITS = "kWh"

# Explicit documentation of proxy-data limitations
PROXY_DATA_LIMITATION_NOTE = (
    "UrbanEV data reflects automotive EV charging plaza demand (kWh and pile occupancy). "
    "It serves strictly as an external high-resolution transfer benchmark for evaluating "
    "Chronos-2 foundation forecasting capabilities. For SmartHub kiosk operations, "
    "device-charging session counts from charge_usage.csv represent actual local user demand. "
    "Automotive kWh and pile counts must not be conflated with kiosk phone charging sessions."
)


# --------------------------------------------------
# SMART RESOURCE ALLOCATION PLANNING
# --------------------------------------------------

def compute_resource_allocation(
    predicted_demand: Optional[float] = None,
    predicted_sessions: Optional[float] = None,
    predicted_volume_kwh: Optional[float] = None,
    predicted_occupancy: Optional[float] = None,
    total_ports: int = DEFAULT_KIOSK_CHARGING_PORTS,
    total_power_banks: int = DEFAULT_KIOSK_POWER_BANKS,
    total_chargers: Optional[int] = None,
    is_proxy_data: bool = False,
) -> Dict[str, Any]:
    """Derive transparent, capacity-aware SmartHub kiosk resource planning decisions."""
    effective_ports = total_chargers if total_chargers is not None else total_ports
    effective_ports = max(1, int(effective_ports))
    effective_power_banks = max(1, int(total_power_banks))

    proxy_caveat = None
    if predicted_sessions is not None:
        sessions = float(predicted_sessions)
    elif predicted_demand is not None and not is_proxy_data:
        sessions = float(predicted_demand)
    elif predicted_volume_kwh is not None:
        is_proxy_data = True
        proxy_caveat = PROXY_DATA_LIMITATION_NOTE
        normalized_load = min(1.0, max(0.0, float(predicted_volume_kwh) / 8000.0))
        sessions = round(normalized_load * effective_ports, 1)
    elif predicted_occupancy is not None:
        is_proxy_data = True
        proxy_caveat = PROXY_DATA_LIMITATION_NOTE
        normalized_load = min(1.0, max(0.0, float(predicted_occupancy) / 274.0))
        sessions = round(normalized_load * effective_ports, 1)
    elif predicted_demand is not None:
        sessions = float(predicted_demand)
    else:
        sessions = 0.0

    sessions = max(0.0, sessions)
    utilization_pct = min(100.0, round((sessions / effective_ports) * 100.0, 1))
    available_ports = max(0, effective_ports - int(np.ceil(sessions)))

    if utilization_pct >= HIGH_PORT_UTILIZATION_THRESHOLD:
        demand_level = "HIGH"
        status_color = "red"

        overflow_users = max(0, int(np.ceil(sessions - (effective_ports * 0.75))))
        staged_power_banks = min(effective_power_banks, max(4, overflow_users))

        power_bank_prep = {
            "status": "DISPENSE_READY",
            "recommended_staged_units": staged_power_banks,
            "total_power_banks_capacity": effective_power_banks,
            "action": (
                f"Stage {staged_power_banks} fully charged power banks in the kiosk dispenser "
                "to serve overflow users when charging ports reach capacity."
            ),
        }
        recommendations = [
            f"High charging-port demand expected ({sessions:.1f} sessions vs {effective_ports} ports, {utilization_pct}% utilization).",
            f"Stage {staged_power_banks} power banks in rapid-dispenser as walk-away alternatives to reduce kiosk port congestion.",
            "Display 'Busy Window' alert on kiosk LCD to encourage standard 30-minute charge turnaround.",
            f"Available physical ports projected at {available_ports} units.",
        ]

    elif utilization_pct >= MODERATE_PORT_UTILIZATION_THRESHOLD:
        demand_level = "MODERATE"
        status_color = "amber"

        staged_power_banks = max(2, int(effective_power_banks * 0.3))
        power_bank_prep = {
            "status": "STANDARD_ROTATION",
            "recommended_staged_units": staged_power_banks,
            "total_power_banks_capacity": effective_power_banks,
            "action": "Maintain normal power-bank rotation in charging bays with 2-4 units staged for regular borrowing.",
        }
        recommendations = [
            f"Moderate charging demand expected ({sessions:.1f} sessions, {utilization_pct}% port utilization).",
            f"Charging port availability is adequate ({available_ports} ports projected vacant).",
            "Maintain standard peripheral rotation and verify returned items via vision kiosk sensor.",
        ]

    else:
        demand_level = "LOW"
        status_color = "green"

        power_bank_prep = {
            "status": "CYCLE_RECHARGE",
            "recommended_staged_units": 0,
            "total_power_banks_capacity": effective_power_banks,
            "action": "Optimal window to recharge all returned, depleted power banks inside internal dock bays.",
        }
        recommendations = [
            f"Low charging demand expected ({sessions:.1f} sessions). Ample vacant ports available ({available_ports} vacant).",
            "Initiate recharge cycles for depleted returned power banks inside kiosk charging bays.",
            "Run automated self-diagnostics on port cable continuity and electronic bay locks.",
        ]

    return {
        "demand_level": demand_level,
        "status_color": status_color,
        "predicted_charging_sessions": round(sessions, 1),
        "total_charging_ports": effective_ports,
        "available_ports_expected": available_ports,
        "estimated_port_utilization_percent": utilization_pct,
        "power_bank_preparation": power_bank_prep,
        "recommendations": recommendations,
        "total_chargers": effective_ports,
        "estimated_utilization_percent": utilization_pct,
        "recommended_reserve_capacity_piles": available_ports,
        "is_proxy_data": is_proxy_data,
        "proxy_data_limitations": proxy_caveat,
    }


def predict_usage(
    context_data: Optional[Union[pd.Series, pd.DataFrame, List[float]]] = None,
    target: str = DEFAULT_KIOSK_TARGET,
    total_ports: int = DEFAULT_KIOSK_CHARGING_PORTS,
    total_power_banks: int = DEFAULT_KIOSK_POWER_BANKS,
    forecaster: Optional[Any] = None,
) -> Dict[str, Any]:
    """Predict next-hour charging demand using Chronos-2 foundation model."""
    from app.ml.chronos_forecaster import ChronosForecaster, get_forecaster

    is_proxy = target in ["volume_kwh", "occupancy_piles"]

    if context_data is None:
        if is_proxy:
            if not PROCESSED_DATA_PATH.exists():
                raise FileNotFoundError(
                    f"Processed benchmark dataset not found at {PROCESSED_DATA_PATH}. "
                    "Run 'python app/ml/preprocess.py' to generate the benchmark data first."
                )
            df = pd.read_csv(PROCESSED_DATA_PATH)
            context_data = df[target].iloc[-168:]
        else:
            if not KIOSK_DATA_PATH.exists():
                raise FileNotFoundError(
                    f"SmartHub kiosk usage dataset not found at {KIOSK_DATA_PATH}."
                )
            df = pd.read_csv(KIOSK_DATA_PATH)
            col = target if target in df.columns else "charge_sessions"
            context_data = df[col]

    units = "sessions" if "session" in target.lower() else ("kWh" if "kwh" in target.lower() else "piles")
    model = forecaster or get_forecaster(model_name=MODEL_NAME)
    forecast_result = model.predict_next_hour(
        context=context_data,
        target=target,
        units=units,
        prediction_length=1,
    )

    if is_proxy:
        forecast_result["is_proxy_benchmark"] = True
        forecast_result["proxy_data_limitations"] = PROXY_DATA_LIMITATION_NOTE

    predicted_val = forecast_result["prediction"]
    plan = compute_resource_allocation(
        predicted_demand=predicted_val,
        predicted_sessions=predicted_val if not is_proxy else None,
        predicted_volume_kwh=predicted_val if target == "volume_kwh" else None,
        predicted_occupancy=predicted_val if target == "occupancy_piles" else None,
        total_ports=total_ports,
        total_power_banks=total_power_banks,
        is_proxy_data=is_proxy,
    )

    return {
        "forecast": forecast_result,
        "resource_plan": plan,
    }


def predict_next_hour_demand(
    historical_data: Optional[list[dict[str, Any]]] = None,
    db_path: Optional[str] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Predict the next-hour charging demand for the kiosk backend."""
    try:
        return _run_prediction(historical_data, db_path)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return _unavailable(str(exc))


def _run_prediction(
    historical_data: Optional[list[dict[str, Any]]],
    db_path: Optional[str],
) -> dict[str, Any]:
    from app.config import Config
    from app.ml.resource_allocation import recommend_resources

    data = historical_data
    if data is None:
        data = _load_historical_data(db_path)

    hourly_counts = _hourly_session_counts(data)
    if len(hourly_counts) < 2:
        return _unavailable("Insufficient hourly history for prediction (need at least 2 hours)")

    # Demand from chronological history (previous comparable hour baseline)
    demand = float(hourly_counts[-1]["count"])
    port_cap = getattr(Config, "CHARGING_PORT_CAPACITY", DEFAULT_KIOSK_CHARGING_PORTS)
    bank_cap = getattr(Config, "POWER_BANK_BAY_CAPACITY", DEFAULT_KIOSK_POWER_BANKS)

    res_plan = compute_resource_allocation(
        predicted_sessions=demand,
        total_ports=port_cap,
        total_power_banks=bank_cap,
    )
    rec_resources = recommend_resources(
        demand,
        charging_port_capacity=port_cap,
        power_bank_bay_capacity=bank_cap,
    )

    return {
        "available": True,
        "predicted_demand": demand,
        "forecast_horizon_hours": 1,
        "prediction_interval": None,
        "confidence": None,
        "model": "previous-hour baseline",
        "historical_hour_count": len(hourly_counts),
        "resource_allocation": rec_resources,
        "resource_plan": res_plan,
        "error": None,
    }


def _load_historical_data(db_path: Optional[str]) -> list[dict[str, Any]]:
    try:
        from pathlib import Path
        from app.database import db

        path = Path(db_path) if db_path else None
        sessions = db.execute_query(
            """
            SELECT s.started_at, s.duration_seconds
            FROM sessions AS s
            JOIN services AS sv ON s.service_id = sv.id
            WHERE sv.name = 'CHARGE'
              AND s.status = 'completed'
            ORDER BY s.id DESC
            LIMIT 200
            """,
            db_path=path,
        )
        return [dict(row) for row in sessions]
    except Exception as exc:
        logger.warning("Could not load historical data: %s", exc)
        return []


def _hourly_session_counts(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timestamps: list[datetime] = []
    for row in data:
        raw = row.get("started_at")
        if not isinstance(raw, str):
            continue
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        timestamps.append(value.replace(tzinfo=value.tzinfo or timezone.utc, minute=0, second=0, microsecond=0))
    if not timestamps:
        return []
    counts = Counter(timestamps)
    current, last = min(counts), max(counts)
    result = []
    while current <= last:
        result.append({"hour": current.isoformat(), "count": counts[current]})
        current += timedelta(hours=1)
    return result


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "predicted_demand": None,
        "confidence": None,
        "model": "none",
        "error": reason,
    }
