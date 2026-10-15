"""
SmartHub Usage Prediction & Smart Resource Allocation Module.

Utilizes Chronos-2 foundation time-series models to forecast next-hour device charging
demand, charging-port availability, and power-bank preparation for SmartHub kiosks.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.chronos_forecaster import ChronosForecaster, get_forecaster


# --------------------------------------------------
# CONFIGURATION & KIOSK HARDWARE PARAMETERS
# --------------------------------------------------

# SmartHub Kiosk Hardware Specifications
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
    """Derive transparent, capacity-aware SmartHub kiosk resource planning decisions.
    
    Aligns forecasts with SmartHub's actual operations:
      1. Next-hour charging-session demand vs physical port capacity.
      2. Kiosk charging-port availability and congestion risk.
      3. Power-bank dispenser preparation and staging.

    Args:
        predicted_demand: Forecasted demand value (interpreted based on context).
        predicted_sessions: Explicit forecasted next-hour charging sessions count.
        predicted_volume_kwh: Legacy parameter for proxy EV energy demand.
        predicted_occupancy: Legacy parameter for proxy pile occupancy.
        total_ports: Total physical device-charging ports on the kiosk (default 16).
        total_power_banks: Total power-bank bays housed in the kiosk (default 12).
        total_chargers: Legacy alias for total_ports.
        is_proxy_data: If True, indicates inputs derive from external proxy benchmark.

    Returns:
        Structured kiosk resource allocation plan with demand level, port availability,
        power bank preparation actions, and clear operational recommendations.
    """
    # Resolve port capacity (supporting legacy 'total_chargers' alias)
    effective_ports = total_chargers if total_chargers is not None else total_ports
    effective_ports = max(1, int(effective_ports))
    effective_power_banks = max(1, int(total_power_banks))

    # Resolve demand metric and detect proxy usage
    proxy_caveat = None
    if predicted_sessions is not None:
        sessions = float(predicted_sessions)
    elif predicted_demand is not None and not is_proxy_data:
        sessions = float(predicted_demand)
    elif predicted_volume_kwh is not None:
        # Transparent proxy benchmark handling
        is_proxy_data = True
        proxy_caveat = PROXY_DATA_LIMITATION_NOTE
        # Normalize relative to benchmark distribution (typical hub peak ~8000 kWh)
        # to approximate relative kiosk demand without pretending kWh == sessions
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

    # Ensure non-negative demand and compute port utilization
    sessions = max(0.0, sessions)
    utilization_pct = min(100.0, round((sessions / effective_ports) * 100.0, 1))
    available_ports = max(0, effective_ports - int(np.ceil(sessions)))

    # Capacity-aware decision logic
    if utilization_pct >= HIGH_PORT_UTILIZATION_THRESHOLD:
        demand_level = "HIGH"
        status_color = "red"

        # Staging power banks for overflow users who cannot access a physical port
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
        # Backward-compatible fields
        "total_chargers": effective_ports,
        "estimated_utilization_percent": utilization_pct,
        "recommended_reserve_capacity_piles": available_ports,
        "is_proxy_data": is_proxy_data,
        "proxy_data_limitations": proxy_caveat,
    }


# --------------------------------------------------
# USAGE PREDICTION PIPELINE (CHRONOS-2)
# --------------------------------------------------

def predict_usage(
    context_data: Optional[Union[pd.Series, pd.DataFrame, List[float]]] = None,
    target: str = DEFAULT_KIOSK_TARGET,
    total_ports: int = DEFAULT_KIOSK_CHARGING_PORTS,
    total_power_banks: int = DEFAULT_KIOSK_POWER_BANKS,
    forecaster: Optional[ChronosForecaster] = None,
) -> Dict[str, Any]:
    """Predict next-hour charging demand using Chronos-2 foundation model.
    
    Supports native SmartHub kiosk charging sessions as well as benchmark proxy series.

    Args:
        context_data: Historical demand series. If None, loads recent history from dataset.
        target: Target metric ('charge_sessions' for native kiosk sessions, 'volume_kwh' for proxy).
        total_ports: Total charging ports on the kiosk.
        total_power_banks: Total power bank bays on the kiosk.
        forecaster: Optional ChronosForecaster instance (for dependency injection / mocking).

    Returns:
        Structured forecast and kiosk resource allocation plan.
    """
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

    # Attach proxy metadata if applicable
    if is_proxy:
        forecast_result["is_proxy_benchmark"] = True
        forecast_result["proxy_data_limitations"] = PROXY_DATA_LIMITATION_NOTE

    # Compute kiosk-aligned resource allocation
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


# --------------------------------------------------
# DIRECT EXECUTION DEMO
# --------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("SMARTHUB - KIOSK USAGE PREDICTION & RESOURCE ALLOCATION")
    print(f"Model: {MODEL_NAME} (Chronos-2 Foundation Model)")
    print("Use Case: Smart Device Kiosk Charging Ports & Power-Bank Staging")
    print("=" * 65)

    try:
        # Run on native SmartHub kiosk session data
        output = predict_usage(target="charge_sessions")
        fc = output["forecast"]
        rp = output["resource_plan"]

        print("\n[NEXT-HOUR FORECAST]")
        print(f"Target:               {fc['target']} ({fc['units']})")
        print(f"Period:               {fc['forecast_period']}")
        print(f"Point Prediction:     {fc['prediction']} {fc['units']}")
        print(f"80% Uncertainty:      [{fc['prediction_q10']} - {fc['prediction_q90']}] {fc['units']}")
        print(f"Generated At:         {fc['generated_at']}")

        print("\n[SMARTHUB KIOSK RESOURCE PLAN]")
        print(f"Demand Classification: {rp['demand_level']} ({rp['status_color']})")
        print(f"Expected Sessions:     {rp['predicted_charging_sessions']} sessions")
        print(f"Port Utilization:      {rp['estimated_port_utilization_percent']}% ({rp['available_ports_expected']} of {rp['total_charging_ports']} ports vacant)")
        print(f"Power Bank Prep:       [{rp['power_bank_preparation']['status']}] - {rp['power_bank_preparation']['action']}")
        print("\nAdvisories:")
        for rec in rp["recommendations"]:
            print(f" - {rec}")
        print("=" * 65 + "\n")

    except Exception as err:
        print(f"Execution notice: {err}")