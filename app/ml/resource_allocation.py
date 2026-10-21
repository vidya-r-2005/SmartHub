"""Kiosk-specific resource recommendations for a demand forecast.

Recommendations do not command hardware; simulation must never claim that it
has physically charged, dispensed, or inspected an item.
"""

from __future__ import annotations

from typing import Any


def recommend_resources(
    predicted_demand: float,
    charging_port_capacity: int = 16,
    power_bank_bay_capacity: int = 12,
    available_power_banks: int | None = None,
) -> dict[str, Any]:
    """Convert next-hour session demand into capacity-bounded advice."""
    if charging_port_capacity < 1 or power_bank_bay_capacity < 0:
        raise ValueError("Resource capacities must be non-negative (ports > 0)")
    demand = max(0.0, float(predicted_demand))
    occupied = min(charging_port_capacity, int(round(demand)))
    utilization = min(1.0, demand / charging_port_capacity)
    if utilization >= 0.85:
        mode, action = "HIGH", "DISPENSE_READY"
    elif utilization >= 0.50:
        mode, action = "MODERATE", "STANDARD_ROTATION"
    else:
        mode, action = "LOW", "CYCLE_RECHARGE"
    ready_banks = min(power_bank_bay_capacity, max(0, int(round(demand))))
    shortfall = max(0, ready_banks - available_power_banks) if available_power_banks is not None else None
    return {
        "mode": mode,
        "recommended_action": action,
        "charging_port_capacity": charging_port_capacity,
        "expected_occupied_ports": occupied,
        "expected_vacant_ports": charging_port_capacity - occupied,
        "estimated_port_utilization": round(utilization, 3),
        "power_bank_bay_capacity": power_bank_bay_capacity,
        "recommended_ready_power_banks": ready_banks,
        "available_power_banks": available_power_banks,
        "power_bank_shortfall": shortfall,
        "simulated_recommendation": True,
    }
