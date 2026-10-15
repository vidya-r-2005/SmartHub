"""
ML integration: Usage Prediction.

This module defines the stable interface that the Flask API calls.
The actual model (Chronos-2 or similar) is a separate implementation.

If the model is not configured or its dependencies are unavailable,
all public functions return a structured "unavailable" response rather
than raising unhandled exceptions.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

# The prediction result structure returned by predict_next_hour_demand().
# External callers should always check the 'available' key before using
# 'predicted_demand'.
#
# {
#   "available": bool,
#   "predicted_demand": float | None,
#   "confidence": float | None,   # 0.0–1.0, optional
#   "model": str,                  # model name / version
#   "error": str | None,           # set only when available=False
# }


def predict_next_hour_demand(
    historical_data: Optional[list[dict[str, Any]]] = None,
    db_path: Optional[str] = None,
) -> dict[str, Any]:
    """Predict the next-hour charging demand.

    Parameters
    ----------
    historical_data : list[dict] | None
        Optional pre-loaded historical records.  If *None*, the function
        will attempt to load data from the database at *db_path* or the
        default database path.
    db_path : str | None
        Path to the SQLite database.  Ignored when *historical_data* is
        supplied.

    Returns
    -------
    dict
        Prediction result with an 'available' boolean.  When False, the
        'error' key explains why.
    """
    try:
        return _run_prediction(historical_data, db_path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        return _unavailable(str(exc))


def _run_prediction(
    historical_data: Optional[list[dict[str, Any]]],
    db_path: Optional[str],
) -> dict[str, Any]:
    """Attempt to run the prediction model."""
    # Try to import the ML model implementation
    try:
        from app.ml import _prediction_model as model_impl  # type: ignore[import]
    except ImportError:
        # Model implementation not yet available — expected during development
        logger.info("Prediction model implementation not found; returning unavailable")
        return _unavailable("Prediction model not yet configured")

    # Validate / load input data
    data = historical_data
    if data is None:
        data = _load_historical_data(db_path)

    if not data:
        return _unavailable("Insufficient historical data for prediction")

    result = model_impl.predict(data)
    return {
        "available": True,
        "predicted_demand": float(result.get("demand", 0)),
        "confidence": result.get("confidence"),
        "model": result.get("model", "unknown"),
        "error": None,
    }


def _load_historical_data(db_path: Optional[str]) -> list[dict[str, Any]]:
    """Load recent charge session data from the database."""
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
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load historical data: %s", exc)
        return []


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "predicted_demand": None,
        "confidence": None,
        "model": "none",
        "error": reason,
    }

