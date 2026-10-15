"""
CHARGE service — business logic layer.

Orchestrates charging sessions, payment collection, hardware control,
and database persistence.  All hardware access goes through injectable
controller objects so tests can pass mocks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.database import db
from app.hardware.charging import ChargingController
from app.hardware.coin import CoinAcceptor

logger = logging.getLogger(__name__)

# Minimum payment required to start a charging session (currency units)
MINIMUM_CHARGE_AMOUNT = 0.0  # Allow free/zero-payment sessions (campus kiosk)
MAXIMUM_CHARGE_AMOUNT = 10_000.0  # Sanity cap


class ChargeServiceError(Exception):
    """Raised when a charging operation cannot be completed."""


def start_charging_session(
    rfid_uid: str,
    amount: float,
    payment_method: str = "coin",
    charging_controller: Optional[ChargingController] = None,
    coin_acceptor: Optional[CoinAcceptor] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Start a charging session for the user identified by *rfid_uid*.

    Parameters
    ----------
    rfid_uid : str
        RFID UID of the user.  The user is auto-registered if not found.
    amount : float
        Payment amount to record.
    payment_method : str
        Payment method string (e.g. "coin", "mock").
    charging_controller : ChargingController | None
        Hardware abstraction.  A simulation controller is used if None.
    coin_acceptor : CoinAcceptor | None
        Coin acceptor abstraction (not used for mock payments).
    db_path : Path | None
        Override database path (tests use a temp path).

    Returns
    -------
    dict with keys: session_id, transaction_id, user_id, status
    """
    # --- Validate inputs ----------------------------------------------------
    if not rfid_uid or not rfid_uid.strip():
        raise ChargeServiceError("RFID UID is required")

    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ChargeServiceError("amount must be a number")

    if amount < MINIMUM_CHARGE_AMOUNT:
        raise ChargeServiceError(
            f"amount must be >= {MINIMUM_CHARGE_AMOUNT}"
        )

    if amount > MAXIMUM_CHARGE_AMOUNT:
        raise ChargeServiceError(
            f"amount must be <= {MAXIMUM_CHARGE_AMOUNT}"
        )

    if not payment_method or not payment_method.strip():
        raise ChargeServiceError("payment_method cannot be empty")

    uid = rfid_uid.strip()

    # --- Ensure user exists -------------------------------------------------
    user = db.get_user_by_rfid(uid, db_path)
    if user is None:
        user_id = db.create_user(uid, db_path)
        if user_id is None:
            # Race: another call registered it between our check and insert
            user = db.get_user_by_rfid(uid, db_path)
            user_id = user["id"]
        logger.info("Auto-registered new user id=%s rfid=%s", user_id, uid)
    else:
        user_id = user["id"]

    # --- Create session ------------------------------------------------------
    try:
        session_id = db.create_session(user_id, "CHARGE", db_path)
    except ValueError as exc:
        raise ChargeServiceError(str(exc)) from exc

    # --- Enable hardware charging port ---------------------------------------
    ctrl = charging_controller or ChargingController(simulation=True)
    hw_ok = ctrl.enable()

    if not hw_ok:
        # Hardware failed — cancel session and do not record payment
        db.end_session(session_id, "failed", db_path)
        db.log_system_event(
            "CHARGE_HW_FAILURE",
            "CHARGE",
            f"Failed to enable charging port for session {session_id}",
            db_path,
        )
        raise ChargeServiceError("Charging hardware failed to enable")

    # --- Record transaction --------------------------------------------------
    try:
        transaction_id = db.record_transaction(
            amount=amount,
            session_id=session_id,
            payment_method=payment_method.strip(),
            status="completed",
            db_path=db_path,
        )
    except (ValueError, Exception) as exc:
        # Attempt to cancel session cleanly; hardware already enabled
        db.end_session(session_id, "failed", db_path)
        ctrl.disable()
        raise ChargeServiceError(f"Failed to record transaction: {exc}") from exc

    db.log_system_event(
        "CHARGE_START",
        "CHARGE",
        f"Session {session_id} started for user {user_id}, amount={amount}",
        db_path,
    )
    logger.info(
        "Charging started: session=%s user=%s amount=%.2f", session_id, user_id, amount
    )
    return {
        "session_id": session_id,
        "transaction_id": transaction_id,
        "user_id": user_id,
        "status": "active",
    }


def stop_charging_session(
    session_id: int,
    charging_controller: Optional[ChargingController] = None,
    final_status: str = "completed",
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Stop an active charging session.

    Parameters
    ----------
    session_id : int
        ID of the session to stop.
    charging_controller : ChargingController | None
        Hardware abstraction.
    final_status : str
        One of 'completed', 'cancelled', or 'failed'.
    db_path : Path | None
        Override database path.

    Returns
    -------
    dict with keys: session_id, status, duration_seconds
    """
    if final_status not in ("completed", "cancelled", "failed"):
        raise ChargeServiceError(
            f"Invalid final status {final_status!r}"
        )

    # Verify session exists and is active
    session = db.get_session(session_id, db_path)
    if session is None:
        raise ChargeServiceError(f"Session {session_id} does not exist")

    if session["status"] != "active":
        raise ChargeServiceError(
            f"Session {session_id} is not active (status={session['status']!r})"
        )

    # Disable hardware
    ctrl = charging_controller or ChargingController(simulation=True)
    hw_ok = ctrl.disable()
    if not hw_ok:
        logger.error("Failed to disable charging port for session %d", session_id)
        # Record as failed but still end the database session
        final_status = "failed"

    ended = db.end_session(session_id, final_status, db_path)
    if not ended:
        raise ChargeServiceError(
            f"Could not end session {session_id} — it may already be closed"
        )

    # Re-fetch to get duration
    updated = db.get_session(session_id, db_path)
    duration = updated["duration_seconds"] if updated else None

    db.log_system_event(
        "CHARGE_STOP",
        "CHARGE",
        f"Session {session_id} ended with status={final_status}",
        db_path,
    )
    logger.info(
        "Charging stopped: session=%s status=%s duration=%ss",
        session_id,
        final_status,
        duration,
    )
    return {
        "session_id": session_id,
        "status": final_status,
        "duration_seconds": duration,
    }


def get_charge_sessions(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    """Return recent CHARGE sessions."""
    rows = db.get_sessions_by_service("CHARGE", limit=limit, db_path=db_path)
    return [dict(row) for row in rows]


def get_charge_session(
    session_id: int, db_path: Optional[Path] = None
) -> Optional[dict]:
    """Return a single session dict or None."""
    row = db.get_session(session_id, db_path)
    if row is None:
        return None
    s = dict(row)
    if s.get("service_name") != "CHARGE":
        return None
    return s

