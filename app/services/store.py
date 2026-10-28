"""
STORE service — business logic layer.

Workflow
--------
1. ``start_storage_session``  — user scans RFID → session created → lock stays locked
2. ``unlock_storage``         — user confirms ready → lock disengages
3. ``finish_storage_session`` — user done → lock re-engages → session ends

Safety guarantees
-----------------
- A hardware failure on unlock immediately marks the session 'failed'.
- A hardware failure on re-lock marks the session 'failed' (lock may be
  open — a physical inspection is needed).
- A user may not have two simultaneous active STORE sessions.
- Cancellation is supported via ``cancel_storage_session``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.database import db
from app.hardware.lock import LockController

logger = logging.getLogger(__name__)


AUTO_LOCK_SECONDS = 25


class StoreServiceError(Exception):
    """Raised when a storage operation cannot be completed."""


def start_storage_session(
    rfid_uid: str,
    allow_auto_register: bool = True,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Start a storage session for the user identified by *rfid_uid*.

    The locker stays locked at this stage — call ``unlock_storage`` when
    the user is ready to place an item.

    Raises ``StoreServiceError`` if the user already has an active STORE
    session.

    Returns
    -------
    dict with keys: session_id, user_id, status
    """
    if not rfid_uid or not rfid_uid.strip():
        raise StoreServiceError("RFID UID is required")

    uid = rfid_uid.strip()

    # Resolve or register user
    user = db.get_user_by_rfid(uid, db_path)
    if user is None:
        if not allow_auto_register:
            raise StoreServiceError(f"User RFID {uid!r} is not registered")
        user_id = db.create_user(uid, db_path)
        if user_id is None:
            user = db.get_user_by_rfid(uid, db_path)
            user_id = user["id"]
        logger.info("Auto-registered new user id=%s rfid=%s", user_id, uid)
    else:
        user_id = user["id"]

    # Prevent duplicate active STORE sessions for the same user
    existing = db.get_active_sessions_for_user(user_id, "STORE", db_path)
    if existing:
        raise StoreServiceError(
            f"User already has an active STORE session (id={existing[0]['id']}). "
            "Finish or cancel the existing session first."
        )

    try:
        session_id = db.create_session(user_id, "STORE", db_path)
    except ValueError as exc:
        raise StoreServiceError(str(exc)) from exc

    db.log_system_event(
        "STORE_START",
        "STORE",
        f"Session {session_id} started for user {user_id}",
        db_path,
    )
    logger.info("Storage session started: session=%s user=%s", session_id, user_id)
    return {
        "session_id": session_id,
        "user_id": user_id,
        "status": "active",
    }


def unlock_storage(
    session_id: int,
    lock_controller: Optional[LockController] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Disengage the lock so the user can place or retrieve an item.

    The session must be active.  If the lock hardware reports failure the
    function raises ``StoreServiceError`` and the session is marked 'failed' —
    never silently returns success.

    Returns
    -------
    dict with keys: session_id, locked (bool)
    """
    session = db.get_session(session_id, db_path)
    if session is None:
        raise StoreServiceError(f"Session {session_id} does not exist")
    if session["status"] != "active":
        raise StoreServiceError(
            f"Session {session_id} is not active (status={session['status']!r})"
        )

    lock = lock_controller or LockController(simulation=True)
    ok = lock.unlock()

    if not ok:
        db.end_session(session_id, "failed", db_path)
        db.log_system_event(
            "STORE_UNLOCK_FAILURE",
            "STORE",
            f"Lock failed to open for session {session_id}",
            db_path,
        )
        raise StoreServiceError("Lock hardware failed to unlock")

    db.log_system_event(
        "STORE_UNLOCK",
        "STORE",
        f"Lock opened for session {session_id}",
        db_path,
    )
    logger.info("Storage unlocked: session=%s", session_id)
    return {"session_id": session_id, "locked": False}


def finish_storage_session(
    session_id: int,
    lock_controller: Optional[LockController] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Re-engage the lock and end the storage session.

    The lock is re-locked before the session is marked complete.  If the
    lock hardware fails to re-engage the session is marked 'failed'.

    Returns
    -------
    dict with keys: session_id, status, duration_seconds
    """
    session = db.get_session(session_id, db_path)
    if session is None:
        raise StoreServiceError(f"Session {session_id} does not exist")
    if session["status"] != "active":
        raise StoreServiceError(
            f"Session {session_id} is not active (status={session['status']!r})"
        )

    lock = lock_controller or LockController(simulation=True)
    ok = lock.lock()

    if not ok:
        final_status = "failed"
        db.log_system_event(
            "STORE_LOCK_FAILURE",
            "STORE",
            f"Lock failed to close for session {session_id}",
            db_path,
        )
        logger.error("Lock failed to re-engage for session %d", session_id)
    else:
        final_status = "completed"

    db.end_session(session_id, final_status, db_path)
    updated = db.get_session(session_id, db_path)
    duration = updated["duration_seconds"] if updated else None

    db.log_system_event(
        "STORE_FINISH",
        "STORE",
        f"Session {session_id} finished with status={final_status}",
        db_path,
    )
    logger.info(
        "Storage session finished: session=%s status=%s duration=%ss",
        session_id,
        final_status,
        duration,
    )
    return {
        "session_id": session_id,
        "status": final_status,
        "duration_seconds": duration,
    }


def cancel_storage_session(
    session_id: int,
    lock_controller: Optional[LockController] = None,
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Cancel an active storage session, ensuring the lock is re-engaged.

    Unlike ``finish_storage_session``, this records the session as
    'cancelled' rather than 'completed'.  Useful for admin intervention
    and timeout recovery.

    Returns
    -------
    dict with keys: session_id, status, duration_seconds
    """
    session = db.get_session(session_id, db_path)
    if session is None:
        raise StoreServiceError(f"Session {session_id} does not exist")
    if session["status"] != "active":
        raise StoreServiceError(
            f"Session {session_id} is not active (status={session['status']!r})"
        )

    # Best-effort lock attempt; even on failure we still cancel the session
    lock = lock_controller or LockController(simulation=True)
    lock_ok = lock.lock()

    final_status = "cancelled" if lock_ok else "failed"
    if not lock_ok:
        db.log_system_event(
            "STORE_CANCEL_LOCK_FAILURE",
            "STORE",
            f"Lock failed to close during cancel of session {session_id}",
            db_path,
        )
        logger.error("Lock failed to re-engage during cancel of session %d", session_id)

    db.end_session(session_id, final_status, db_path)
    updated = db.get_session(session_id, db_path)
    duration = updated["duration_seconds"] if updated else None

    db.log_system_event(
        "STORE_CANCEL",
        "STORE",
        f"Session {session_id} cancelled (lock_ok={lock_ok})",
        db_path,
    )
    logger.info("Storage session cancelled: session=%s", session_id)
    return {
        "session_id": session_id,
        "status": final_status,
        "duration_seconds": duration,
    }


def get_store_sessions(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    """Return recent STORE sessions."""
    rows = db.get_sessions_by_service("STORE", limit=limit, db_path=db_path)
    return [dict(row) for row in rows]
