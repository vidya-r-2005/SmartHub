"""
BORROW service — business logic layer.

Workflow
--------
1. ``list_available_items``  — show what can be borrowed
2. ``start_borrow``          — validate RFID + item, create borrow record
3. ``return_item``           — validate RFID + item, close borrow record

Safety guarantees
-----------------
- Vision mismatch always raises ``VerificationMismatchError``; never
  silently authorises the operation.
- RFID-only flows are allowed, but a requested camera verification fails
  closed when the model is unavailable or cannot classify the image.
- A user may not borrow the same item twice simultaneously.
- Session closure on return is done in the same DB connection block,
  ensuring consistency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.database import db

logger = logging.getLogger(__name__)


class BorrowServiceError(Exception):
    """Raised when a borrowing operation cannot be completed."""


class VerificationMismatchError(BorrowServiceError):
    """Raised when the vision model detects a category mismatch."""


def list_available_items(db_path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Return all items currently available to borrow."""
    rows = db.get_available_items(db_path)
    return [dict(row) for row in rows]


def start_borrow(
    rfid_uid: str,
    item_rfid: str,
    image_data: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Begin a borrow session for *item_rfid* by the user with *rfid_uid*.

    Parameters
    ----------
    rfid_uid : str
        User RFID UID.
    item_rfid : str
        RFID UID of the item to borrow.
    image_data : bytes | None
        Optional image for vision verification.  When provided the vision
        module is called.  A mismatch raises ``VerificationMismatchError``.
        When the vision module is unavailable, the operation proceeds on
        RFID identity alone.
    mime_type : str
        MIME type of *image_data*.
    db_path : Path | None
        Override database path.

    Returns
    -------
    dict with keys: session_id, borrow_record_id, user_id, item_id,
                    item_type, vision_result
    """
    # --- Validate inputs --------------------------------------------------------
    if not rfid_uid or not rfid_uid.strip():
        raise BorrowServiceError("User RFID UID is required")
    if not item_rfid or not item_rfid.strip():
        raise BorrowServiceError("Item RFID UID is required")

    uid = rfid_uid.strip()
    iuid = item_rfid.strip()

    # --- Resolve or register user -----------------------------------------------
    user = db.get_user_by_rfid(uid, db_path)
    if user is None:
        user_id = db.create_user(uid, db_path)
        if user_id is None:
            user = db.get_user_by_rfid(uid, db_path)
            user_id = user["id"]
    else:
        user_id = user["id"]

    # --- Locate item by RFID ----------------------------------------------------
    item = db.get_item_by_rfid(iuid, db_path)
    if item is None:
        raise BorrowServiceError(f"No item with RFID {iuid!r}")
    if item["status"] != "available":
        raise BorrowServiceError(
            f"Item {iuid!r} is not available (status={item['status']!r})"
        )

    # --- Optional vision verification -------------------------------------------
    vision_result: Optional[dict] = None
    if image_data:
        vision_result = _verify_vision(image_data, item["item_type"], mime_type, db_path)
        if not vision_result.get("available"):
            raise BorrowServiceError(
                "Vision verification is unavailable; refusing camera-verified borrow"
            )
        if vision_result.get("available") and vision_result.get("match") is False:
            # Mismatch confirmed — log and refuse
            db.log_system_event(
                "BORROW_VISION_MISMATCH",
                "BORROW",
                (
                    f"Vision mismatch for item {item['id']}: "
                    f"expected={item['item_type']!r} "
                    f"predicted={vision_result.get('category')!r}"
                ),
                db_path,
            )
            raise VerificationMismatchError(
                f"Vision mismatch: expected {item['item_type']!r}, "
                f"got {vision_result.get('category')!r}"
            )

    # --- Create session and borrow record atomically ----------------------------
    try:
        session_id = db.create_session(user_id, "BORROW", db_path)
        borrow_id = db.record_borrow(session_id, item["id"], db_path)
    except ValueError as exc:
        raise BorrowServiceError(str(exc)) from exc

    db.log_system_event(
        "BORROW_START",
        "BORROW",
        f"Item {item['id']} borrowed by user {user_id} in session {session_id}",
        db_path,
    )
    logger.info(
        "Borrow started: session=%s user=%s item=%s", session_id, user_id, item["id"]
    )
    return {
        "session_id": session_id,
        "borrow_record_id": borrow_id,
        "user_id": user_id,
        "item_id": item["id"],
        "item_type": item["item_type"],
        "vision_result": vision_result,
    }


def return_item(
    item_rfid: str,
    image_data: Optional[bytes] = None,
    mime_type: str = "image/jpeg",
    db_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Process the return of an item identified by *item_rfid*.

    Parameters
    ----------
    item_rfid : str
        RFID UID of the item being returned.
    image_data : bytes | None
        Optional image for vision verification.
    mime_type : str
        MIME type of *image_data*.
    db_path : Path | None
        Override database path.

    Returns
    -------
    dict with keys: item_id, item_type, status, session_id, vision_result
    """
    if not item_rfid or not item_rfid.strip():
        raise BorrowServiceError("Item RFID UID is required")

    iuid = item_rfid.strip()

    # --- Locate item ------------------------------------------------------------
    item = db.get_item_by_rfid(iuid, db_path)
    if item is None:
        raise BorrowServiceError(f"No item with RFID {iuid!r}")
    if item["status"] != "borrowed":
        raise BorrowServiceError(
            f"Item {iuid!r} is not currently borrowed (status={item['status']!r})"
        )

    # --- Optional vision verification -------------------------------------------
    vision_result: Optional[dict] = None
    if image_data:
        vision_result = _verify_vision(image_data, item["item_type"], mime_type, db_path)
        if not vision_result.get("available"):
            raise BorrowServiceError(
                "Vision verification is unavailable; refusing camera-verified return"
            )
        if vision_result.get("available") and vision_result.get("match") is False:
            db.log_system_event(
                "RETURN_VISION_MISMATCH",
                "BORROW",
                (
                    f"Vision mismatch on return for item {item['id']}: "
                    f"expected={item['item_type']!r} "
                    f"predicted={vision_result.get('category')!r}"
                ),
                db_path,
            )
            raise VerificationMismatchError(
                f"Return vision mismatch: expected {item['item_type']!r}, "
                f"got {vision_result.get('category')!r}"
            )

    # --- Return inventory and close the associated session atomically ------------
    try:
        session_id = db.record_return_and_end_session(item["id"], db_path)
    except ValueError as exc:
        raise BorrowServiceError(str(exc)) from exc

    db.log_system_event(
        "BORROW_RETURN",
        "BORROW",
        f"Item {item['id']} returned",
        db_path,
    )
    logger.info("Return recorded: item=%s session=%s", item["id"], session_id)
    return {
        "item_id": item["id"],
        "item_type": item["item_type"],
        "status": "returned",
        "session_id": session_id,
        "vision_result": vision_result,
    }


def get_borrow_records(limit: int = 50, db_path: Optional[Path] = None) -> list[dict]:
    """Return recent borrow records."""
    rows = db.get_borrow_records(limit=limit, db_path=db_path)
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _verify_vision(
    image_data: bytes,
    expected_category: str,
    mime_type: str,
    db_path: Optional[Path],
) -> dict:
    """Call the vision module and return its result dict.

    Any exception from the vision module (import error, model not loaded,
    etc.) is caught and returned as an ``available=False`` result.
    The caller must treat ``available=False`` as 'model not available'
    and proceed based on policy (currently: proceed on RFID alone).
    """
    try:
        from app.ml.vision import verify_item

        return verify_item(image_data, expected_category, mime_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vision module raised an exception: %s", exc)
        return {
            "available": False,
            "match": None,
            "category": None,
            "confidence": None,
            "expected": expected_category,
            "error": str(exc),
        }
