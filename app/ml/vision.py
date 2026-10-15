"""
ML integration: Vision-based item verification.

This module defines the stable interface that the BORROW service calls.
The actual CNN model is a separate implementation that may not yet be
configured.

If the model is unavailable the functions return a structured
"unavailable" response — they never silently approve a borrow or return.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Allowed image types for upload
ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB

# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

# verify_item() returns a dict with this structure:
# {
#   "available": bool,        # False when model is not configured
#   "match": bool | None,     # True when visual category matches expected
#   "category": str | None,   # Predicted category from the model
#   "confidence": float|None, # 0.0–1.0
#   "expected": str | None,   # The expected item category
#   "error": str | None,
# }


def verify_item(
    image_data: bytes,
    expected_category: str,
    mime_type: str = "image/jpeg",
) -> dict[str, Any]:
    """Verify that image data shows an object matching *expected_category*.

    Parameters
    ----------
    image_data : bytes
        Raw image bytes (JPEG, PNG, or WebP).
    expected_category : str
        The item_type value from the items table (e.g. "power_bank").
    mime_type : str
        MIME type of *image_data*.

    Returns
    -------
    dict
        Result dict.  Always check 'available' before trusting 'match'.
        When 'available' is False the operation must NOT be authorised.
    """
    # --- Input validation ---------------------------------------------------
    if not image_data:
        return _error("No image data provided", expected_category)

    if len(image_data) > MAX_IMAGE_BYTES:
        return _error(
            f"Image exceeds maximum size of {MAX_IMAGE_BYTES // 1024} KB",
            expected_category,
        )

    if mime_type not in ALLOWED_MIME_TYPES:
        return _error(
            f"Unsupported image type {mime_type!r}. "
            f"Allowed: {sorted(ALLOWED_MIME_TYPES)}",
            expected_category,
        )

    if not expected_category or not expected_category.strip():
        return _error("expected_category cannot be empty", expected_category)

    # --- Model invocation ---------------------------------------------------
    try:
        return _run_vision(image_data, expected_category.strip())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision verification failed: %s", exc)
        return _error(str(exc), expected_category)


def _run_vision(image_data: bytes, expected_category: str) -> dict[str, Any]:
    """Attempt to classify the image using the CNN model."""
    try:
        from app.ml import _vision_model as model_impl  # type: ignore[import]
    except ImportError:
        logger.info("Vision model implementation not found; returning unavailable")
        return _unavailable(expected_category)

    result = model_impl.classify(image_data)
    predicted = result.get("category", "")
    confidence = result.get("confidence")
    matched = predicted.lower() == expected_category.lower()

    if not matched:
        logger.warning(
            "Vision mismatch: expected=%s predicted=%s confidence=%s",
            expected_category,
            predicted,
            confidence,
        )

    return {
        "available": True,
        "match": matched,
        "category": predicted,
        "confidence": confidence,
        "expected": expected_category,
        "error": None,
    }


def _unavailable(expected_category: str) -> dict[str, Any]:
    return {
        "available": False,
        "match": None,
        "category": None,
        "confidence": None,
        "expected": expected_category,
        "error": "Vision model not configured",
    }


def _error(reason: str, expected_category: str) -> dict[str, Any]:
    return {
        "available": False,
        "match": None,
        "category": None,
        "confidence": None,
        "expected": expected_category,
        "error": reason,
    }

