"""
ML integration: Vision-based item verification.

Provides a unified interface combining deep convolutional neural network
inference (MobileNetV2 / ResNet18 transfer learning) with safe fallback
behavior for borrow/return services and REST API endpoints.
"""

from __future__ import annotations

import io
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# Ensure workspace root is always discoverable for imports
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.device_classifier import DeviceClassifier
from app.ml.device_dataset import CLASS_NAMES, HUMAN_LABELS, get_transforms

logger = logging.getLogger(__name__)

# Allowed image types for upload
ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB

MODEL_CHECKPOINT_PATH = BASE_DIR / "app" / "ml" / "device_model.pth"

# Friendly alias normalization for expected item queries
ITEM_ALIASES: Dict[str, str] = {
    "powerbank": "power_bank",
    "power_bank": "power_bank",
    "power-bank": "power_bank",
    "battery": "power_bank",
    "cable": "charging_cable",
    "charging_cable": "charging_cable",
    "charging-cable": "charging_cable",
    "charge_cable": "charging_cable",
    "charger_cable": "charging_cable",
    "charger": "charger",
    "adapter": "charger",
    "charger_adapter": "charger",
    "charger-adapter": "charger",
    "power_adapter": "charger",
    "plug": "charger",
}


class DeviceVerifier:
    """Singleton model cache and inference manager for peripheral device vision."""

    _instance: Optional[DeviceVerifier] = None
    _model: Optional[DeviceClassifier] = None
    _device: torch.device
    _metadata: Dict[str, Any]

    def __init__(self, checkpoint_path: Optional[str | Path] = None) -> None:
        self.checkpoint_path = Path(checkpoint_path or MODEL_CHECKPOINT_PATH)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = get_transforms()["test"]
        self._metadata = {}
        self._load_model()

    def _load_model(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Device classification model checkpoint not found at: {self.checkpoint_path}. "
                "Please run `python app/ml/train_vision.py` first."
            )
        self._model, self._metadata = DeviceClassifier.load_from_checkpoint(
            self.checkpoint_path,
            device=self._device,
        )

    def _prepare_image(self, image_input: Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]) -> torch.Tensor:
        """Standardize various image input formats into a normalized PyTorch tensor."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            image = Image.open(path).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input.astype("uint8")).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self._device)

    def predict(self, image_input: Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform classification inference on an input image."""
        if self._model is None:
            self._load_model()
        assert self._model is not None, "Model failed to load"

        start_time = time.time()
        tensor = self._prepare_image(image_input)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        latency_ms = (time.time() - start_time) * 1000.0

        probabilities_dict = {
            CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(CLASS_NAMES))
        }

        return {
            "status": "success",
            "predicted_class": pred_class,
            "class_label": HUMAN_LABELS.get(pred_class, pred_class),
            "confidence": round(confidence, 4),
            "probabilities": probabilities_dict,
            "inference_latency_ms": round(latency_ms, 2),
            "model_architecture": self._model.backbone_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def verify(
        self,
        image_input: Union[str, Path, bytes, bytearray, Image.Image, np.ndarray],
        expected_item: Optional[str] = None,
        confidence_threshold: float = 0.60,
    ) -> Dict[str, Any]:
        """Verify if an image contains a valid device or matches an expected device item."""
        result = self.predict(image_input)

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        if expected_item is not None:
            norm_expected = ITEM_ALIASES.get(expected_item.lower().strip().replace(" ", "_"), expected_item.lower().strip())
            is_match = (predicted_class == norm_expected)
            is_verified = is_match and (confidence >= confidence_threshold)

            expected_label = HUMAN_LABELS.get(norm_expected, expected_item)
            detected_label = result["class_label"]

            if is_verified:
                message = f"Item verified successfully as {detected_label} (Confidence: {confidence*100:.1f}%)"
            elif is_match and not is_verified:
                message = (
                    f"Item identified as {detected_label}, but confidence ({confidence*100:.1f}%) "
                    f"was below required threshold ({confidence_threshold*100:.1f}%)"
                )
            else:
                message = f"Verification mismatch: Expected {expected_label}, but detected {detected_label}"

            return {
                "status": "success" if is_match else "mismatch",
                "available": True,
                "verified": is_verified,
                "match": is_match,
                "expected_item": norm_expected,
                "expected": norm_expected,
                "expected_label": expected_label,
                "category": predicted_class,
                "detected_item": predicted_class,
                "detected_label": detected_label,
                "confidence": confidence,
                "confidence_threshold": confidence_threshold,
                "probabilities": result["probabilities"],
                "message": message,
                "inference_latency_ms": result["inference_latency_ms"],
                "error": None,
            }
        else:
            is_verified = confidence >= confidence_threshold
            return {
                "status": "success",
                "available": True,
                "verified": is_verified,
                "match": None,
                "expected": None,
                "expected_item": None,
                "category": predicted_class,
                "detected_item": predicted_class,
                "detected_label": result["class_label"],
                "confidence": confidence,
                "confidence_threshold": confidence_threshold,
                "probabilities": result["probabilities"],
                "message": f"Detected device: {result['class_label']} (Confidence: {confidence*100:.1f}%)",
                "inference_latency_ms": result["inference_latency_ms"],
                "error": None,
            }


_verifier_instance: Optional[DeviceVerifier] = None


def get_verifier() -> DeviceVerifier:
    """Retrieve or initialize the singleton DeviceVerifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = DeviceVerifier()
    return _verifier_instance


def classify_device(image: Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Classify device in image into charger, charging_cable, or power_bank."""
    verifier = get_verifier()
    return verifier.predict(image)


def verify_item(
    image_data: Union[str, Path, bytes, bytearray, Image.Image, np.ndarray],
    expected_category: Optional[str] = None,
    mime_type: str = "image/jpeg",
    threshold: float = 0.60,
    expected_item: Optional[str] = None,
) -> dict[str, Any]:
    """Unified verification entry point supporting backend, service and standalone callers."""
    target_category = expected_item if expected_item is not None else expected_category

    # File path validation: if a Path or str path is given and does not exist, raise FileNotFoundError directly
    if isinstance(image_data, (Path, str)):
        path = Path(image_data)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")

    # Byte input validation
    if isinstance(image_data, (bytes, bytearray)):
        if not image_data:
            return _error("No image data provided", target_category)
        if len(image_data) > MAX_IMAGE_BYTES:
            return _error(
                f"Image exceeds maximum size of {MAX_IMAGE_BYTES // 1024} KB",
                target_category,
            )
        if mime_type not in ALLOWED_MIME_TYPES:
            return _error(
                f"Unsupported image type {mime_type!r}. Allowed: {sorted(ALLOWED_MIME_TYPES)}",
                target_category,
            )

    try:
        verifier = get_verifier()
        return verifier.verify(
            image_input=image_data,
            expected_item=target_category,
            confidence_threshold=threshold,
        )
    except FileNotFoundError as exc:
        if isinstance(image_data, (Path, str)) and "Image file does not exist" in str(exc):
            raise
        logger.warning("Model checkpoint missing or unavailable: %s", exc)
        return _unavailable(target_category, str(exc))
    except Exception as exc:
        logger.exception("Vision verification failed: %s", exc)
        return _error(str(exc), target_category)


def _unavailable(expected_category: Optional[str], reason: str = "Vision model not configured") -> dict[str, Any]:
    return {
        "status": "unavailable",
        "available": False,
        "match": None,
        "verified": False,
        "category": None,
        "detected_item": None,
        "confidence": None,
        "expected": expected_category,
        "expected_item": expected_category,
        "error": reason,
    }


def _error(reason: str, expected_category: Optional[str]) -> dict[str, Any]:
    return {
        "status": "error",
        "available": False,
        "match": None,
        "verified": False,
        "category": None,
        "detected_item": None,
        "confidence": None,
        "expected": expected_category,
        "expected_item": expected_category,
        "error": reason,
    }
