"""
Hardware abstraction: Camera module for item verification.

Captures images on Raspberry Pi using Picamera2 or OpenCV and returns raw
image bytes or PIL Image objects suitable for direct ingestion by app/ml/vision.py.
In simulation mode, delivers a configurable mock image or sample asset.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


class Camera:
    """Controls the Raspberry Pi camera module for capturing item snapshots.

    Parameters
    ----------
    simulation : bool
        When True, returns a mock image from disk or generated in memory.
    mock_image_path : str | Path | None
        Optional path to an image file to serve during simulation mode.
    resolution : tuple[int, int]
        Image resolution (width, height). Default is (640, 480).
    """

    def __init__(
        self,
        simulation: bool = True,
        mock_image_path: Optional[Union[str, Path]] = None,
        resolution: tuple[int, int] = (640, 480),
    ):
        self._simulation = simulation
        self._mock_path = Path(mock_image_path) if mock_image_path else None
        self._resolution = resolution
        self._cam = None
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(self) -> bytes:
        """Capture an image and return JPEG bytes.

        Raises RuntimeError if the camera is closed or capture fails.
        """
        if self._closed:
            raise RuntimeError("Camera is closed")

        if self._simulation:
            return self._capture_simulation()

        return self._capture_hardware()

    def capture_to_file(self, destination: Union[str, Path]) -> Path:
        """Capture an image and save it directly to the specified destination path."""
        data = self.capture()
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def close(self) -> None:
        """Release camera hardware resources."""
        self._closed = True
        if not self._simulation and self._cam is not None:
            try:
                if hasattr(self._cam, "stop"):
                    self._cam.stop()
                if hasattr(self._cam, "close"):
                    self._cam.close()
                elif hasattr(self._cam, "release"):
                    self._cam.release()
                logger.info("Camera released successfully")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing camera: %s", exc)
            finally:
                self._cam = None

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        """Initialize Picamera2 (standard on Bullseye/Bookworm) or fallback to OpenCV."""
        try:
            try:
                from picamera2 import Picamera2  # type: ignore[import]

                picam = Picamera2()
                config = picam.create_still_configuration(
                    main={"size": self._resolution, "format": "RGB888"}
                )
                picam.configure(config)
                picam.start()
                self._cam = picam
                logger.info("Picamera2 initialised successfully")
                return
            except (ImportError, Exception) as pi_err:
                logger.debug("Picamera2 init failed (%s), trying cv2 VideoCapture", pi_err)

            import cv2  # type: ignore[import]

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("No USB or CSI camera accessible via OpenCV index 0")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
            self._cam = cap
            logger.info("OpenCV VideoCapture(0) initialised successfully")
        except ImportError as exc:
            raise RuntimeError(
                "Neither picamera2 nor opencv-python is installed. "
                "Install picamera2/opencv or set simulation=True."
            ) from exc

    def _capture_hardware(self) -> bytes:
        try:
            if hasattr(self._cam, "capture_array"):
                # Picamera2
                import cv2  # type: ignore[import]

                frame = self._cam.capture_array()
                success, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if not success:
                    raise RuntimeError("Failed to encode Picamera2 frame to JPEG")
                return encoded.tobytes()

            # OpenCV VideoCapture
            import cv2  # type: ignore[import]

            ret, frame = self._cam.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read frame from camera")
            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                raise RuntimeError("Failed to encode camera frame to JPEG")
            return encoded.tobytes()
        except Exception as exc:
            logger.error("Camera capture failure: %s", exc)
            raise RuntimeError(f"Camera capture failure: {exc}") from exc

    def _capture_simulation(self) -> bytes:
        # 1. Use user-configured mock image if it exists
        if self._mock_path and self._mock_path.is_file():
            return self._mock_path.read_bytes()

        # 2. Look for any sample image in the project dataset
        sample_dirs = [
            BASE_DIR / "data" / "devices" / "valid",
            BASE_DIR / "data" / "devices" / "train",
        ]
        for sdir in sample_dirs:
            if sdir.is_dir():
                for img_file in sdir.glob("*.jpg"):
                    return img_file.read_bytes()

        # 3. Fallback: generate a simple synthetic JPEG in memory using PIL
        try:
            from PIL import Image

            img = Image.new("RGB", self._resolution, color=(73, 109, 137))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
        except ImportError:
            # Minimal 1x1 valid JPEG byte sequence fallback
            return (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00"
                b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
                b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
                b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
                b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4"
                b"\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
                b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
            )


class MockCamera(Camera):
    """Convenience simulation camera for tests and development."""

    def __init__(
        self,
        mock_image_path: Optional[Union[str, Path]] = None,
        resolution: tuple[int, int] = (640, 480),
    ):
        super().__init__(
            simulation=True,
            mock_image_path=mock_image_path,
            resolution=resolution,
        )
