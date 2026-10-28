"""
Hardware abstraction: RFID reader (MFRC522).

In simulation mode, queued or pre-configured UIDs are returned without hardware.
In physical mode, connects via SPI0:
  SDA/SS: GPIO 8  (Pin 24)
  SCK:    GPIO 11 (Pin 23)
  MOSI:   GPIO 10 (Pin 19)
  MISO:   GPIO 9  (Pin 21)
  RST:    GPIO 25 (Pin 22)

Includes:
- UID normalization (uppercase stripped string)
- Card-held debouncing (prevents a held card from repeatedly triggering)
- Sequential UID queueing for simulation/tests (user tag then item tag)
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel value returned when no card is present
NO_CARD = None


def normalize_uid(raw_uid: Optional[str | int | bytes]) -> str:
    """Normalize any UID representation to an uppercase stripped hexadecimal string."""
    if raw_uid is None:
        raise ValueError("UID cannot be None")

    if isinstance(raw_uid, bytes):
        return "".join(f"{b:02X}" for b in raw_uid)
    if isinstance(raw_uid, int):
        return hex(raw_uid).upper().replace("0X", "").strip()

    s = str(raw_uid).strip().upper()
    if not s:
        raise ValueError("UID cannot be empty")
    return s


class RFIDReader:
    """Abstraction over an RFID reader (MFRC522-compatible).

    Parameters
    ----------
    simulation : bool
        When True, software simulation is used instead of GPIO/SPI hardware.
    mock_uid : str | None
        Default fallback UID that ``read_uid()`` returns in simulation mode.
    timeout_seconds : float
        How long to poll before returning None.
    hold_off_seconds : float
        Minimum duration before the same card held against the reader can be read again.
    """

    def __init__(
        self,
        simulation: bool = True,
        mock_uid: Optional[str] = None,
        timeout_seconds: float = 10.0,
        hold_off_seconds: float = 2.0,
    ):
        self._simulation = simulation
        self._mock_uid = normalize_uid(mock_uid) if mock_uid is not None else None
        self._timeout = max(0.1, float(timeout_seconds))
        self._hold_off = max(0.0, float(hold_off_seconds))

        self._queue: list[str] = []
        self._last_read_uid: Optional[str] = None
        self._last_read_time: float = 0.0

        self._reader = None
        self._reader_type = ""
        self._closed: bool = False
        self._lock = threading.Lock()

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Simulation Interface
    # ------------------------------------------------------------------

    def queue_uid(self, uid: str) -> None:
        """Queue a UID to be returned by sequential ``read_uid()`` calls (simulation mode)."""
        norm = normalize_uid(uid)
        with self._lock:
            if self._closed:
                raise RuntimeError("RFIDReader is closed")
            self._queue.append(norm)

    def set_mock_uid(self, uid: Optional[str]) -> None:
        """Update the default mock UID returned when queue is empty."""
        with self._lock:
            self._mock_uid = normalize_uid(uid) if uid is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_uid(self) -> Optional[str]:
        """Poll for a card UID until detected or timeout expires.

        Returns normalized uppercase UID string on success, or None on timeout.
        Suppresses duplicate scans if the same tag is held against the reader.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("RFIDReader is closed")

            if self._simulation:
                if self._queue:
                    return self._queue.pop(0)
                return self._mock_uid

        return self._read_hardware_uid()

    def close(self) -> None:
        """Release hardware and simulation resources."""
        with self._lock:
            self._closed = True
            self._queue.clear()
            if not self._simulation and self._reader is not None:
                try:
                    if hasattr(self._reader, "cleanup"):
                        self._reader.cleanup()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error closing RFID reader: %s", exc)
                finally:
                    self._reader = None

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        try:
            try:
                from mfrc522 import SimpleMFRC522  # type: ignore[import]

                self._reader = SimpleMFRC522()
                self._reader_type = "simple"
                logger.info("RFID SimpleMFRC522 hardware initialised")
                return
            except (ImportError, AttributeError):
                pass

            import mfrc522  # type: ignore[import]

            self._reader = mfrc522.MFRC522()
            self._reader_type = "raw"
            logger.info("RFID raw MFRC522 hardware initialised")
        except ImportError as exc:
            raise RuntimeError(
                "mfrc522 library not available. "
                "Run on a Raspberry Pi with mfrc522 and spidev installed, or set simulation=True."
            ) from exc

    def _read_hardware_uid(self) -> Optional[str]:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._closed or self._reader is None:
                    return None

            try:
                candidate_uid: Optional[str] = None

                if self._reader_type == "simple":
                    if hasattr(self._reader, "read_no_block"):
                        uid, _ = self._reader.read_no_block()
                    else:
                        uid, _ = self._reader.read()
                    if uid is not None:
                        candidate_uid = normalize_uid(hex(uid).upper().replace("0X", ""))
                else:
                    (status, _tag_type) = self._reader.MFRC522_Request(
                        self._reader.PICC_REQIDL
                    )
                    if status == self._reader.MI_OK:
                        (status, uid_bytes) = self._reader.MFRC522_Anticoll()
                        if status == self._reader.MI_OK and uid_bytes:
                            candidate_uid = "".join(f"{b:02X}" for b in uid_bytes[:4])

                if candidate_uid:
                    # Prevent rapid repeated triggers for a held card
                    now = time.monotonic()
                    if (
                        candidate_uid == self._last_read_uid
                        and (now - self._last_read_time) < self._hold_off
                    ):
                        time.sleep(0.08)
                        continue

                    self._last_read_uid = candidate_uid
                    self._last_read_time = now
                    return candidate_uid

            except Exception as exc:
                logger.warning("RFID hardware read exception: %s", exc)
                return None

            time.sleep(0.08)

        logger.debug("RFID read timed out after %.1fs", self._timeout)
        return None


class MockRFID(RFIDReader):
    """Convenience simulation reader for testing and development."""

    def __init__(self, mock_uid: Optional[str] = None):
        super().__init__(simulation=True, mock_uid=mock_uid)


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_reader: Optional[RFIDReader] = None


def get_reader(
    simulation: bool = True,
    mock_uid: Optional[str] = None,
) -> RFIDReader:
    """Return the module-level default reader, creating it if necessary."""
    global _default_reader
    if _default_reader is None:
        _default_reader = RFIDReader(simulation=simulation, mock_uid=mock_uid)
    return _default_reader


def reset_reader() -> None:
    """Close and discard the default reader (useful for tests)."""
    global _default_reader
    if _default_reader is not None:
        _default_reader.close()
        _default_reader = None
