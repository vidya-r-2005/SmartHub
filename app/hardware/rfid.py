"""
Hardware abstraction: RFID reader.

In simulation mode a pre-configured UID is returned instead of reading
physical hardware.  Physical mode defers import of RPi-specific libraries
so the module can be imported on any machine.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Sentinel value returned when no card is present
NO_CARD = None


class RFIDReader:
    """Abstraction over an RFID reader (MFRC522-compatible).

    Parameters
    ----------
    simulation : bool
        When True, use software simulation instead of GPIO/SPI hardware.
    mock_uid : str | None
        UID that ``read_uid()`` returns in simulation mode.
    timeout_seconds : float
        How long to poll before giving up.
    """

    def __init__(
        self,
        simulation: bool = True,
        mock_uid: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ):
        self._simulation = simulation
        self._mock_uid = mock_uid
        self._timeout = timeout_seconds
        self._reader = None

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_uid(self) -> Optional[str]:
        """Block until a card is detected or timeout expires.

        Returns the UID string on success, or *None* on timeout / error.
        """
        if self._simulation:
            return self._mock_uid

        return self._read_hardware_uid()

    def close(self) -> None:
        """Release hardware resources."""
        if not self._simulation and self._reader is not None:
            try:
                self._reader.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing RFID reader: %s", exc)
            self._reader = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        try:
            import mfrc522  # type: ignore[import]

            self._reader = mfrc522.MFRC522()
            logger.info("RFID hardware initialised")
        except ImportError:
            raise RuntimeError(
                "mfrc522 library not available. "
                "Run on a Raspberry Pi with the library installed, "
                "or set simulation=True."
            )

    def _read_hardware_uid(self) -> Optional[str]:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            (status, _tag_type) = self._reader.MFRC522_Request(
                self._reader.PICC_REQIDL
            )
            if status == self._reader.MI_OK:
                (status, uid) = self._reader.MFRC522_Anticoll()
                if status == self._reader.MI_OK:
                    return "".join(f"{b:02X}" for b in uid)
            time.sleep(0.1)
        logger.warning("RFID read timeout after %.1fs", self._timeout)
        return None


# ---------------------------------------------------------------------------
# Module-level convenience singleton (simulation by default)
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

