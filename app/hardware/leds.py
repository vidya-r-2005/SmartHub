"""
Hardware abstraction: Status indicator LEDs.

Manages three status LEDs:
  - Green: GPIO 20 (Physical pin 38) — Operation successful / system ready
  - Red:   GPIO 21 (Physical pin 40) — Error or access denied
  - Blue:  GPIO 26 (Physical pin 37) — Busy or active charging session
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_LED_PINS = {
    "GREEN": 20,  # Physical pin 38
    "RED": 21,    # Physical pin 40
    "BLUE": 26,   # Physical pin 37
}

# Supported status states and which LED colors should be active
STATUS_MAP: Dict[str, set[str]] = {
    "READY": {"GREEN"},
    "SUCCESS": {"GREEN"},
    "GREEN": {"GREEN"},
    "CHARGING": {"BLUE"},
    "BUSY": {"BLUE"},
    "BLUE": {"BLUE"},
    "ERROR": {"RED"},
    "DENIED": {"RED"},
    "RED": {"RED"},
    "ALL_OFF": set(),
}


class LEDs:
    """Controls the kiosk three-color status LEDs.

    Parameters
    ----------
    simulation : bool
        When True, LED states are tracked in software without GPIO access.
    pins : dict[str, int] | None
        Mapping of 'GREEN', 'RED', 'BLUE' to BCM GPIO pin numbers.
    """

    def __init__(
        self,
        simulation: bool = True,
        pins: Optional[Dict[str, int]] = None,
    ):
        self._simulation = simulation
        self._pins = pins or dict(DEFAULT_LED_PINS)
        self._state: Dict[str, bool] = {"GREEN": False, "RED": False, "BLUE": False}
        self._current_status: str = "ALL_OFF"
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_status(self) -> str:
        """The currently applied status name."""
        return self._current_status

    @property
    def status(self) -> str:
        """Normalized lowercase status string."""
        return self._current_status.lower()

    def is_on(self, color: str) -> bool:
        """Return True if the specified color LED ('GREEN', 'RED', 'BLUE') is lit."""
        return self._state.get(color.upper(), False)

    def set_status(self, name: str) -> None:
        """Set LED status according to predefined state names.

        Supported names: 'READY', 'SUCCESS', 'CHARGING', 'BUSY', 'ERROR', 'DENIED', 'ALL_OFF'.
        """
        if self._closed:
            return

        normalized = name.strip().upper()
        active_colors = STATUS_MAP.get(normalized, set())
        self._current_status = normalized

        for color in ("GREEN", "RED", "BLUE"):
            turn_on = color in active_colors
            self._state[color] = turn_on
            if not self._simulation:
                self._set_hardware_pin(self._pins[color], turn_on)

        logger.debug(
            "[LEDs] status=%s (Green=%s, Red=%s, Blue=%s)",
            normalized,
            self._state["GREEN"],
            self._state["RED"],
            self._state["BLUE"],
        )

    def all_off(self) -> None:
        """Turn off all status LEDs."""
        self.set_status("ALL_OFF")

    def close(self) -> None:
        """Turn off all LEDs and release hardware resources."""
        self._closed = True
        self.all_off()

        if not self._simulation:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                for pin in self._pins.values():
                    GPIO.cleanup(pin)
                logger.info("Status LEDs cleaned up")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up LED pins: %s", exc)

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            for color, pin in self._pins.items():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)  # Initial safe OFF state
            logger.info("LEDs initialised on pins %s", self._pins)
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc

    def _set_hardware_pin(self, pin: int, high: bool) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.output(pin, GPIO.HIGH if high else GPIO.LOW)
        except Exception as exc:
            logger.error("Failed to set LED pin %d: %s", pin, exc)


class MockLEDs(LEDs):
    """Convenience simulation LEDs for testing and development."""

    def __init__(self, pins: Optional[Dict[str, int]] = None):
        super().__init__(simulation=True, pins=pins)
