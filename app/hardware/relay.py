"""
Hardware abstraction: Physical charging output relay.

Switches low-voltage (5V) power for the charging port.
Supports configurable polarity (active-high or active-low).
Always ensures safe OFF state during initialization and cleanup.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RELAY_PIN = 27  # BCM GPIO 27 (Physical Pin 13)


class Relay:
    """Controls a physical relay or MOSFET switching the charging output.

    Parameters
    ----------
    simulation : bool
        When True, relay state changes are recorded in memory only.
    pin : int
        BCM GPIO pin number. Default is GPIO 27.
    active_high : bool
        If True, GPIO HIGH turns the relay ON (closed).
        If False, GPIO LOW turns the relay ON (active-low module).
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: int = DEFAULT_RELAY_PIN,
        active_high: bool = True,
    ):
        self._simulation = simulation
        self._pin = pin
        self._active_high = active_high
        self._is_on: bool = False
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_on(self) -> bool:
        """True when the relay output is currently powered ON."""
        return self._is_on

    def turn_on(self) -> bool:
        """Turn on the relay output. Returns True on success."""
        if self._closed:
            logger.warning("Attempted to turn on a closed Relay")
            return False

        self._is_on = True
        if self._simulation:
            logger.debug("[SIM] Relay pin %d turned ON", self._pin)
            return True

        return self._set_hardware(True)

    def turn_off(self) -> bool:
        """Turn off the relay output. Returns True on success."""
        self._is_on = False
        if self._simulation:
            logger.debug("[SIM] Relay pin %d turned OFF", self._pin)
            return True

        return self._set_hardware(False)

    def close(self) -> None:
        """Ensure relay is turned OFF and release GPIO resources."""
        self._closed = True
        self.turn_off()

        if not self._simulation and self._pin is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                # Force safe output state before cleanup
                off_level = GPIO.LOW if self._active_high else GPIO.HIGH
                GPIO.output(self._pin, off_level)
                GPIO.cleanup(self._pin)
                logger.info("Relay pin %d cleaned up", self._pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up relay pin %d: %s", self._pin, exc)

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical relay")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            # Default safe OFF state at boot
            off_level = GPIO.LOW if self._active_high else GPIO.HIGH
            GPIO.output(self._pin, off_level)
            self._is_on = False
            logger.info(
                "Relay initialised on GPIO pin %d (active_%s, initial=OFF)",
                self._pin,
                "high" if self._active_high else "low",
            )
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc
        except Exception as exc:
            self._is_on = False
            logger.error("Failed to initialise relay hardware on pin %d: %s", self._pin, exc)
            raise

    def _set_hardware(self, on: bool) -> bool:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            level = GPIO.HIGH if (on == self._active_high) else GPIO.LOW
            GPIO.output(self._pin, level)
            self._is_on = on
            logger.debug(
                "Relay pin %d set to %s (GPIO %s)",
                self._pin,
                "ON" if on else "OFF",
                "HIGH" if level == GPIO.HIGH else "LOW",
            )
            return True
        except Exception as exc:
            logger.error("Failed to set relay pin %d: %s", self._pin, exc)
            return False


class MockRelay(Relay):
    """Convenience simulation relay for testing and Windows development."""

    def __init__(self, pin: int = DEFAULT_RELAY_PIN, active_high: bool = True):
        super().__init__(simulation=True, pin=pin, active_high=active_high)
