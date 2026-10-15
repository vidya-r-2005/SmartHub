"""
Hardware abstraction: Charging power control (relay / MOSFET).

Controls whether the charging port is enabled or disabled.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ChargingController:
    """Control a relay or MOSFET that switches a charging port.

    Parameters
    ----------
    simulation : bool
        When True, state changes are only tracked in memory.
    pin : int | None
        BCM GPIO pin number for the relay.  Required when *simulation* is False.
    active_high : bool
        If True, HIGH signal enables charging.  If False, LOW enables charging
        (common for active-low relay modules).
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: Optional[int] = None,
        active_high: bool = True,
    ):
        self._simulation = simulation
        self._pin = pin
        self._active_high = active_high
        self._charging: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_charging(self) -> bool:
        """True when the charging port is enabled."""
        return self._charging

    def enable(self) -> bool:
        """Enable the charging port.  Returns True on success."""
        if self._simulation:
            self._charging = True
            logger.debug("[SIM] Charging port ENABLED")
            return True
        return self._set_hardware(True)

    def disable(self) -> bool:
        """Disable the charging port.  Returns True on success."""
        if self._simulation:
            self._charging = False
            logger.debug("[SIM] Charging port DISABLED")
            return True
        return self._set_hardware(False)

    def close(self) -> None:
        """Release hardware resources and ensure port is disabled."""
        self.disable()
        if not self._simulation and self._pin is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                GPIO.cleanup(self._pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up charging pin: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical charging controller")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            # Start with charging disabled
            GPIO.output(self._pin, GPIO.LOW if self._active_high else GPIO.HIGH)
            logger.info("Charging controller initialised on pin %d", self._pin)
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not available. "
                "Run on a Raspberry Pi or set simulation=True."
            )

    def _set_hardware(self, enable: bool) -> bool:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            level = GPIO.HIGH if (enable == self._active_high) else GPIO.LOW
            GPIO.output(self._pin, level)
            self._charging = enable
            logger.debug(
                "Charging port %s via GPIO pin %d",
                "ENABLED" if enable else "DISABLED",
                self._pin,
            )
            return True
        except Exception as exc:
            logger.error("Failed to set charging pin: %s", exc)
            return False

