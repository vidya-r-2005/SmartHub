"""
Hardware abstraction: Electromagnetic lock / solenoid actuator.

Controls a door lock for the STORE service.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LockController:
    """Control an electromagnetic lock or solenoid actuator.

    Parameters
    ----------
    simulation : bool
        When True, lock state is tracked in software only.
    pin : int | None
        BCM GPIO pin number.  Required when *simulation* is False.
    active_high : bool
        If True, HIGH unlocks.  If False, LOW unlocks (common for
        fail-secure magnetic locks powered through a relay).
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: Optional[int] = None,
        active_high: bool = True,
        fail_result: Optional[bool] = None,
    ):
        self._simulation = simulation
        self._pin = pin
        self._active_high = active_high
        self._locked: bool = True
        # For testing: inject a fixed result for unlock/lock calls
        self._fail_result = fail_result

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True when the lock is engaged."""
        return self._locked

    def unlock(self) -> bool:
        """Disengage the lock.  Returns True on success."""
        if self._fail_result is not None:
            return self._fail_result

        if self._simulation:
            self._locked = False
            logger.debug("[SIM] Lock UNLOCKED")
            return True
        return self._set_hardware(locked=False)

    def lock(self) -> bool:
        """Re-engage the lock.  Returns True on success."""
        if self._fail_result is not None:
            return self._fail_result

        if self._simulation:
            self._locked = True
            logger.debug("[SIM] Lock LOCKED")
            return True
        return self._set_hardware(locked=True)

    def close(self) -> None:
        """Release hardware resources — always locks first."""
        self.lock()
        if not self._simulation and self._pin is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                GPIO.cleanup(self._pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error cleaning up lock pin: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical lock controller")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            # Ensure locked on startup
            GPIO.output(self._pin, GPIO.LOW if self._active_high else GPIO.HIGH)
            logger.info("Lock controller initialised on pin %d", self._pin)
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not available. "
                "Run on a Raspberry Pi or set simulation=True."
            )

    def _set_hardware(self, locked: bool) -> bool:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            # locked=True → we want to *engage* the lock
            # For active_high unlock: unlock=HIGH, lock=LOW
            unlocked = not locked
            level = GPIO.HIGH if (unlocked == self._active_high) else GPIO.LOW
            GPIO.output(self._pin, level)
            self._locked = locked
            logger.debug(
                "Lock %s via GPIO pin %d",
                "LOCKED" if locked else "UNLOCKED",
                self._pin,
            )
            return True
        except Exception as exc:
            logger.error("Failed to set lock pin: %s", exc)
            return False

