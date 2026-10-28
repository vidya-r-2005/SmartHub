"""
Hardware abstraction: STORE servo mechanism.

Pin: BCM GPIO 18 (Physical pin 12).
Operates the physical compartment lock using PWM servo pulses:
  - Locked position:   0 degrees  (~2.5% - 5.0% duty cycle)
  - Unlocked position: 90 degrees (~7.5% - 10.0% duty cycle)

Applies 'Safe Release': After commanding the servo to rotate, the PWM duty
cycle is set to 0 to prevent continuous pulse driving, jitter, and motor heating.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SERVO_PIN = 18  # BCM GPIO 18 (Physical pin 12)
SERVO_PWM_FREQ = 50     # 50 Hz standard hobby servo frequency

# Duty cycles (percentage) for 50Hz PWM
DUTY_LOCKED = 3.0       # ~0 degrees
DUTY_UNLOCKED = 8.5     # ~90 degrees
SERVO_MOVE_DELAY = 0.4  # seconds to allow servo arm to reach position


class LockController:
    """Controls the STORE locker servo actuator with safe pulse release.

    Parameters
    ----------
    simulation : bool
        When True, lock state is tracked in software without GPIO PWM.
    pin : int
        BCM GPIO pin number. Default is GPIO 18.
    fail_result : bool | None
        Optional fixed boolean return value injected for automated unit tests.
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: int = DEFAULT_SERVO_PIN,
        active_high: bool = True,  # Maintained for legacy signature compatibility
        fail_result: Optional[bool] = None,
    ):
        self._simulation = simulation
        self._pin = pin
        self._locked: bool = True
        self._fail_result = fail_result
        self._pwm = None
        self._lock = threading.Lock()
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_locked(self) -> bool:
        """True when the locker is engaged in the locked position."""
        with self._lock:
            return self._locked

    def unlock(self) -> bool:
        """Command the servo to rotate to the UNLOCKED position.

        Returns True on success.
        """
        if self._fail_result is not None:
            return self._fail_result

        with self._lock:
            if self._closed:
                return False

            self._locked = False
            if self._simulation:
                logger.debug("[SIM] Servo lock UNLOCKED")
                return True

            return self._command_servo(DUTY_UNLOCKED, locked=False)

    def lock(self) -> bool:
        """Command the servo to rotate to the LOCKED position.

        Returns True on success.
        """
        if self._fail_result is not None:
            return self._fail_result

        with self._lock:
            if self._closed:
                return False

            self._locked = True
            if self._simulation:
                logger.debug("[SIM] Servo lock LOCKED")
                return True

            return self._command_servo(DUTY_LOCKED, locked=True)

    def close(self) -> None:
        """Ensure servo is locked and release PWM hardware resources."""
        with self._lock:
            self._closed = True
            if not self._simulation and self._pwm is not None:
                try:
                    self._command_servo(DUTY_LOCKED, locked=True)
                    self._pwm.stop()
                    import RPi.GPIO as GPIO  # type: ignore[import]

                    GPIO.cleanup(self._pin)
                    logger.info("Servo pin %d cleaned up", self._pin)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error cleaning up servo lock pin %d: %s", self._pin, exc)
                finally:
                    self._pwm = None

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical servo lock")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, SERVO_PWM_FREQ)
            self._pwm.start(0)  # Start with 0 duty cycle (safe release)
            # Ensure safe locked position on startup
            self._command_servo(DUTY_LOCKED, locked=True)
            logger.info("Servo LockController initialised on GPIO pin %d (50Hz)", self._pin)
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc

    def _command_servo(self, duty: float, locked: bool) -> bool:
        """Move servo to target duty cycle, wait for transit, and release pulse train."""
        if self._pwm is None:
            return False
        try:
            self._pwm.ChangeDutyCycle(duty)
            time.sleep(SERVO_MOVE_DELAY)
            # Safe Release: Set duty to 0 so servo doesn't buzz or overheat
            self._pwm.ChangeDutyCycle(0)
            self._locked = locked
            logger.debug(
                "Servo on GPIO %d moved to %s (duty=%.1f%%) and released",
                self._pin,
                "LOCKED" if locked else "UNLOCKED",
                duty,
            )
            return True
        except Exception as exc:
            logger.error("Failed commanding servo on pin %d: %s", self._pin, exc)
            return False


class MockLock(LockController):
    """Convenience simulation lock for tests and development."""

    def __init__(self, pin: int = DEFAULT_SERVO_PIN, fail_result: Optional[bool] = None):
        super().__init__(simulation=True, pin=pin, fail_result=fail_result)
