"""
Hardware abstraction: Audio feedback buzzer.

Pin: BCM GPIO 22 (Physical pin 15).
Provides non-blocking audio feedback patterns for system events:
  - beep(duration, count, gap)
  - success(): short double confirmation beep
  - error(): longer alert tone
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_BUZZER_PIN = 22  # BCM GPIO 22 (Physical pin 15)


class Buzzer:
    """Controls the piezo buzzer via GPIO PWM.

    Parameters
    ----------
    simulation : bool
        When True, sound sequences are logged rather than played on hardware.
    pin : int
        BCM GPIO pin number. Default is GPIO 22.
    frequency : int
        Tone frequency in Hz (default 1000 Hz).
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: int = DEFAULT_BUZZER_PIN,
        frequency: int = 1000,
    ):
        self._simulation = simulation
        self._pin = pin
        self._frequency = frequency
        self._pwm = None
        self._lock = threading.Lock()
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def beep(
        self,
        duration: float = 0.2,
        count: int = 1,
        gap: float = 0.1,
        blocking: bool = False,
    ) -> None:
        """Play *count* beeps of *duration* seconds separated by *gap* seconds.

        Runs asynchronously in a background thread by default so it does not
        block the application.
        """
        if self._closed:
            return

        if self._simulation:
            logger.debug(
                "[SIM] Buzzer: %d beep(s) duration=%.2fs gap=%.2fs freq=%dHz",
                count,
                duration,
                gap,
                self._frequency,
            )
            return

        def _play():
            with self._lock:
                for i in range(count):
                    if self._closed:
                        break
                    self._start_tone()
                    time.sleep(duration)
                    self._stop_tone()
                    if i < count - 1:
                        time.sleep(gap)

        if blocking:
            _play()
        else:
            threading.Thread(target=_play, daemon=True).start()

    def success(self) -> None:
        """Play a short double-beep confirmation pattern."""
        self.beep(duration=0.08, count=2, gap=0.05)

    def success_tone(self) -> None:
        """Backwards-compatible alias for success()."""
        self.success()

    def error(self) -> None:
        """Play a long alert tone for errors or access denial."""
        self.beep(duration=0.45, count=1)

    def error_tone(self) -> None:
        """Backwards-compatible alias for error()."""
        self.error()

    def close(self) -> None:
        """Release PWM and GPIO resources."""
        with self._lock:
            self._closed = True
            if not self._simulation and self._pwm is not None:
                try:
                    self._pwm.stop()
                    import RPi.GPIO as GPIO  # type: ignore[import]

                    GPIO.cleanup(self._pin)
                    logger.info("Buzzer pin %d cleaned up", self._pin)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error closing buzzer pin %d: %s", self._pin, exc)
                finally:
                    self._pwm = None

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical buzzer")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, self._frequency)
            logger.info("Buzzer initialised on GPIO pin %d (%d Hz)", self._pin, self._frequency)
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc

    def _start_tone(self) -> None:
        if self._pwm is not None:
            try:
                self._pwm.start(50)  # 50% duty cycle
            except Exception:
                pass

    def _stop_tone(self) -> None:
        if self._pwm is not None:
            try:
                self._pwm.stop()
            except Exception:
                pass


class MockBuzzer(Buzzer):
    """Convenience simulation buzzer for tests and development."""

    def __init__(self, pin: int = DEFAULT_BUZZER_PIN, frequency: int = 1000):
        super().__init__(simulation=True, pin=pin, frequency=frequency)
