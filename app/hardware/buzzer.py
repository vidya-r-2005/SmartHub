"""
Hardware abstraction: Passive buzzer.

In simulation mode buzzer actions are only logged.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class Buzzer:
    """Passive buzzer controlled via PWM.

    Parameters
    ----------
    simulation : bool
        When True, beeps are only logged.
    pin : int | None
        BCM GPIO pin number.  Required when *simulation* is False.
    frequency : int
        Tone frequency in Hz (default 1000 Hz).
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: Optional[int] = None,
        frequency: int = 1000,
    ):
        self._simulation = simulation
        self._pin = pin
        self._frequency = frequency
        self._pwm = None

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def beep(self, duration: float = 0.2, count: int = 1, gap: float = 0.1) -> None:
        """Emit *count* beeps each lasting *duration* seconds.

        Parameters
        ----------
        duration : float
            Length of each beep in seconds.
        count : int
            Number of beeps.
        gap : float
            Silence between beeps in seconds.
        """
        if self._simulation:
            logger.debug(
                "[SIM] Buzzer: %d beep(s) duration=%.2fs gap=%.2fs",
                count,
                duration,
                gap,
            )
            return

        for i in range(count):
            self._start_tone()
            time.sleep(duration)
            self._stop_tone()
            if i < count - 1:
                time.sleep(gap)

    def success_tone(self) -> None:
        """Play a short success sequence."""
        self.beep(duration=0.1, count=2, gap=0.05)

    def error_tone(self) -> None:
        """Play a long error tone."""
        self.beep(duration=0.5, count=1)

    def close(self) -> None:
        """Release hardware resources."""
        if not self._simulation and self._pwm is not None:
            try:
                self._pwm.stop()
                import RPi.GPIO as GPIO  # type: ignore[import]

                GPIO.cleanup(self._pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing buzzer: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical buzzer")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pin, GPIO.OUT)
            self._pwm = GPIO.PWM(self._pin, self._frequency)
            logger.info("Buzzer initialised on pin %d", self._pin)
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not available. "
                "Run on a Raspberry Pi or set simulation=True."
            )

    def _start_tone(self) -> None:
        if self._pwm is not None:
            self._pwm.start(50)  # 50% duty cycle

    def _stop_tone(self) -> None:
        if self._pwm is not None:
            self._pwm.stop()

