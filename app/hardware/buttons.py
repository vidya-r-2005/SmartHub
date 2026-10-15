"""
Hardware abstraction: Physical push-buttons.

In simulation mode buttons are never pressed (useful for automated tests)
unless explicitly triggered via ``simulate_press()``.
"""

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class Button:
    """Single push-button with optional callback.

    Parameters
    ----------
    simulation : bool
        When True, no GPIO is initialised.
    pin : int | None
        BCM GPIO pin number.  Required when *simulation* is False.
    pull_up : bool
        If True, GPIO pull-up is enabled (button connects to GND).
        If False, pull-down (button connects to VCC).
    callback : Callable | None
        Function called on button press.  Receives no arguments.
    bouncetime_ms : int
        Debounce time in milliseconds.
    """

    def __init__(
        self,
        simulation: bool = True,
        pin: Optional[int] = None,
        pull_up: bool = True,
        callback: Optional[Callable[[], None]] = None,
        bouncetime_ms: int = 200,
    ):
        self._simulation = simulation
        self._pin = pin
        self._pull_up = pull_up
        self._callback = callback
        self._bouncetime = bouncetime_ms
        self._press_count = 0
        self._lock = threading.Lock()

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def press_count(self) -> int:
        """Number of times the button has been pressed."""
        with self._lock:
            return self._press_count

    def simulate_press(self) -> None:
        """Manually trigger a button press (simulation mode only)."""
        if not self._simulation:
            raise RuntimeError("simulate_press is only available in simulation mode")
        self._on_press(None)

    def close(self) -> None:
        """Release hardware resources."""
        if not self._simulation and self._pin is not None:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                GPIO.remove_event_detect(self._pin)
                GPIO.cleanup(self._pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing button pin %d: %s", self._pin, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pin is None:
            raise ValueError("pin must be set for physical button")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            pud = GPIO.PUD_UP if self._pull_up else GPIO.PUD_DOWN
            GPIO.setup(self._pin, GPIO.IN, pull_up_down=pud)
            edge = GPIO.FALLING if self._pull_up else GPIO.RISING
            GPIO.add_event_detect(
                self._pin,
                edge,
                callback=self._on_press,
                bouncetime=self._bouncetime,
            )
            logger.info("Button initialised on pin %d", self._pin)
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not available. "
                "Run on a Raspberry Pi or set simulation=True."
            )

    def _on_press(self, _channel) -> None:
        with self._lock:
            self._press_count += 1
        logger.debug("Button pressed (pin=%s, total=%d)", self._pin, self._press_count)
        if self._callback is not None:
            try:
                self._callback()
            except Exception as exc:
                logger.error("Button callback raised an exception: %s", exc)

