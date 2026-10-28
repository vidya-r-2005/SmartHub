"""
Hardware abstraction: Physical push-buttons.

Supports:
- Button: Single GPIO push-button with optional callback and debouncing.
- GPIOButtons: 4 navigation buttons on Raspberry Pi:
    CHARGE       (GPIO 5, Pin 29)
    STORE        (GPIO 6, Pin 31)
    BORROW       (GPIO 13, Pin 33)
    BACK_CONFIRM (GPIO 19, Pin 35)
- MockButtons: Development and automated test button simulator.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)

BUTTON_PINS: Dict[str, int] = {
    "CHARGE": 5,        # Physical Pin 29
    "STORE": 6,         # Physical Pin 31
    "BORROW": 13,       # Physical Pin 33
    "BACK_CONFIRM": 19, # Physical Pin 35
}


class MockButtons:
    """Simulated 4-button interface for Windows testing and automated development."""

    def __init__(self):
        self._pressed: Set[str] = set()
        self._callbacks: Dict[str, list[Callable[[], None]]] = {
            name: [] for name in self.valid_names()
        }
        self._lock = threading.Lock()

    @staticmethod
    def valid_names() -> Set[str]:
        return {"CHARGE", "STORE", "BORROW", "BACK_CONFIRM"}

    def is_pressed(self, name: str) -> bool:
        """Check whether the given button is currently pressed."""
        normalized = name.strip().upper()
        if normalized not in self.valid_names():
            raise ValueError(f"Unknown button: {name!r}. Valid: {sorted(self.valid_names())}")
        with self._lock:
            return normalized in self._pressed

    def press(self, name: str) -> None:
        """Simulate pressing a button."""
        normalized = name.strip().upper()
        if normalized not in self.valid_names():
            raise ValueError(f"Unknown button: {name!r}. Valid: {sorted(self.valid_names())}")
        with self._lock:
            self._pressed.add(normalized)
            callbacks = list(self._callbacks.get(normalized, []))

        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("Mock button callback error for %s: %s", normalized, exc)

    def release(self, name: str) -> None:
        """Simulate releasing a button."""
        normalized = name.strip().upper()
        if normalized not in self.valid_names():
            raise ValueError(f"Unknown button: {name!r}. Valid: {sorted(self.valid_names())}")
        with self._lock:
            self._pressed.discard(normalized)

    def on_press(self, name: str, callback: Callable[[], None]) -> None:
        """Register a callback for when this button is pressed."""
        normalized = name.strip().upper()
        if normalized not in self.valid_names():
            raise ValueError(f"Unknown button: {name!r}. Valid: {sorted(self.valid_names())}")
        with self._lock:
            self._callbacks[normalized].append(callback)

    def close(self) -> None:
        """Clear all active pressed states and callbacks."""
        with self._lock:
            self._pressed.clear()
            self._callbacks.clear()


class GPIOButtons:
    """Hardware button controller for Raspberry Pi using RPi.GPIO."""

    def __init__(
        self,
        pins: Optional[Dict[str, int]] = None,
        bouncetime_ms: int = 200,
    ):
        self._pins = pins or dict(BUTTON_PINS)
        self._bouncetime = bouncetime_ms
        self._pressed_states: Dict[str, bool] = {name: False for name in self._pins}
        self._callbacks: Dict[str, list[Callable[[], None]]] = {
            name: [] for name in self._pins
        }
        self._lock = threading.Lock()
        self._closed = False
        self._init_hardware()

    def is_pressed(self, name: str) -> bool:
        """Return True if button is currently physically pressed (reads low with pull-up)."""
        normalized = name.strip().upper()
        if normalized not in self._pins:
            raise ValueError(f"Unknown button: {name!r}. Valid: {list(self._pins.keys())}")

        if self._closed:
            return False

        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            pin = self._pins[normalized]
            # Pull-up resistor: pressed connects to GND -> LOW (0)
            return GPIO.input(pin) == GPIO.LOW
        except Exception as exc:
            logger.error("Error reading button pin for %s: %s", normalized, exc)
            return False

    def on_press(self, name: str, callback: Callable[[], None]) -> None:
        """Register a callback executed on falling edge (press)."""
        normalized = name.strip().upper()
        if normalized not in self._pins:
            raise ValueError(f"Unknown button: {name!r}")
        with self._lock:
            self._callbacks[normalized].append(callback)

    def close(self) -> None:
        """Clean up GPIO button interrupts."""
        self._closed = True
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            for pin in self._pins.values():
                try:
                    GPIO.remove_event_detect(pin)
                    GPIO.cleanup(pin)
                except Exception:
                    pass
            logger.info("GPIOButtons cleaned up successfully")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing GPIOButtons: %s", exc)

    def _init_hardware(self) -> None:
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            for name, pin in self._pins.items():
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                GPIO.add_event_detect(
                    pin,
                    GPIO.FALLING,
                    callback=lambda ch, n=name: self._handle_press(n),
                    bouncetime=self._bouncetime,
                )
            logger.info("GPIOButtons initialised on pins %s", self._pins)
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or use MockButtons."
            ) from exc

    def _handle_press(self, name: str) -> None:
        with self._lock:
            callbacks = list(self._callbacks.get(name, []))
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("Button callback exception for %s: %s", name, exc)


class Button:
    """Single push-button with optional callback (backwards-compatible).

    Parameters
    ----------
    simulation : bool
        When True, no GPIO is initialised.
    pin : int | None
        BCM GPIO pin number. Required when *simulation* is False.
    pull_up : bool
        If True, GPIO pull-up is enabled (button connects to GND).
        If False, pull-down (button connects to VCC).
    callback : Callable | None
        Function called on button press. Receives no arguments.
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
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc

    def _on_press(self, _channel) -> None:
        with self._lock:
            self._press_count += 1
        logger.debug("Button pressed (pin=%s, total=%d)", self._pin, self._press_count)
        if self._callback is not None:
            try:
                self._callback()
            except Exception as exc:
                logger.error("Button callback raised an exception: %s", exc)
