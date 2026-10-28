"""
Hardware abstraction: Charging-session controller.

Manages timed charging sessions, deadlines, remaining time calculations,
and delegates physical power control to app/hardware/relay.Relay.

Pricing conversion rule:
  duration_seconds = accepted_coins * 300
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Union

from app.hardware.relay import MockRelay, Relay

logger = logging.getLogger(__name__)

SECONDS_PER_COIN = 300  # 5 minutes per ₹5 coin


class ChargingState(int):
    """Backwards-compatible truthy integer that also supports callable evaluation."""

    def __call__(self) -> bool:
        return bool(self)


class ChargingController:
    """Session controller managing charging power and timed deadlines.

    Parameters
    ----------
    relay_or_simulation : Relay | bool
        Injected Relay instance, or boolean simulation flag.
    simulation : bool | None
        Explicit simulation override.
    seconds_per_credit : float
        Seconds allocated per credit unit (default 300s = 5 mins).
    clock : Callable[[], float] | None
        Monotonic time provider (for deterministic testing).
    pin : int | None
        Relay GPIO pin (default 27).
    active_high : bool
        Relay active polarity (default True).
    """

    def __init__(
        self,
        relay_or_simulation: Union[Relay, bool] = True,
        simulation: Optional[bool] = None,
        seconds_per_credit: float = SECONDS_PER_COIN,
        clock: Optional[Callable[[], float]] = None,
        pin: Optional[int] = 27,
        active_high: bool = True,
    ):
        if not isinstance(seconds_per_credit, (int, float)) or isinstance(seconds_per_credit, bool) or seconds_per_credit <= 0:
            raise ValueError("seconds_per_credit must be a positive number")

        self.seconds_per_credit: float = float(seconds_per_credit)
        self.clock: Callable[[], float] = clock or time.monotonic
        self._lock = threading.Lock()
        self._deadline: Optional[float] = None
        self._charging: bool = False
        self._closed: bool = False

        # Resolve relay instance
        if isinstance(relay_or_simulation, Relay):
            self.relay: Relay = relay_or_simulation
            self._simulation = False
        else:
            sim = simulation if simulation is not None else bool(relay_or_simulation)
            self._simulation = sim
            if sim:
                self.relay = MockRelay(pin=pin or 27, active_high=active_high)
            else:
                self.relay = Relay(simulation=False, pin=pin or 27, active_high=active_high)

        # Safe initial state
        self.relay.turn_off()

    # ------------------------------------------------------------------
    # State Inspection
    # ------------------------------------------------------------------

    @property
    def is_charging(self) -> ChargingState:
        """True when the charging output is actively powered."""
        with self._lock:
            # First check if deadline has expired
            if self._deadline is not None and self.clock() >= self._deadline:
                self._charging = False
                self._deadline = None
                self.relay.turn_off()

            active = self._charging or self.relay.is_on
            return ChargingState(1 if active else 0)

    def seconds_remaining(self) -> int:
        """Return the integer seconds remaining until session deadline expires."""
        with self._lock:
            if not self._charging or self._deadline is None:
                return 0
            remaining = int(self._deadline - self.clock())
            if remaining <= 0:
                self._charging = False
                self._deadline = None
                self.relay.turn_off()
                return 0
            return remaining

    # ------------------------------------------------------------------
    # Timed Session API
    # ------------------------------------------------------------------

    def start(self, duration_seconds: float) -> float:
        """Start a timed charging session for the given duration in seconds.

        Returns the monotonic deadline timestamp.
        """
        if not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool):
            raise ValueError("Duration must be numeric")
        if duration_seconds <= 0:
            raise ValueError("Duration must be strictly positive")

        with self._lock:
            if self._closed:
                raise RuntimeError("ChargingController is closed")
            self._deadline = self.clock() + float(duration_seconds)
            self._charging = True
            self.relay.turn_on()
            logger.info(
                "Charging session started: duration=%.1fs, deadline=%.1f",
                duration_seconds,
                self._deadline,
            )
            return self._deadline

    def start_with_coins(self, coin_count: int) -> float:
        """Convenience method to start session using accepted ₹5 coin count."""
        if not isinstance(coin_count, int) or isinstance(coin_count, bool) or coin_count <= 0:
            raise ValueError("Coin count must be a positive integer")
        return self.start(coin_count * self.seconds_per_credit)

    def update(self) -> bool:
        """Evaluate session deadline; turns off relay if time has expired.

        Returns True if charging was stopped due to expiry, False otherwise.
        """
        with self._lock:
            if self._deadline is not None and self.clock() >= self._deadline:
                logger.info("Charging session completed (deadline reached)")
                self._charging = False
                self._deadline = None
                self.relay.turn_off()
                return True
            return False

    def stop(self) -> None:
        """Manually stop the charging session and disable power."""
        with self._lock:
            self._charging = False
            self._deadline = None
            self.relay.turn_off()
            logger.info("Charging session stopped manually")

    # ------------------------------------------------------------------
    # Direct Power Control (Backwards-compatibility for services/tests)
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        """Enable power output directly."""
        with self._lock:
            if self._closed:
                return False
            self._charging = True
            return self.relay.turn_on()

    def disable(self) -> bool:
        """Disable power output directly."""
        with self._lock:
            self._charging = False
            self._deadline = None
            return self.relay.turn_off()

    def close(self) -> None:
        """Release relay resources and ensure power is OFF."""
        with self._lock:
            self._closed = True
            self._charging = False
            self._deadline = None
            self.relay.close()
            logger.info("ChargingController closed")
