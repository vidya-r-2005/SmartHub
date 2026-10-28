"""
Hardware abstraction: Single ₹5 coin acceptor.

Physical Pin: BCM GPIO 17 (Physical pin 11).
Accepted denomination: ₹5.
Pricing Rule:
  1 coin (₹5)  -> 5 minutes  (300 seconds)
  2 coins (₹10) -> 10 minutes (600 seconds)
  3 coins (₹15) -> 15 minutes (900 seconds)

Features:
- Pulse-edge detection with noise debouncing
- Thread-safe credit and coin counting
- Non-blocking callback on accepted coins
- Backwards-compatible balance and collect() API for existing services
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_COIN_PIN = 17  # BCM GPIO 17 (Physical pin 11)
DEFAULT_COIN_VALUE = 5.0  # ₹5 per accepted coin
SECONDS_PER_COIN = 300   # 5 minutes per ₹5 coin


class CoinAcceptor:
    """Controls the single ₹5 pulse-count coin acceptor.

    Parameters
    ----------
    simulation : bool
        When True, software simulation is used instead of GPIO pulse interrupts.
    coin_value : float
        Value in currency units per coin (default 5.0).
    pulse_pin : int | None
        GPIO pin number (BCM). Defaults to GPIO 17.
    pull_up : bool
        If True, internal pull-up is enabled (falling edge active).
        If False, pull-down is enabled (rising edge active).
    bouncetime_ms : int
        Hardware debounce window in milliseconds.
    timeout_seconds : float
        Payment collection timeout for collect().
    on_coin_accepted : Callable[[int], None] | None
        Optional callback invoked whenever a valid coin is registered.
    """

    def __init__(
        self,
        simulation: bool = True,
        coin_value: float = DEFAULT_COIN_VALUE,
        credit_per_coin: Optional[float] = None,
        pulse_pin: Optional[int] = DEFAULT_COIN_PIN,
        pull_up: bool = True,
        bouncetime_ms: int = 60,
        timeout_seconds: float = 30.0,
        on_coin_accepted: Optional[Callable[[int], None]] = None,
    ):
        val = credit_per_coin if credit_per_coin is not None else coin_value
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val <= 0:
            raise ValueError("credit_per_coin must be a positive number")

        self.coin_value: float = float(val)
        self.credit_per_coin: float = float(val)
        self._pulse_pin: Optional[int] = pulse_pin
        self._pull_up: bool = pull_up
        self._bouncetime_ms: int = max(10, int(bouncetime_ms))
        self._timeout: float = max(0.1, float(timeout_seconds))
        self._simulation: bool = simulation
        self._on_coin_callback = on_coin_accepted

        self._lock = threading.Lock()
        self._coin_count: int = 0
        self._total_credit: float = 0.0
        self._balance: float = 0.0
        self._listening: bool = False
        self._closed: bool = False

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public & Proposed Spec API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin listening for coin pulses."""
        with self._lock:
            self._listening = True
            logger.debug("[CoinAcceptor] Started listening for coins")

    def stop(self) -> None:
        """Stop listening for new coin pulses."""
        with self._lock:
            self._listening = False
            logger.debug("[CoinAcceptor] Stopped listening for coins")

    def get_coin_count(self) -> int:
        """Return the total number of accepted coins since last reset."""
        with self._lock:
            return self._coin_count

    @property
    def coin_count(self) -> int:
        """Property returning the current coin count."""
        with self._lock:
            return self._coin_count

    @property
    def balance(self) -> float:
        """Current uncollected credit balance in rupees."""
        with self._lock:
            return self._balance

    def get_credit(self) -> float:
        """Total accumulated credit since last reset."""
        with self._lock:
            return self._total_credit

    def add_coin(self, count: int = 1) -> float:
        """Record accepted coins (simulation mode or test injection).

        Returns the added credit amount in rupees.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Coin count must be a positive integer")

        with self._lock:
            if self._closed:
                raise RuntimeError("CoinAcceptor is closed")
            added = count * self.coin_value
            self._coin_count += count
            self._total_credit += added
            self._balance += added
            callback = self._on_coin_callback
            current_total = self._coin_count

        if callback is not None:
            try:
                callback(current_total)
            except Exception as exc:
                logger.error("Coin accepted callback error: %s", exc)

        logger.debug(
            "[CoinAcceptor] Added %d coin(s) (total=%d, balance=%.2f)",
            count,
            current_total,
            self.balance,
        )
        return added

    def simulate_insert(self, coins: int = 1) -> None:
        """Convenience alias for simulation mode."""
        self.add_coin(coins)

    def insert_coins(self, count: int = 1) -> float:
        """Alias for add_coin() for simulation testing."""
        return self.add_coin(count)

    def collect(self, required_amount: float) -> bool:
        """Collect required_amount from balance.

        Returns True if sufficient balance is available or collected within timeout.
        Deducts required_amount from balance.
        """
        if not isinstance(required_amount, (int, float)) or isinstance(required_amount, bool):
            raise ValueError("required_amount must be numeric")
        if required_amount < 0:
            raise ValueError("required_amount cannot be negative")

        if required_amount == 0:
            return True

        if self._simulation:
            with self._lock:
                if self._balance >= required_amount:
                    self._balance -= required_amount
                    return True
                return False

        return self._collect_hardware(required_amount)

    def reset_credit(self) -> None:
        """Reset accumulated coins and credit to zero."""
        with self._lock:
            self._coin_count = 0
            self._total_credit = 0.0
            self._balance = 0.0

    def reset(self) -> None:
        """Synonym for reset_credit."""
        self.reset_credit()

    def close(self) -> None:
        """Release hardware resources cleanly."""
        with self._lock:
            self._closed = True
            self._listening = False
            if not self._simulation and self._pulse_pin is not None:
                try:
                    import RPi.GPIO as GPIO  # type: ignore[import]

                    GPIO.remove_event_detect(self._pulse_pin)
                    GPIO.cleanup(self._pulse_pin)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error closing coin acceptor pin %d: %s", self._pulse_pin, exc)

    # ------------------------------------------------------------------
    # Internal Hardware Methods
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pulse_pin is None:
            raise ValueError("pulse_pin must be set for physical coin acceptor")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            pud = GPIO.PUD_UP if self._pull_up else GPIO.PUD_DOWN
            edge = GPIO.FALLING if self._pull_up else GPIO.RISING
            GPIO.setup(self._pulse_pin, GPIO.IN, pull_up_down=pud)
            GPIO.add_event_detect(
                self._pulse_pin,
                edge,
                callback=self._on_pulse,
                bouncetime=self._bouncetime_ms,
            )
            self._listening = True
            logger.info("Coin acceptor initialised on GPIO pin %d", self._pulse_pin)
        except ImportError as exc:
            raise RuntimeError(
                "RPi.GPIO not available. Run on a Raspberry Pi or set simulation=True."
            ) from exc

    def _on_pulse(self, _channel: int) -> None:
        """Interrupt handler triggered by physical pulse signal."""
        with self._lock:
            if not self._listening or self._closed:
                return
            self._coin_count += 1
            self._total_credit += self.coin_value
            self._balance += self.coin_value
            callback = self._on_coin_callback
            total = self._coin_count

        if callback is not None:
            try:
                callback(total)
            except Exception as exc:
                logger.error("Coin pulse callback raised exception: %s", exc)

    def _collect_hardware(self, required_amount: float) -> bool:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._balance >= required_amount:
                    self._balance -= required_amount
                    return True
            time.sleep(0.05)
        logger.warning(
            "Coin acceptor timeout: required %.2f, collected %.2f",
            required_amount,
            self.balance,
        )
        return False


class MockCoinAcceptor(CoinAcceptor):
    """Convenience subclass for simulated coin acceptor."""

    def __init__(self, **kwargs):
        kwargs["simulation"] = True
        super().__init__(**kwargs)

