"""
Hardware abstraction: Coin acceptor.

In simulation mode the acceptor accumulates a configurable mock balance
instead of listening to physical pulse signals.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class CoinAcceptor:
    """Abstraction over a pulse-count coin acceptor.

    Each pulse from the acceptor represents one coin unit (configurable).

    Parameters
    ----------
    simulation : bool
        When True, software simulation is used.
    coin_value : float
        Value in currency units per pulse / coin.
    pulse_pin : int | None
        GPIO pin number (BCM) for the pulse signal.  Required when
        *simulation* is False.
    timeout_seconds : float
        How long to wait for payment before timing out.
    """

    def __init__(
        self,
        simulation: bool = True,
        coin_value: float = 5.0,
        pulse_pin: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ):
        self._simulation = simulation
        self._coin_value = coin_value
        self._pulse_pin = pulse_pin
        self._timeout = timeout_seconds
        self._balance: float = 0.0
        self._pulse_count: int = 0

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Simulation interface
    # ------------------------------------------------------------------

    def simulate_insert(self, coins: int = 1) -> None:
        """Add *coins* to the simulated balance (simulation mode only)."""
        if not self._simulation:
            raise RuntimeError("simulate_insert is only available in simulation mode")
        self._pulse_count += coins
        self._balance += coins * self._coin_value
        logger.debug(
            "Simulated coin insert: %d coins, balance=%.2f",
            coins,
            self._balance,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current accumulated balance."""
        if self._simulation:
            return self._balance
        return self._pulse_count * self._coin_value

    def collect(self, required_amount: float) -> bool:
        """Block until *required_amount* has been collected or timeout.

        Returns True if sufficient payment was received, False on timeout.
        """
        if required_amount < 0:
            raise ValueError("required_amount cannot be negative")

        if self._simulation:
            # In simulation mode balance is set externally
            return self._balance >= required_amount

        return self._collect_hardware(required_amount)

    def reset(self) -> None:
        """Reset the accumulated balance / pulse count."""
        self._balance = 0.0
        self._pulse_count = 0

    def close(self) -> None:
        """Release hardware resources."""
        if not self._simulation:
            try:
                import RPi.GPIO as GPIO  # type: ignore[import]

                GPIO.cleanup(self._pulse_pin)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing coin acceptor: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        if self._pulse_pin is None:
            raise ValueError("pulse_pin must be set for physical coin acceptor")
        try:
            import RPi.GPIO as GPIO  # type: ignore[import]

            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._pulse_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            GPIO.add_event_detect(
                self._pulse_pin,
                GPIO.RISING,
                callback=self._on_pulse,
                bouncetime=50,
            )
            logger.info("Coin acceptor hardware initialised on pin %d", self._pulse_pin)
        except ImportError:
            raise RuntimeError(
                "RPi.GPIO not available. "
                "Run on a Raspberry Pi or set simulation=True."
            )

    def _on_pulse(self, _channel: int) -> None:
        self._pulse_count += 1

    def _collect_hardware(self, required_amount: float) -> bool:
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._pulse_count * self._coin_value >= required_amount:
                return True
            time.sleep(0.05)
        logger.warning(
            "Coin acceptor timeout: collected %.2f of %.2f required",
            self._pulse_count * self._coin_value,
            required_amount,
        )
        return False

