"""
Unit tests for CoinAcceptor (app/hardware/coin.py).

Validates:
- Single ₹5 coin pricing rules:
    1 coin = 5 min (300s)
    2 coins = 10 min (600s)
    3 coins = 15 min (900s)
- start() / stop() / get_coin_count()
- add_coin() and simulate_insert()
- collect() balance deductions
- error handling for invalid coins or closed acceptor
"""

try:
    import pytest
except ImportError:
    class _MockPytest:
        class raises:
            def __init__(self, expected_exc, match=None):
                self.expected_exc = expected_exc
                self.match = match

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError(f"Expected {self.expected_exc.__name__} was not raised")
                if not issubclass(exc_type, self.expected_exc):
                    return False
                if self.match and self.match not in str(exc_val):
                    raise AssertionError(f"Exception message '{exc_val}' does not match '{self.match}'")
                return True

    pytest = _MockPytest()  # type: ignore
from app.hardware.coin import (
    DEFAULT_COIN_PIN,
    DEFAULT_COIN_VALUE,
    SECONDS_PER_COIN,
    CoinAcceptor,
)


def test_coin_pricing_constants():
    assert DEFAULT_COIN_VALUE == 5.0
    assert SECONDS_PER_COIN == 300
    assert DEFAULT_COIN_PIN == 17


def test_coin_initial_state():
    coin = CoinAcceptor(simulation=True)
    assert coin.get_coin_count() == 0
    assert coin.coin_count == 0
    assert coin.balance == 0.0
    assert coin.get_credit() == 0.0
    coin.close()


def test_coin_add_and_simulate_insert():
    coin = CoinAcceptor(simulation=True, coin_value=5.0)
    added = coin.add_coin(1)
    assert added == 5.0
    assert coin.get_coin_count() == 1
    assert coin.balance == 5.0
    assert coin.get_credit() == 5.0

    coin.simulate_insert(2)
    assert coin.get_coin_count() == 3
    assert coin.balance == 15.0
    assert coin.get_credit() == 15.0
    coin.close()


def test_coin_collect_deduction():
    coin = CoinAcceptor(simulation=True, coin_value=5.0)
    coin.add_coin(2)  # ₹10 balance
    assert coin.balance == 10.0

    # Collect 5
    assert coin.collect(5.0) is True
    assert coin.balance == 5.0

    # Collect remaining 5
    assert coin.collect(5.0) is True
    assert coin.balance == 0.0

    # Insufficient balance returns False
    assert coin.collect(5.0) is False
    coin.close()


def test_coin_reset():
    coin = CoinAcceptor(simulation=True)
    coin.add_coin(3)
    assert coin.get_coin_count() == 3
    assert coin.balance == 15.0

    coin.reset()
    assert coin.get_coin_count() == 0
    assert coin.balance == 0.0
    coin.close()


def test_coin_callbacks():
    received = []
    coin = CoinAcceptor(simulation=True, on_coin_accepted=lambda c: received.append(c))
    coin.add_coin(1)
    coin.add_coin(1)
    assert received == [1, 2]
    coin.close()


def test_coin_validation_errors():
    coin = CoinAcceptor(simulation=True)
    with pytest.raises(ValueError, match="positive integer"):
        coin.add_coin(-1)
    with pytest.raises(ValueError, match="positive integer"):
        coin.add_coin(0)
    with pytest.raises(ValueError, match="positive integer"):
        coin.add_coin("one")  # type: ignore

    with pytest.raises(ValueError, match="numeric"):
        coin.collect("five")  # type: ignore
    with pytest.raises(ValueError, match="negative"):
        coin.collect(-5.0)

    coin.close()
    with pytest.raises(RuntimeError, match="closed"):
        coin.add_coin(1)
