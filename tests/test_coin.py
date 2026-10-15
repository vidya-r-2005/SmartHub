
import pytest

from app.hardware.coin import CoinAcceptor


def test_single_coin_adds_credit():
    coin = CoinAcceptor(credit_per_coin=5)

    assert coin.add_coin() == 5
    assert coin.get_credit() == 5
    assert coin.coin_count == 1


def test_multiple_identical_coins_accumulate_credit():
    coin = CoinAcceptor(credit_per_coin=5)

    coin.add_coin()
    coin.add_coin()

    assert coin.coin_count == 2
    assert coin.get_credit() == 10


def test_reset_credit():
    coin = CoinAcceptor(credit_per_coin=5)
    coin.add_coin()

    coin.reset_credit()

    assert coin.coin_count == 0
    assert coin.get_credit() == 0


def test_invalid_coin_count_is_rejected():
    coin = CoinAcceptor()

    with pytest.raises(ValueError):
        coin.add_coin(0)
