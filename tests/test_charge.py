
import pytest

from app.hardware.charging import ChargingController, MockRelay


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_charging_starts_and_stops_at_deadline():
    clock = FakeClock()
    relay = MockRelay()
    controller = ChargingController(
        relay, seconds_per_credit=10, clock=clock
    )

    controller.start(2)
    assert controller.is_charging()
    assert relay.is_on
    assert controller.seconds_remaining() == 20

    clock.advance(19)
    assert controller.update() is False
    assert relay.is_on

    clock.advance(1)
    assert controller.update() is True
    assert not relay.is_on
    assert not controller.is_charging()


def test_invalid_credit_does_not_start_charging():
    relay = MockRelay()
    controller = ChargingController(relay)

    with pytest.raises(ValueError):
        controller.start(0)

    assert not relay.is_on


def test_manual_stop_disables_relay():
    relay = MockRelay()
    controller = ChargingController(relay)

    controller.start(1)
    controller.stop()

    assert not relay.is_on
    assert not controller.is_charging()
