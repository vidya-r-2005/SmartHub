"""
Unit tests for SmartHub hardware abstraction drivers.

Tests run in simulation/mock mode without requiring physical GPIO hardware.
Validates:
- Relay / MockRelay
- LEDs / MockLEDs
- Camera / MockCamera
- Buttons / MockButtons
- LCD / MockLCD
- LockController / MockLock (Servo)
- Buzzer / MockBuzzer
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
from app.hardware.buttons import BUTTON_PINS, Button, MockButtons
from app.hardware.buzzer import DEFAULT_BUZZER_PIN, Buzzer, MockBuzzer
from app.hardware.camera import Camera, MockCamera
from app.hardware.charging import SECONDS_PER_COIN, ChargingController
from app.hardware.lcd import LCD, MockLCD
from app.hardware.leds import DEFAULT_LED_PINS, LEDs, MockLEDs
from app.hardware.lock import DEFAULT_SERVO_PIN, LockController, MockLock
from app.hardware.relay import DEFAULT_RELAY_PIN, MockRelay, Relay


# ---------------------------------------------------------------------------
# Relay Tests
# ---------------------------------------------------------------------------


def test_relay_initial_state_is_off():
    relay = MockRelay(pin=DEFAULT_RELAY_PIN)
    assert not relay.is_on
    assert relay.turn_on() is True
    assert relay.is_on is True
    assert relay.turn_off() is True
    assert relay.is_on is False
    relay.close()


def test_relay_close_turns_off():
    relay = MockRelay()
    relay.turn_on()
    assert relay.is_on is True
    relay.close()
    assert relay.is_on is False
    assert relay.turn_on() is False


# ---------------------------------------------------------------------------
# LEDs Tests
# ---------------------------------------------------------------------------


def test_leds_status_mapping():
    leds = MockLEDs()
    assert leds.current_status == "ALL_OFF"
    assert not leds.is_on("GREEN")
    assert not leds.is_on("RED")
    assert not leds.is_on("BLUE")

    leds.set_status("READY")
    assert leds.is_on("GREEN")
    assert not leds.is_on("RED")
    assert not leds.is_on("BLUE")

    leds.set_status("CHARGING")
    assert not leds.is_on("GREEN")
    assert not leds.is_on("RED")
    assert leds.is_on("BLUE")

    leds.set_status("ERROR")
    assert not leds.is_on("GREEN")
    assert leds.is_on("RED")
    assert not leds.is_on("BLUE")

    leds.all_off()
    assert not leds.is_on("GREEN")
    assert not leds.is_on("RED")
    assert not leds.is_on("BLUE")
    leds.close()


# ---------------------------------------------------------------------------
# Camera Tests
# ---------------------------------------------------------------------------


def test_camera_simulation_capture():
    camera = MockCamera()
    data = camera.capture()
    assert isinstance(data, bytes)
    assert len(data) > 0
    # First 2 bytes of valid JPEG are 0xFF, 0xD8
    assert data[:2] == b"\xff\xd8"
    camera.close()


def test_camera_closed_raises():
    camera = MockCamera()
    camera.close()
    with pytest.raises(RuntimeError, match="closed"):
        camera.capture()


# ---------------------------------------------------------------------------
# Buttons Tests
# ---------------------------------------------------------------------------


def test_mock_buttons_query_and_press():
    buttons = MockButtons()
    for name in ("CHARGE", "STORE", "BORROW", "BACK_CONFIRM"):
        assert buttons.is_pressed(name) is False

    buttons.press("CHARGE")
    assert buttons.is_pressed("CHARGE") is True
    assert buttons.is_pressed("STORE") is False

    buttons.release("CHARGE")
    assert buttons.is_pressed("CHARGE") is False

    with pytest.raises(ValueError, match="Unknown button"):
        buttons.is_pressed("INVALID_BTN")

    buttons.close()


def test_mock_buttons_callbacks():
    buttons = MockButtons()
    called = []
    buttons.on_press("BORROW", lambda: called.append("borrowed"))
    buttons.press("BORROW")
    assert called == ["borrowed"]
    buttons.close()


def test_legacy_single_button():
    btn = Button(simulation=True)
    assert btn.press_count == 0
    btn.simulate_press()
    assert btn.press_count == 1
    btn.close()


# ---------------------------------------------------------------------------
# LCD Tests
# ---------------------------------------------------------------------------


def test_lcd_show_and_truncate():
    lcd = MockLCD(cols=16, rows=2)
    lcd.show("Welcome to", "SmartHub")
    assert lcd.last_lines == ["Welcome to", "SmartHub"]

    long_msg = "This is a very long line that exceeds sixteen chars"
    lcd.show(long_msg, "Line 2")
    assert lcd.last_lines[0] == long_msg[:16]
    assert len(lcd.last_lines[0]) == 16

    lcd.clear()
    assert lcd.last_lines == ["", ""]
    lcd.close()


# ---------------------------------------------------------------------------
# Lock / Servo Tests
# ---------------------------------------------------------------------------


def test_servo_lock_unlock():
    lock = MockLock()
    assert lock.is_locked is True

    assert lock.unlock() is True
    assert lock.is_locked is False

    assert lock.lock() is True
    assert lock.is_locked is True
    lock.close()


def test_servo_lock_injected_failure():
    failing_lock = MockLock(fail_result=False)
    assert failing_lock.unlock() is False
    assert failing_lock.lock() is False
    failing_lock.close()


# ---------------------------------------------------------------------------
# Buzzer Tests
# ---------------------------------------------------------------------------


def test_buzzer_methods():
    buzzer = MockBuzzer()
    # In simulation mode, sound patterns are logged safely without errors
    buzzer.beep(duration=0.1, count=1, blocking=True)
    buzzer.success()
    buzzer.error()
    buzzer.success_tone()
    buzzer.error_tone()
    buzzer.close()


# ---------------------------------------------------------------------------
# Charging Controller Session & Countdown Tests
# ---------------------------------------------------------------------------


def test_charging_controller_timed_session():
    current_time = 1000.0
    fake_clock = lambda: current_time

    ctrl = ChargingController(simulation=True, clock=fake_clock, seconds_per_credit=300)
    assert not ctrl.is_charging
    assert ctrl.seconds_remaining() == 0

    # Start 300s session (1 coin)
    ctrl.start(300)
    assert ctrl.is_charging
    assert ctrl.seconds_remaining() == 300

    # Advance clock by 100 seconds
    current_time = 1100.0
    assert ctrl.is_charging
    assert ctrl.seconds_remaining() == 200

    # Advance clock past deadline
    current_time = 1301.0
    assert ctrl.update() is True
    assert not ctrl.is_charging
    assert ctrl.seconds_remaining() == 0
    ctrl.close()
