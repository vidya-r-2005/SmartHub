"""
SmartHub hardware abstraction package.

Exposes dual-mode (physical GPIO and software simulation) interfaces for:
  - buttons:  Push-button input (CHARGE, STORE, BORROW, BACK_CONFIRM)
  - buzzer:   Audio feedback
  - camera:   Visual peripheral inspection
  - charging: Timed charging session controller
  - coin:     Pulse-count ₹5 coin acceptor
  - lcd:      16x2 I2C character display
  - leds:     Green, Red, Blue status indicators
  - lock:     STORE compartment PWM servo
  - relay:    Charging output power switch
  - rfid:     MFRC522 contactless card reader
"""

from app.hardware.buttons import BUTTON_PINS, Button, GPIOButtons, MockButtons
from app.hardware.buzzer import DEFAULT_BUZZER_PIN, Buzzer, MockBuzzer
from app.hardware.camera import Camera, MockCamera
from app.hardware.charging import SECONDS_PER_COIN, ChargingController, ChargingState
from app.hardware.coin import DEFAULT_COIN_PIN, DEFAULT_COIN_VALUE, CoinAcceptor
from app.hardware.lcd import LCD, MockLCD
from app.hardware.leds import DEFAULT_LED_PINS, LEDs, MockLEDs
from app.hardware.lock import DEFAULT_SERVO_PIN, LockController, MockLock
from app.hardware.relay import DEFAULT_RELAY_PIN, MockRelay, Relay
from app.hardware.rfid import RFIDReader, MockRFID, get_reader, normalize_uid, reset_reader

__all__ = [
    "BUTTON_PINS",
    "Button",
    "GPIOButtons",
    "MockButtons",
    "DEFAULT_BUZZER_PIN",
    "Buzzer",
    "MockBuzzer",
    "Camera",
    "MockCamera",
    "ChargingController",
    "ChargingState",
    "SECONDS_PER_COIN",
    "DEFAULT_COIN_PIN",
    "DEFAULT_COIN_VALUE",
    "CoinAcceptor",
    "LCD",
    "MockLCD",
    "DEFAULT_LED_PINS",
    "LEDs",
    "MockLEDs",
    "DEFAULT_SERVO_PIN",
    "LockController",
    "MockLock",
    "DEFAULT_RELAY_PIN",
    "Relay",
    "MockRelay",
    "RFIDReader",
    "MockRFID",
    "normalize_uid",
    "get_reader",
    "reset_reader",
]
