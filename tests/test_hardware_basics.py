
from app.hardware.buttons import MockButtons
from app.hardware.lcd import MockLCD


def test_mock_buttons():
    buttons = MockButtons()

    assert buttons.is_pressed("CHARGE") is False

    buttons.press("CHARGE")
    assert buttons.is_pressed("CHARGE") is True

    buttons.release("CHARGE")
    assert buttons.is_pressed("CHARGE") is False

    buttons.close()


def test_mock_lcd():
    lcd = MockLCD()
    lcd.show("SmartHub", "Select option")

    assert lcd.lines[0].strip() == "SmartHub"
    assert lcd.lines[1].strip() == "Select option"

    lcd.clear()
    assert lcd.lines[0].strip() == ""
    lcd.close()
