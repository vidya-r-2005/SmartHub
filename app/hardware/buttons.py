
"""SmartHub physical button interface.

Buttons connect between their GPIO pin and GND.
Uses mock buttons on Windows and gpiozero on Raspberry Pi.
"""


class MockButtons:
    """Button interface for development without physical hardware."""

    def __init__(self):
        self._pressed = set()

    def is_pressed(self, name):
        return name in self._pressed

    def press(self, name):
        """Simulate a button press during testing."""
        if name not in self._valid_names():
            raise ValueError(f"Unknown button: {name}")
        self._pressed.add(name)

    def release(self, name):
        self._pressed.discard(name)

    @staticmethod
    def _valid_names():
        return {"CHARGE", "STORE", "BORROW", "BACK_CONFIRM"}

    def close(self):
        self._pressed.clear()


class GPIOButtons:
    """Physical buttons using gpiozero on Raspberry Pi."""

    PINS = {
        "CHARGE": 5,
        "STORE": 6,
        "BORROW": 13,
        "BACK_CONFIRM": 19,
    }

    def __init__(self):
        from gpiozero import Button

        self._buttons = {
            name: Button(pin, pull_up=True, bounce_time=0.08)
            for name, pin in self.PINS.items()
        }

    def is_pressed(self, name):
        if name not in self._buttons:
            raise ValueError(f"Unknown button: {name}")
        return self._buttons[name].is_pressed

    def close(self):
        for button in self._buttons.values():
            button.close()
