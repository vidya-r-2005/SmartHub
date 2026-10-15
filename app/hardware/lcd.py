
"""SmartHub 16x2 I2C LCD interface."""


class MockLCD:
    """LCD substitute for Windows development and tests."""

    def __init__(self):
        self.lines = ["", ""]
        self.closed = False

    def show(self, line1="", line2=""):
        if self.closed:
            raise RuntimeError("LCD is closed")

        self.lines = [
            str(line1)[:16].ljust(16),
            str(line2)[:16].ljust(16),
        ]
        print(f"[LCD] {self.lines[0].strip()}")
        print(f"[LCD] {self.lines[1].strip()}")

    def clear(self):
        self.show("", "")

    def close(self):
        self.closed = True


class I2CLCD:
    """Physical 16x2 LCD using a PCF8574 I2C backpack."""

    def __init__(self, address=0x27):
        from RPLCD.i2c import CharLCD

        self._lcd = CharLCD(
            "PCF8574",
            address=address,
            port=1,
            cols=16,
            rows=2,
            auto_linebreaks=False,
        )

    def show(self, line1="", line2=""):
        self._lcd.cursor_pos = (0, 0)
        self._lcd.write_string(str(line1)[:16].ljust(16))
        self._lcd.cursor_pos = (1, 0)
        self._lcd.write_string(str(line2)[:16].ljust(16))

    def clear(self):
        self._lcd.clear()

    def close(self):
        self._lcd.close(clear=True)
