"""
Hardware abstraction: LCD display (I2C character LCD).

In simulation mode messages are written to the logger instead of a
physical display.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_LCD_COLS = 16
_LCD_ROWS = 2


class LCD:
    """16x2 character LCD accessed via I2C.

    Parameters
    ----------
    simulation : bool
        When True, messages are only logged, not sent to hardware.
    i2c_address : int
        I2C address of the PCF8574 backpack (commonly 0x27 or 0x3F).
    i2c_bus : int
        I2C bus number (1 on Raspberry Pi).
    cols : int
        Number of display columns.
    rows : int
        Number of display rows.
    """

    def __init__(
        self,
        simulation: bool = True,
        i2c_address: int = 0x27,
        i2c_bus: int = 1,
        cols: int = _LCD_COLS,
        rows: int = _LCD_ROWS,
    ):
        self._simulation = simulation
        self._address = i2c_address
        self._bus = i2c_bus
        self._cols = cols
        self._rows = rows
        self._display = None
        self._last_lines: list[str] = ["", ""]

        if not simulation:
            self._init_hardware()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self, line1: str = "", line2: str = "") -> None:
        """Display up to two lines of text, truncated to fit.

        Parameters
        ----------
        line1 : str
            Text for the first row.
        line2 : str
            Text for the second row.
        """
        l1 = str(line1)[: self._cols]
        l2 = str(line2)[: self._cols]
        self._last_lines = [l1, l2]

        if self._simulation:
            logger.debug("[LCD] %-*s | %-*s", self._cols, l1, self._cols, l2)
            return

        self._write_hardware(l1, l2)

    def clear(self) -> None:
        """Clear the display."""
        self.show("", "")

    @property
    def last_lines(self) -> list[str]:
        """The most recently shown lines (useful for tests)."""
        return list(self._last_lines)

    def close(self) -> None:
        """Release hardware resources."""
        if not self._simulation and self._display is not None:
            try:
                self._display.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Error closing LCD: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_hardware(self) -> None:
        try:
            from RPLCD.i2c import CharLCD  # type: ignore[import]

            self._display = CharLCD(
                i2c_expander="PCF8574",
                address=self._address,
                port=self._bus,
                cols=self._cols,
                rows=self._rows,
                dotsize=8,
            )
            logger.info("LCD initialised at I2C address 0x%02X", self._address)
        except ImportError:
            raise RuntimeError(
                "RPLCD library not available. "
                "Install RPLCD or set simulation=True."
            )

    def _write_hardware(self, line1: str, line2: str) -> None:
        try:
            self._display.clear()
            self._display.cursor_pos = (0, 0)
            self._display.write_string(line1.ljust(self._cols))
            if self._rows > 1:
                self._display.cursor_pos = (1, 0)
                self._display.write_string(line2.ljust(self._cols))
        except Exception as exc:
            logger.error("Failed to write to LCD: %s", exc)

