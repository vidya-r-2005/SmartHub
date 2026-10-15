
"""SmartHub RFID reader interface.

The same reader can scan users and items sequentially.
RC522 hardware support is intended for Raspberry Pi.
"""


class MockRFID:
    """Simulated RFID reader for Windows development and tests."""

    def __init__(self):
        self._queued_uids = []
        self.closed = False

    def queue_uid(self, uid):
        """Queue a UID to be returned by the next scan."""
        if uid is not None:
            uid = str(uid).strip().upper()
            if not uid:
                raise ValueError("UID cannot be empty")
            self._queued_uids.append(uid)

    def read_uid(self):
        """Return the next simulated UID, or None if no scan is queued."""
        if self.closed:
            raise RuntimeError("RFID reader is closed")
        return self._queued_uids.pop(0) if self._queued_uids else None

    def close(self):
        self.closed = True
        self._queued_uids.clear()


class RC522Reader:
    """RC522 reader adapter.

    Requires a compatible RC522 Python library and SPI enabled on the Pi.
    """

    def __init__(self):
        try:
            from mfrc522 import SimpleMFRC522
        except ImportError as exc:
            raise RuntimeError(
                "Install a compatible MFRC522 library on the Raspberry Pi"
            ) from exc

        self._reader = SimpleMFRC522()

    def read_uid(self):
        """Read a tag; this call may block until a tag is presented."""
        uid, _text = self._reader.read()
        return str(uid) if uid is not None else None

    def close(self):
        # Library cleanup behavior varies by package version.
        # Do not assume every version exposes the same cleanup method.
        pass
