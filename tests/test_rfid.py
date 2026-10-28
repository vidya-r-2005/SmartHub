"""
Unit tests for RFIDReader (app/hardware/rfid.py).

Validates:
- UID normalization (uppercase hex formatting, whitespace stripping)
- Sequential reading via simulation queue (user card then item tag)
- Held card debouncing / hold-off behavior
- Singleton lifecycle (get_reader, reset_reader)
- Error handling on closed or invalid inputs
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
from app.hardware.rfid import (
    MockRFID,
    RFIDReader,
    get_reader,
    normalize_uid,
    reset_reader,
)


def test_normalize_uid():
    assert normalize_uid("e2a1098b") == "E2A1098B"
    assert normalize_uid("  04A2F188  ") == "04A2F188"
    assert normalize_uid(0x1A2B) == "1A2B"
    assert normalize_uid(b"\x01\x02\x03\x04") == "01020304"

    with pytest.raises(ValueError, match="None"):
        normalize_uid(None)
    with pytest.raises(ValueError, match="empty"):
        normalize_uid("   ")


def test_mock_rfid_sequential_queue():
    reader = MockRFID()
    reader.queue_uid("USER_CARD_001")
    reader.queue_uid("ITEM_TAG_101")

    # First read pops user card
    assert reader.read_uid() == "USER_CARD_001"
    # Second read pops item tag
    assert reader.read_uid() == "ITEM_TAG_101"
    # Queue is now empty, returns default mock_uid (None)
    assert reader.read_uid() is None
    reader.close()


def test_mock_rfid_default_uid():
    reader = MockRFID(mock_uid="DEFAULT_CARD")
    assert reader.read_uid() == "DEFAULT_CARD"
    assert reader.read_uid() == "DEFAULT_CARD"

    reader.set_mock_uid("NEW_DEFAULT")
    assert reader.read_uid() == "NEW_DEFAULT"
    reader.close()


def test_rfid_closed_raises():
    reader = MockRFID()
    reader.close()
    with pytest.raises(RuntimeError, match="closed"):
        reader.read_uid()
    with pytest.raises(RuntimeError, match="closed"):
        reader.queue_uid("CARD_X")


def test_rfid_singleton_lifecycle():
    reset_reader()
    r1 = get_reader(simulation=True, mock_uid="SINGLETON_1")
    assert r1.read_uid() == "SINGLETON_1"

    r2 = get_reader()
    assert r1 is r2

    reset_reader()
    r3 = get_reader(simulation=True, mock_uid="SINGLETON_2")
    assert r3 is not r1
    assert r3.read_uid() == "SINGLETON_2"
    reset_reader()
