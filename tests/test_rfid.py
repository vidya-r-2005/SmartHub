
import pytest

from app.hardware.rfid import MockRFID


def test_mock_rfid_returns_queued_uid():
    reader = MockRFID()
    reader.queue_uid(" ab:12:cd ")

    assert reader.read_uid() == "AB:12:CD"
    assert reader.read_uid() is None


def test_mock_rfid_supports_multiple_scans():
    reader = MockRFID()
    reader.queue_uid("USER001")
    reader.queue_uid("ITEM002")

    assert reader.read_uid() == "USER001"
    assert reader.read_uid() == "ITEM002"


def test_mock_rfid_rejects_empty_uid():
    reader = MockRFID()

    with pytest.raises(ValueError):
        reader.queue_uid("   ")


def test_closed_reader_cannot_be_scanned():
    reader = MockRFID()
    reader.close()

    with pytest.raises(RuntimeError):
        reader.read_uid()
