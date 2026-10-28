"""
Tests for the BORROW service (app/services/borrow.py).

Vision module is mocked at the module level to avoid real model loading.
All tests use isolated temporary databases.
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

from unittest.mock import patch

from app.services.borrow import (
    BorrowServiceError,
    VerificationMismatchError,
    list_available_items,
    start_borrow,
    return_item,
    get_borrow_records,
)
from app.database import db as database


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user(rfid: str, tmp_db) -> int:
    uid = database.create_user(rfid, db_path=tmp_db)
    return uid


def _make_item(rfid: str, item_type: str, tmp_db) -> int:
    iid = database.create_item(rfid, item_type, db_path=tmp_db)
    return iid


def _vision_match(expected_category: str):
    """Return a vision result that matches the expected category."""
    return {
        "available": True,
        "match": True,
        "category": expected_category,
        "confidence": 0.95,
        "expected": expected_category,
        "error": None,
    }


def _vision_mismatch(expected_category: str, got: str):
    """Return a vision result indicating a mismatch."""
    return {
        "available": True,
        "match": False,
        "category": got,
        "confidence": 0.80,
        "expected": expected_category,
        "error": None,
    }


def _vision_unavailable(expected_category: str):
    return {
        "available": False,
        "match": None,
        "category": None,
        "confidence": None,
        "expected": expected_category,
        "error": "Vision model not configured",
    }


# ---------------------------------------------------------------------------
# list_available_items
# ---------------------------------------------------------------------------


def test_list_available_items_empty(tmp_db):
    items = list_available_items(db_path=tmp_db)
    assert items == []


def test_list_available_items_returns_available(tmp_db):
    _make_item("AVAIL_A", "power_bank", tmp_db)
    _make_item("AVAIL_B", "cable", tmp_db)
    items = list_available_items(db_path=tmp_db)
    assert len(items) == 2


def test_list_available_items_excludes_borrowed(tmp_db):
    _make_user("BW_USER1", tmp_db)
    _make_item("BORROW_EXCL", "power_bank", tmp_db)
    start_borrow(rfid_uid="BW_USER1", item_rfid="BORROW_EXCL", db_path=tmp_db)
    items = list_available_items(db_path=tmp_db)
    assert not any(i["rfid_uid"] == "BORROW_EXCL" for i in items)


# ---------------------------------------------------------------------------
# start_borrow
# ---------------------------------------------------------------------------


def test_start_borrow_succeeds(tmp_db):
    _make_user("BU_1", tmp_db)
    _make_item("BI_1", "power_bank", tmp_db)
    result = start_borrow(rfid_uid="BU_1", item_rfid="BI_1", db_path=tmp_db)
    assert "session_id" in result
    assert "borrow_record_id" in result
    assert result["item_type"] == "power_bank"


def test_start_borrow_missing_user_rfid_raises(tmp_db):
    _make_item("BI_2", "power_bank", tmp_db)
    with pytest.raises(BorrowServiceError, match="RFID"):
        start_borrow(rfid_uid="", item_rfid="BI_2", db_path=tmp_db)


def test_start_borrow_missing_item_rfid_raises(tmp_db):
    _make_user("BU_2", tmp_db)
    with pytest.raises(BorrowServiceError, match="Item RFID"):
        start_borrow(rfid_uid="BU_2", item_rfid="", db_path=tmp_db)


def test_start_borrow_nonexistent_item_raises(tmp_db):
    _make_user("BU_3", tmp_db)
    with pytest.raises(BorrowServiceError, match="No item"):
        start_borrow(rfid_uid="BU_3", item_rfid="NOSUCHITEM", db_path=tmp_db)


def test_start_borrow_unavailable_item_raises(tmp_db):
    _make_user("BU_4", tmp_db)
    _make_item("BI_4", "power_bank", tmp_db)
    # First borrow
    start_borrow(rfid_uid="BU_4", item_rfid="BI_4", db_path=tmp_db)
    # Second borrow of same item
    with pytest.raises(BorrowServiceError, match="not available"):
        start_borrow(rfid_uid="BU_4", item_rfid="BI_4", db_path=tmp_db)


def test_start_borrow_auto_registers_user(tmp_db):
    _make_item("BI_5", "power_bank", tmp_db)
    result = start_borrow(rfid_uid="NEW_BORROW_USER", item_rfid="BI_5", db_path=tmp_db)
    user = database.get_user_by_rfid("NEW_BORROW_USER", db_path=tmp_db)
    assert user is not None
    assert result["user_id"] == user["id"]


# ---------------------------------------------------------------------------
# Vision integration
# ---------------------------------------------------------------------------


def test_start_borrow_with_vision_match(tmp_db):
    _make_user("VIS_U1", tmp_db)
    _make_item("VIS_I1", "power_bank", tmp_db)

    fake_vision = _vision_match("power_bank")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        result = start_borrow(
            rfid_uid="VIS_U1",
            item_rfid="VIS_I1",
            image_data=b"fake_image_bytes",
            db_path=tmp_db,
        )
    assert result["vision_result"]["match"] is True


def test_start_borrow_vision_mismatch_raises(tmp_db):
    _make_user("VIS_U2", tmp_db)
    _make_item("VIS_I2", "power_bank", tmp_db)

    fake_vision = _vision_mismatch("power_bank", "cable")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        with pytest.raises(VerificationMismatchError, match="mismatch"):
            start_borrow(
                rfid_uid="VIS_U2",
                item_rfid="VIS_I2",
                image_data=b"fake_image_bytes",
                db_path=tmp_db,
            )


def test_start_borrow_vision_mismatch_does_not_borrow(tmp_db):
    """A vision mismatch must not create a borrow record."""
    _make_user("VIS_U3", tmp_db)
    _make_item("VIS_I3", "power_bank", tmp_db)

    fake_vision = _vision_mismatch("power_bank", "umbrella")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        try:
            start_borrow(
                rfid_uid="VIS_U3",
                item_rfid="VIS_I3",
                image_data=b"x",
                db_path=tmp_db,
            )
        except VerificationMismatchError:
            pass

    # Item must still be available
    items = list_available_items(db_path=tmp_db)
    assert any(i["rfid_uid"] == "VIS_I3" for i in items)


def test_start_borrow_requested_vision_unavailable_refuses_borrow(tmp_db):
    """A requested camera check must fail closed when no model is available."""
    _make_user("VIS_U4", tmp_db)
    _make_item("VIS_I4", "power_bank", tmp_db)

    fake_vision = _vision_unavailable("power_bank")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        with pytest.raises(BorrowServiceError, match="unavailable"):
            start_borrow(
                rfid_uid="VIS_U4",
                item_rfid="VIS_I4",
                image_data=b"x",
                db_path=tmp_db,
            )
    # Should succeed — unavailable model is not a mismatch
    assert any(i["rfid_uid"] == "VIS_I4" for i in list_available_items(db_path=tmp_db))


# ---------------------------------------------------------------------------
# return_item
# ---------------------------------------------------------------------------


def test_return_item_succeeds(tmp_db):
    _make_user("RET_U1", tmp_db)
    _make_item("RET_I1", "power_bank", tmp_db)
    start_borrow(rfid_uid="RET_U1", item_rfid="RET_I1", db_path=tmp_db)

    result = return_item(item_rfid="RET_I1", db_path=tmp_db)
    assert result["status"] == "returned"


def test_return_item_makes_available_again(tmp_db):
    _make_user("RET_U2", tmp_db)
    _make_item("RET_I2", "cable", tmp_db)
    start_borrow(rfid_uid="RET_U2", item_rfid="RET_I2", db_path=tmp_db)
    return_item(item_rfid="RET_I2", db_path=tmp_db)

    items = list_available_items(db_path=tmp_db)
    assert any(i["rfid_uid"] == "RET_I2" for i in items)


def test_return_nonexistent_item_raises(tmp_db):
    with pytest.raises(BorrowServiceError, match="No item"):
        return_item(item_rfid="NOSUCHITEM", db_path=tmp_db)


def test_return_item_not_borrowed_raises(tmp_db):
    _make_item("NOTBORROWED", "power_bank", tmp_db)
    with pytest.raises(BorrowServiceError, match="not currently borrowed"):
        return_item(item_rfid="NOTBORROWED", db_path=tmp_db)


def test_return_missing_rfid_raises(tmp_db):
    with pytest.raises(BorrowServiceError, match="RFID"):
        return_item(item_rfid="", db_path=tmp_db)


def test_return_vision_mismatch_raises(tmp_db):
    _make_user("RVIS_U1", tmp_db)
    _make_item("RVIS_I1", "power_bank", tmp_db)
    start_borrow(rfid_uid="RVIS_U1", item_rfid="RVIS_I1", db_path=tmp_db)

    fake_vision = _vision_mismatch("power_bank", "umbrella")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        with pytest.raises(VerificationMismatchError):
            return_item(item_rfid="RVIS_I1", image_data=b"x", db_path=tmp_db)


# ---------------------------------------------------------------------------
# get_borrow_records
# ---------------------------------------------------------------------------


def test_get_borrow_records_returns_list(tmp_db):
    _make_user("REC_U1", tmp_db)
    _make_item("REC_I1", "power_bank", tmp_db)
    start_borrow(rfid_uid="REC_U1", item_rfid="REC_I1", db_path=tmp_db)

    records = get_borrow_records(db_path=tmp_db)
    assert len(records) >= 1


# ---------------------------------------------------------------------------
# Business Rule & Specification Tests
# ---------------------------------------------------------------------------


def test_start_borrow_daily_limit_enforced(tmp_db):
    _make_user("LIMIT_USER", tmp_db)
    _make_item("ITEM_L1", "cable", tmp_db)
    _make_item("ITEM_L2", "charger", tmp_db)
    _make_item("ITEM_L3", "power_bank", tmp_db)

    # 1st borrow succeeds
    res1 = start_borrow("LIMIT_USER", "ITEM_L1", db_path=tmp_db)
    assert res1["borrow_record_id"] is not None

    # 2nd borrow succeeds
    res2 = start_borrow("LIMIT_USER", "ITEM_L2", db_path=tmp_db)
    assert res2["borrow_record_id"] is not None

    # 3rd borrow on same day MUST fail
    with pytest.raises(BorrowServiceError, match="Daily borrow limit"):
        start_borrow("LIMIT_USER", "ITEM_L3", db_path=tmp_db)


def test_start_borrow_rejects_unregistered_when_disallowed(tmp_db):
    _make_item("UNREG_ITEM", "cable", tmp_db)
    with pytest.raises(BorrowServiceError, match="not registered"):
        start_borrow(
            rfid_uid="UNREGISTERED_CARD",
            item_rfid="UNREG_ITEM",
            allow_auto_register=False,
            db_path=tmp_db,
        )


def test_return_item_with_unregistered_user_rfid_raises(tmp_db):
    _make_user("VALID_BORROWER", tmp_db)
    _make_item("RET_USER_ITEM", "power_bank", tmp_db)
    start_borrow("VALID_BORROWER", "RET_USER_ITEM", db_path=tmp_db)

    with pytest.raises(BorrowServiceError, match="not registered"):
        return_item("RET_USER_ITEM", user_rfid="UNKNOWN_RETURNER", db_path=tmp_db)


def test_vision_result_recorded_in_database(tmp_db):
    _make_user("VR_USER", tmp_db)
    _make_item("VR_ITEM", "power_bank", tmp_db)

    fake_vision = _vision_match("power_bank")
    with patch("app.services.borrow._verify_vision", return_value=fake_vision):
        start_borrow(
            rfid_uid="VR_USER",
            item_rfid="VR_ITEM",
            image_data=b"dummy_data",
            db_path=tmp_db,
        )

    rows = database.execute_query(
        "SELECT * FROM vision_results ORDER BY id DESC LIMIT 1", (), tmp_db
    )
    assert len(rows) == 1
    assert rows[0]["expected_item"] == "power_bank"
    assert rows[0]["predicted_class"] == "power_bank"
    assert rows[0]["match"] == 1


