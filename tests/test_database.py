"""
Tests for app/database/db.py

All tests use the ``tmp_db`` fixture so the production database is never
touched and tests run independently of each other.
"""

import pytest

from app.database import db as database


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_initialize_creates_tables(tmp_db):
    """Initialised database must contain all required tables."""
    rows = database.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        db_path=tmp_db,
    )
    table_names = {row["name"] for row in rows}
    expected = {"users", "services", "sessions", "transactions", "items",
                "borrow_records", "system_events"}
    assert expected.issubset(table_names)


def test_initialize_seeds_services(tmp_db):
    """Services table must contain CHARGE, STORE, and BORROW after init."""
    rows = database.execute_query("SELECT name FROM services ORDER BY name", db_path=tmp_db)
    names = {row["name"] for row in rows}
    assert names == {"CHARGE", "STORE", "BORROW"}


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


def test_create_user_returns_id(tmp_db):
    user_id = database.create_user("AABBCCDD", db_path=tmp_db)
    assert isinstance(user_id, int)
    assert user_id > 0


def test_get_user_by_rfid_returns_user(tmp_db):
    database.create_user("AABBCCDD", db_path=tmp_db)
    user = database.get_user_by_rfid("AABBCCDD", db_path=tmp_db)
    assert user is not None
    assert user["rfid_uid"] == "AABBCCDD"


def test_get_user_by_rfid_unknown_returns_none(tmp_db):
    assert database.get_user_by_rfid("UNKNOWN", db_path=tmp_db) is None


def test_duplicate_rfid_returns_none(tmp_db):
    database.create_user("DUPLICATE", db_path=tmp_db)
    result = database.create_user("DUPLICATE", db_path=tmp_db)
    assert result is None


def test_create_user_empty_rfid_raises(tmp_db):
    with pytest.raises(ValueError, match="RFID"):
        database.create_user("", db_path=tmp_db)


def test_create_user_whitespace_rfid_raises(tmp_db):
    with pytest.raises(ValueError, match="RFID"):
        database.create_user("   ", db_path=tmp_db)


def test_get_user_by_id_returns_user(tmp_db):
    uid = database.create_user("ID001", db_path=tmp_db)
    user = database.get_user_by_id(uid, db_path=tmp_db)
    assert user is not None
    assert user["rfid_uid"] == "ID001"


def test_get_user_by_id_unknown_returns_none(tmp_db):
    assert database.get_user_by_id(99999, db_path=tmp_db) is None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def test_create_session_returns_id(tmp_db):
    user_id = database.create_user("SES001", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    assert isinstance(session_id, int)
    assert session_id > 0


def test_create_session_unknown_service_raises(tmp_db):
    user_id = database.create_user("SES002", db_path=tmp_db)
    with pytest.raises(ValueError, match="Unknown service"):
        database.create_session(user_id, "NONEXISTENT", db_path=tmp_db)


def test_end_session_succeeds(tmp_db):
    user_id = database.create_user("SES003", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    result = database.end_session(session_id, "completed", db_path=tmp_db)
    assert result is True


def test_end_session_already_ended_returns_false(tmp_db):
    user_id = database.create_user("SES004", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    database.end_session(session_id, "completed", db_path=tmp_db)
    # Second call — session is no longer active
    result = database.end_session(session_id, "completed", db_path=tmp_db)
    assert result is False


def test_end_session_invalid_status_raises(tmp_db):
    user_id = database.create_user("SES005", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    with pytest.raises(ValueError, match="Invalid final session status"):
        database.end_session(session_id, "invalid_status", db_path=tmp_db)


def test_get_session_returns_row(tmp_db):
    user_id = database.create_user("SES006", db_path=tmp_db)
    session_id = database.create_session(user_id, "STORE", db_path=tmp_db)
    session = database.get_session(session_id, db_path=tmp_db)
    assert session is not None
    assert session["service_name"] == "STORE"
    assert session["status"] == "active"


def test_get_sessions_by_service(tmp_db):
    user_id = database.create_user("SES007", db_path=tmp_db)
    database.create_session(user_id, "CHARGE", db_path=tmp_db)
    database.create_session(user_id, "CHARGE", db_path=tmp_db)
    database.create_session(user_id, "STORE", db_path=tmp_db)
    rows = database.get_sessions_by_service("CHARGE", db_path=tmp_db)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def test_record_transaction_returns_id(tmp_db):
    user_id = database.create_user("TX001", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    tx_id = database.record_transaction(
        amount=10.0, session_id=session_id, db_path=tmp_db
    )
    assert isinstance(tx_id, int)


def test_record_transaction_negative_amount_raises(tmp_db):
    with pytest.raises(ValueError, match="negative"):
        database.record_transaction(amount=-1.0, db_path=tmp_db)


def test_record_transaction_invalid_status_raises(tmp_db):
    with pytest.raises(ValueError, match="Invalid transaction status"):
        database.record_transaction(amount=5.0, status="bad_status", db_path=tmp_db)


def test_get_transactions_returns_list(tmp_db):
    user_id = database.create_user("TX002", db_path=tmp_db)
    session_id = database.create_session(user_id, "CHARGE", db_path=tmp_db)
    database.record_transaction(amount=5.0, session_id=session_id, db_path=tmp_db)
    txs = database.get_transactions(limit=10, db_path=tmp_db)
    assert len(txs) >= 1
    assert txs[0]["amount"] == 5.0


def test_get_transactions_invalid_limit_raises(tmp_db):
    with pytest.raises(ValueError, match="positive integer"):
        database.get_transactions(limit=0, db_path=tmp_db)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def test_create_item_returns_id(tmp_db):
    item_id = database.create_item("ITEM001", "power_bank", db_path=tmp_db)
    assert isinstance(item_id, int)


def test_create_item_duplicate_rfid_returns_none(tmp_db):
    database.create_item("ITEM002", "power_bank", db_path=tmp_db)
    result = database.create_item("ITEM002", "cable", db_path=tmp_db)
    assert result is None


def test_create_item_empty_rfid_raises(tmp_db):
    with pytest.raises(ValueError, match="RFID"):
        database.create_item("", "power_bank", db_path=tmp_db)


def test_create_item_empty_type_raises(tmp_db):
    with pytest.raises(ValueError, match="Item type"):
        database.create_item("ITEM003", "", db_path=tmp_db)


def test_get_available_items_returns_available(tmp_db):
    database.create_item("AVAIL001", "power_bank", db_path=tmp_db)
    database.create_item("AVAIL002", "cable", db_path=tmp_db)
    items = database.get_available_items(db_path=tmp_db)
    rfids = {i["rfid_uid"] for i in items}
    assert "AVAIL001" in rfids
    assert "AVAIL002" in rfids


def test_get_item_by_rfid_returns_item(tmp_db):
    database.create_item("FINDME", "power_bank", db_path=tmp_db)
    item = database.get_item_by_rfid("FINDME", db_path=tmp_db)
    assert item is not None
    assert item["item_type"] == "power_bank"


def test_get_item_by_rfid_unknown_returns_none(tmp_db):
    assert database.get_item_by_rfid("NOTHERE", db_path=tmp_db) is None


# ---------------------------------------------------------------------------
# Borrow / return
# ---------------------------------------------------------------------------


def test_record_borrow_and_return(tmp_db):
    user_id = database.create_user("BORR001", db_path=tmp_db)
    item_id = database.create_item("BITM001", "power_bank", db_path=tmp_db)
    session_id = database.create_session(user_id, "BORROW", db_path=tmp_db)

    borrow_id = database.record_borrow(session_id, item_id, db_path=tmp_db)
    assert isinstance(borrow_id, int)

    # Item should now be unavailable
    items = database.get_available_items(db_path=tmp_db)
    assert not any(i["id"] == item_id for i in items)

    # Return
    result = database.record_return(item_id, db_path=tmp_db)
    assert result is True

    # Item should be available again
    items = database.get_available_items(db_path=tmp_db)
    assert any(i["id"] == item_id for i in items)


def test_borrow_unavailable_item_raises(tmp_db):
    user_id = database.create_user("BORR002", db_path=tmp_db)
    item_id = database.create_item("BITM002", "power_bank", db_path=tmp_db)
    session_id = database.create_session(user_id, "BORROW", db_path=tmp_db)

    database.record_borrow(session_id, item_id, db_path=tmp_db)

    # Try to borrow the same item again
    session_id2 = database.create_session(user_id, "BORROW", db_path=tmp_db)
    with pytest.raises(ValueError, match="not available"):
        database.record_borrow(session_id2, item_id, db_path=tmp_db)


def test_borrow_nonexistent_item_raises(tmp_db):
    user_id = database.create_user("BORR003", db_path=tmp_db)
    session_id = database.create_session(user_id, "BORROW", db_path=tmp_db)
    with pytest.raises(ValueError, match="does not exist"):
        database.record_borrow(session_id, 99999, db_path=tmp_db)


def test_return_without_active_borrow_raises(tmp_db):
    item_id = database.create_item("BITM003", "power_bank", db_path=tmp_db)
    with pytest.raises(ValueError, match="No active borrow"):
        database.record_return(item_id, db_path=tmp_db)


def test_borrow_inactive_session_raises(tmp_db):
    user_id = database.create_user("BORR004", db_path=tmp_db)
    item_id = database.create_item("BITM004", "cable", db_path=tmp_db)
    session_id = database.create_session(user_id, "BORROW", db_path=tmp_db)
    database.end_session(session_id, "completed", db_path=tmp_db)

    with pytest.raises(ValueError, match="not active"):
        database.record_borrow(session_id, item_id, db_path=tmp_db)


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------


def test_log_system_event_returns_id(tmp_db):
    ev_id = database.log_system_event("TEST_EVENT", "CHARGE", "msg", db_path=tmp_db)
    assert isinstance(ev_id, int)


def test_log_system_event_empty_type_raises(tmp_db):
    with pytest.raises(ValueError, match="Event type"):
        database.log_system_event("", db_path=tmp_db)


def test_get_recent_events_returns_list(tmp_db):
    database.log_system_event("EV_A", db_path=tmp_db)
    database.log_system_event("EV_B", db_path=tmp_db)
    events = database.get_recent_system_events(limit=10, db_path=tmp_db)
    assert len(events) >= 2


def test_get_recent_events_invalid_limit_raises(tmp_db):
    with pytest.raises(ValueError, match="positive integer"):
        database.get_recent_system_events(limit=-1, db_path=tmp_db)


# ---------------------------------------------------------------------------
# Foreign-key integrity
# ---------------------------------------------------------------------------


def test_foreign_key_session_user(tmp_db):
    """Session with invalid user_id must not be created."""
    import sqlite3

    with pytest.raises((sqlite3.IntegrityError, Exception)):
        database.execute_insert(
            "INSERT INTO sessions (user_id, service_id, status) VALUES (?, 1, 'active')",
            (99999,),
            db_path=tmp_db,
        )


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------


def test_dashboard_summary_structure(tmp_db):
    summary = database.get_dashboard_summary(db_path=tmp_db)
    for key in ("total_sessions", "active_sessions", "total_users",
                "total_revenue", "available_items", "borrowed_items",
                "service_counts"):
        assert key in summary
