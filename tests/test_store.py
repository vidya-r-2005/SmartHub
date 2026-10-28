"""
Tests for the STORE service (app/services/store.py).

Lock hardware is replaced by simulation or injected mock controllers.
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

from app.services.store import (
    StoreServiceError,
    start_storage_session,
    unlock_storage,
    finish_storage_session,
    get_store_sessions,
)
from app.hardware.lock import LockController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_lock(fail: bool = False) -> LockController:
    """Return a simulation lock, optionally rigged to fail."""
    return LockController(simulation=True, fail_result=(False if fail else None))


# ---------------------------------------------------------------------------
# start_storage_session
# ---------------------------------------------------------------------------


def test_start_storage_session_returns_session(tmp_db):
    result = start_storage_session(rfid_uid="STORE_USER_1", db_path=tmp_db)
    assert "session_id" in result
    assert result["status"] == "active"


def test_start_storage_missing_rfid_raises(tmp_db):
    with pytest.raises(StoreServiceError, match="RFID"):
        start_storage_session(rfid_uid="", db_path=tmp_db)


def test_start_storage_auto_registers_user(tmp_db):
    from app.database import db as database

    start_storage_session(rfid_uid="STORE_AUTO", db_path=tmp_db)
    user = database.get_user_by_rfid("STORE_AUTO", db_path=tmp_db)
    assert user is not None


# ---------------------------------------------------------------------------
# unlock_storage
# ---------------------------------------------------------------------------


def test_unlock_storage_succeeds(tmp_db):
    result = start_storage_session(rfid_uid="UNLOCK_USER", db_path=tmp_db)
    unlock_result = unlock_storage(
        session_id=result["session_id"],
        lock_controller=_sim_lock(),
        db_path=tmp_db,
    )
    assert unlock_result["locked"] is False


def test_unlock_nonexistent_session_raises(tmp_db):
    with pytest.raises(StoreServiceError, match="does not exist"):
        unlock_storage(session_id=99999, lock_controller=_sim_lock(), db_path=tmp_db)


def test_unlock_already_ended_session_raises(tmp_db):
    from app.database import db as database

    result = start_storage_session(rfid_uid="ENDED_SESS", db_path=tmp_db)
    sid = result["session_id"]
    database.end_session(sid, "completed", db_path=tmp_db)

    with pytest.raises(StoreServiceError, match="not active"):
        unlock_storage(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)


def test_unlock_hardware_failure_raises_and_marks_failed(tmp_db):
    result = start_storage_session(rfid_uid="LOCK_FAIL", db_path=tmp_db)
    sid = result["session_id"]

    with pytest.raises(StoreServiceError, match="hardware failed"):
        unlock_storage(
            session_id=sid,
            lock_controller=_sim_lock(fail=True),
            db_path=tmp_db,
        )

    # Session must be marked failed
    from app.database import db as database
    session = database.get_session(sid, db_path=tmp_db)
    assert session["status"] == "failed"


def test_unlock_no_false_success_on_hardware_failure(tmp_db):
    """After a hardware lock failure, the session must NOT remain active."""
    result = start_storage_session(rfid_uid="NO_FALSE_OK", db_path=tmp_db)
    sid = result["session_id"]

    try:
        unlock_storage(session_id=sid, lock_controller=_sim_lock(fail=True), db_path=tmp_db)
    except StoreServiceError:
        pass

    from app.database import db as database
    session = database.get_session(sid, db_path=tmp_db)
    assert session["status"] != "active"


# ---------------------------------------------------------------------------
# finish_storage_session
# ---------------------------------------------------------------------------


def test_finish_storage_session_completes(tmp_db):
    result = start_storage_session(rfid_uid="FINISH_USER", db_path=tmp_db)
    sid = result["session_id"]

    unlock_storage(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)
    finish_result = finish_storage_session(
        session_id=sid,
        lock_controller=_sim_lock(),
        db_path=tmp_db,
    )
    assert finish_result["status"] == "completed"
    assert finish_result["duration_seconds"] is not None


def test_finish_nonexistent_session_raises(tmp_db):
    with pytest.raises(StoreServiceError, match="does not exist"):
        finish_storage_session(session_id=99999, lock_controller=_sim_lock(), db_path=tmp_db)


def test_finish_already_finished_session_raises(tmp_db):
    result = start_storage_session(rfid_uid="DOUBLE_FINISH", db_path=tmp_db)
    sid = result["session_id"]
    finish_storage_session(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)

    with pytest.raises(StoreServiceError, match="not active"):
        finish_storage_session(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)


def test_finish_lock_failure_marks_failed(tmp_db):
    result = start_storage_session(rfid_uid="RE_LOCK_FAIL", db_path=tmp_db)
    sid = result["session_id"]

    finish_result = finish_storage_session(
        session_id=sid,
        lock_controller=_sim_lock(fail=True),
        db_path=tmp_db,
    )
    assert finish_result["status"] == "failed"


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------


def test_events_logged_for_storage_workflow(tmp_db):
    from app.database import db as database

    result = start_storage_session(rfid_uid="EVENT_LOG", db_path=tmp_db)
    sid = result["session_id"]
    unlock_storage(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)
    finish_storage_session(session_id=sid, lock_controller=_sim_lock(), db_path=tmp_db)

    events = database.get_recent_system_events(limit=20, db_path=tmp_db)
    event_types = {ev["event_type"] for ev in events}
    assert "STORE_START" in event_types
    assert "STORE_UNLOCK" in event_types
    assert "STORE_FINISH" in event_types


# ---------------------------------------------------------------------------
# get_store_sessions
# ---------------------------------------------------------------------------


def test_get_store_sessions_returns_list(tmp_db):
    start_storage_session(rfid_uid="SL_1", db_path=tmp_db)
    start_storage_session(rfid_uid="SL_2", db_path=tmp_db)
    sessions = get_store_sessions(db_path=tmp_db)
    assert len(sessions) >= 2


def test_start_storage_rejects_unregistered_when_disallowed(tmp_db):
    with pytest.raises(StoreServiceError, match="not registered"):
        start_storage_session(
            rfid_uid="UNKNOWN_STORAGE_USER",
            allow_auto_register=False,
            db_path=tmp_db,
        )


