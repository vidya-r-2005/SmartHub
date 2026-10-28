"""
Tests for the CHARGE service (app/services/charge.py).

Hardware is replaced by simulation controllers.
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

from app.services.charge import (
    ChargeServiceError,
    start_charging_session,
    stop_charging_session,
    get_charge_sessions,
    get_charge_session,
)
from app.hardware.charging import ChargingController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim_ctrl(fail: bool = False) -> ChargingController:
    """Return a simulation controller, optionally rigged to fail."""
    ctrl = ChargingController(simulation=True)
    if fail:
        # Monkey-patch enable/disable to return False
        ctrl.enable = lambda: False
        ctrl.disable = lambda: False
    return ctrl


# ---------------------------------------------------------------------------
# start_charging_session
# ---------------------------------------------------------------------------


def test_start_session_creates_session(tmp_db):
    result = start_charging_session(
        rfid_uid="CHARGE_USER_1",
        amount=10.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert "session_id" in result
    assert "transaction_id" in result
    assert result["status"] == "active"


def test_start_session_auto_registers_user(tmp_db):
    from app.database import db as database

    result = start_charging_session(
        rfid_uid="NEW_USER_AUTO",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    user = database.get_user_by_rfid("NEW_USER_AUTO", db_path=tmp_db)
    assert user is not None
    assert user["id"] == result["user_id"]


def test_start_session_refuses_when_configured_port_capacity_is_full(tmp_db):
    start_charging_session(
        rfid_uid="PORT_USER_1", amount=0, port_capacity=1,
        charging_controller=_sim_ctrl(), db_path=tmp_db,
    )
    with pytest.raises(ChargeServiceError, match="No charging port"):
        start_charging_session(
            rfid_uid="PORT_USER_2", amount=0, port_capacity=1,
            charging_controller=_sim_ctrl(), db_path=tmp_db,
        )


def test_start_session_missing_rfid_raises(tmp_db):
    with pytest.raises(ChargeServiceError, match="RFID"):
        start_charging_session(
            rfid_uid="",
            amount=5.0,
            charging_controller=_sim_ctrl(),
            db_path=tmp_db,
        )


def test_start_session_negative_amount_raises(tmp_db):
    with pytest.raises(ChargeServiceError, match="amount"):
        start_charging_session(
            rfid_uid="CHG001",
            amount=-1.0,
            charging_controller=_sim_ctrl(),
            db_path=tmp_db,
        )


def test_start_session_non_numeric_amount_raises(tmp_db):
    with pytest.raises(ChargeServiceError, match="number"):
        start_charging_session(
            rfid_uid="CHG002",
            amount="not_a_number",
            charging_controller=_sim_ctrl(),
            db_path=tmp_db,
        )


def test_start_session_hardware_failure_raises_and_rollback(tmp_db):
    from app.database import db as database

    with pytest.raises(ChargeServiceError, match="hardware failed"):
        start_charging_session(
            rfid_uid="CHG_HW_FAIL",
            amount=5.0,
            charging_controller=_sim_ctrl(fail=True),
            db_path=tmp_db,
        )

    # No active sessions should remain
    sessions = get_charge_sessions(db_path=tmp_db)
    active = [s for s in sessions if s["status"] == "active"]
    assert len(active) == 0


# ---------------------------------------------------------------------------
# stop_charging_session
# ---------------------------------------------------------------------------


def test_stop_session_completes_session(tmp_db):
    start_result = start_charging_session(
        rfid_uid="STOP_USER_1",
        amount=10.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    session_id = start_result["session_id"]

    stop_result = stop_charging_session(
        session_id=session_id,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert stop_result["status"] == "completed"
    assert stop_result["session_id"] == session_id


def test_stop_session_records_duration(tmp_db):
    start_result = start_charging_session(
        rfid_uid="DUR_USER",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    stop_result = stop_charging_session(
        session_id=start_result["session_id"],
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    # Duration should be a non-negative integer
    assert stop_result["duration_seconds"] is not None
    assert stop_result["duration_seconds"] >= 0


def test_stop_session_already_stopped_raises(tmp_db):
    start_result = start_charging_session(
        rfid_uid="DOUBLE_STOP",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    session_id = start_result["session_id"]
    stop_charging_session(session_id=session_id, charging_controller=_sim_ctrl(), db_path=tmp_db)

    with pytest.raises(ChargeServiceError, match="not active"):
        stop_charging_session(session_id=session_id, charging_controller=_sim_ctrl(), db_path=tmp_db)


def test_stop_nonexistent_session_raises(tmp_db):
    with pytest.raises(ChargeServiceError, match="does not exist"):
        stop_charging_session(session_id=99999, db_path=tmp_db)


def test_stop_session_invalid_status_raises(tmp_db):
    start_result = start_charging_session(
        rfid_uid="BAD_STATUS",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    with pytest.raises(ChargeServiceError, match="Invalid final status"):
        stop_charging_session(
            session_id=start_result["session_id"],
            final_status="bad_value",
            db_path=tmp_db,
        )


def test_cancel_session(tmp_db):
    start_result = start_charging_session(
        rfid_uid="CANCEL_USER",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    stop_result = stop_charging_session(
        session_id=start_result["session_id"],
        final_status="cancelled",
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert stop_result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# get_charge_sessions / get_charge_session
# ---------------------------------------------------------------------------


def test_get_charge_sessions_returns_list(tmp_db):
    start_charging_session(rfid_uid="LIST_1", amount=5.0, charging_controller=_sim_ctrl(), db_path=tmp_db)
    start_charging_session(rfid_uid="LIST_2", amount=10.0, charging_controller=_sim_ctrl(), db_path=tmp_db)
    sessions = get_charge_sessions(db_path=tmp_db)
    assert len(sessions) >= 2


def test_get_charge_session_returns_session(tmp_db):
    result = start_charging_session(
        rfid_uid="SINGLE_SESSION",
        amount=7.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    s = get_charge_session(result["session_id"], db_path=tmp_db)
    assert s is not None
    assert s["service_name"] == "CHARGE"


def test_get_charge_session_nonexistent_returns_none(tmp_db):
    assert get_charge_session(99999, db_path=tmp_db) is None


# ---------------------------------------------------------------------------
# Pricing Rule Tests (1 coin = 5 min = 300s)
# ---------------------------------------------------------------------------


def test_start_session_pricing_duration_conversion(tmp_db):
    # 1 coin (₹5) -> 300s
    res1 = start_charging_session(
        rfid_uid="COIN_USER_1",
        amount=5.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert res1["duration_seconds"] == 300

    # 2 coins (₹10) -> 600s
    res2 = start_charging_session(
        rfid_uid="COIN_USER_2",
        amount=10.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert res2["duration_seconds"] == 600

    # 3 coins (₹15) -> 900s
    res3 = start_charging_session(
        rfid_uid="COIN_USER_3",
        amount=15.0,
        charging_controller=_sim_ctrl(),
        db_path=tmp_db,
    )
    assert res3["duration_seconds"] == 900


