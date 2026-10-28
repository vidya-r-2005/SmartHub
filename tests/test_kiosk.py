"""
Unit and state machine tests for KioskCoordinator (app/kiosk.py).

Tests run entirely in simulation mode without physical hardware.
Uses isolated SQLite databases for each test.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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

from app.database import db
from app.kiosk import KioskCoordinator, KioskState
from app.services.borrow import start_borrow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.current = start

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def _make_user(rfid: str, tmp_db: Path) -> int:
    return db.create_user(rfid, db_path=tmp_db)


def _make_item(rfid: str, item_type: str, tmp_db: Path) -> int:
    return db.create_item(rfid, item_type, db_path=tmp_db)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_kiosk_initial_state(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    assert kiosk.state == KioskState.IDLE
    assert not kiosk.relay.is_on
    assert kiosk.lock.is_locked
    assert kiosk.leds.status == "green"
    kiosk.stop()


def test_kiosk_self_test(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    results = kiosk.self_test()
    assert all(results.values())
    assert results["database"] is True
    assert results["relay"] is True
    assert results["lock"] is True
    assert results["camera"] is True
    assert kiosk.leds.status == "green"
    kiosk.stop()


def test_kiosk_charge_flow(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)

    # 1. Press CHARGE button
    kiosk.buttons.press("CHARGE")
    kiosk.step()
    assert kiosk.state == KioskState.CHARGE_WAITING_FOR_COIN
    assert kiosk.leds.status == "blue"

    # 2. Insert 2 coins (₹10 = 600s)
    kiosk.coin.insert_coins(2)
    kiosk.step()
    assert kiosk.coins_inserted == 2

    # 3. Press BACK_CONFIRM to start charging
    kiosk.buttons.press("BACK_CONFIRM")
    kiosk.step()
    assert kiosk.state == KioskState.CHARGING
    assert kiosk.relay.is_on
    assert kiosk.charging_ctrl.is_charging
    assert kiosk.charging_ctrl.seconds_remaining() == 600

    # 4. Advance clock by 601 seconds -> charging completes
    clock.advance(601)
    kiosk.step()
    assert kiosk.state == KioskState.IDLE
    assert not kiosk.relay.is_on
    assert not kiosk.charging_ctrl.is_charging

    kiosk.stop()


def test_kiosk_store_flow_registered_user(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    _make_user("STORE_USER_CARD", tmp_db)

    # 1. Press STORE button
    kiosk.buttons.press("STORE")
    kiosk.step()
    assert kiosk.state == KioskState.STORE_WAITING_FOR_USER

    # 2. Scan registered user RFID card
    kiosk.rfid.set_tag("STORE_USER_CARD")
    kiosk.step()
    assert kiosk.state == KioskState.STORE_UNLOCKED
    assert not kiosk.lock.is_locked  # Locker unlocked

    # 3. Advance past 25s auto-lock timeout
    clock.advance(26)
    kiosk.step()
    assert kiosk.state == KioskState.IDLE
    assert kiosk.lock.is_locked  # Locker relocked safely

    kiosk.stop()


def test_kiosk_store_flow_unregistered_user_rejected(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)

    # 1. Press STORE
    kiosk.buttons.press("STORE")
    kiosk.step()

    # 2. Scan UNREGISTERED RFID
    kiosk.rfid.set_tag("UNKNOWN_CARD")
    kiosk.step()

    # Should reject and return to IDLE without unlocking
    assert kiosk.state == KioskState.IDLE
    assert kiosk.lock.is_locked

    kiosk.stop()


def test_kiosk_borrow_flow_match(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    _make_user("BORROWER_1", tmp_db)
    _make_item("TAG_PB_1", "power_bank", tmp_db)

    # 1. Press BORROW
    kiosk.buttons.press("BORROW")
    kiosk.step()
    assert kiosk.state == KioskState.BORROW_WAITING_FOR_USER

    # 2. Scan User Card
    kiosk.rfid.set_tag("BORROWER_1")
    kiosk.step()
    assert kiosk.state == KioskState.BORROW_WAITING_FOR_ITEM
    assert kiosk.current_user_uid == "BORROWER_1"

    # 3. Scan Item Tag
    kiosk.rfid.set_tag("TAG_PB_1")
    kiosk.step()
    assert kiosk.state == KioskState.VERIFYING_BORROW_IMAGE

    # 4. Vision matches
    fake_match = {
        "available": True,
        "match": True,
        "category": "power_bank",
        "confidence": 0.96,
        "expected": "power_bank",
    }
    with patch("app.services.borrow._verify_vision", return_value=fake_match):
        kiosk.step()

    assert kiosk.state == KioskState.COMPARTMENT_ACTIVE
    assert not kiosk.lock.is_locked

    # 5. Compartment auto-relocks after 25s
    clock.advance(26)
    kiosk.step()
    assert kiosk.state == KioskState.IDLE
    assert kiosk.lock.is_locked

    # Verify item status in DB is now 'borrowed'
    item = db.get_item_by_rfid("TAG_PB_1", tmp_db)
    assert item["status"] == "borrowed"

    kiosk.stop()


def test_kiosk_borrow_flow_mismatch_triggers_alert(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    _make_user("BORROWER_2", tmp_db)
    _make_item("TAG_PB_2", "power_bank", tmp_db)

    kiosk.buttons.press("BORROW")
    kiosk.step()
    kiosk.rfid.set_tag("BORROWER_2")
    kiosk.step()
    kiosk.rfid.set_tag("TAG_PB_2")
    kiosk.step()

    fake_mismatch = {
        "available": True,
        "match": False,
        "category": "umbrella",
        "confidence": 0.90,
        "expected": "power_bank",
    }
    with patch("app.services.borrow._verify_vision", return_value=fake_mismatch):
        kiosk.step()

    assert kiosk.state == KioskState.MISMATCH_ALERT
    assert kiosk.leds.status == "red"
    assert kiosk.lock.is_locked  # Locker remains locked!

    # Item must NOT be borrowed
    item = db.get_item_by_rfid("TAG_PB_2", tmp_db)
    assert item["status"] == "available"

    kiosk.stop()


def test_kiosk_borrow_daily_limit_rejected(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    _make_user("LIMIT_TESTER", tmp_db)
    _make_item("ITEM_1", "cable", tmp_db)
    _make_item("ITEM_2", "cable", tmp_db)
    _make_item("ITEM_3", "cable", tmp_db)

    # Pre-record 2 borrows for this user today
    start_borrow("LIMIT_TESTER", "ITEM_1", db_path=tmp_db)
    start_borrow("LIMIT_TESTER", "ITEM_2", db_path=tmp_db)

    # Attempt 3rd borrow via kiosk
    kiosk.buttons.press("BORROW")
    kiosk.step()
    kiosk.rfid.set_tag("LIMIT_TESTER")
    kiosk.step()
    kiosk.rfid.set_tag("ITEM_3")
    kiosk.step()

    # Must be rejected and return to IDLE without unlocking
    assert kiosk.state == KioskState.IDLE
    assert kiosk.lock.is_locked

    item = db.get_item_by_rfid("ITEM_3", tmp_db)
    assert item["status"] == "available"

    kiosk.stop()


def test_kiosk_return_flow(tmp_db: Path) -> None:
    clock = MockClock()
    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    _make_user("RET_USER", tmp_db)
    _make_item("RET_TAG", "power_bank", tmp_db)

    # Initial borrow
    start_borrow("RET_USER", "RET_TAG", db_path=tmp_db)

    # Return flow: press BORROW -> scan user -> scan item
    kiosk.buttons.press("BORROW")
    kiosk.step()
    kiosk.rfid.set_tag("RET_USER")
    kiosk.step()
    kiosk.rfid.set_tag("RET_TAG")
    kiosk.step()
    assert kiosk.state == KioskState.VERIFYING_RETURN_IMAGE

    fake_match = {
        "available": True,
        "match": True,
        "category": "power_bank",
        "confidence": 0.95,
        "expected": "power_bank",
    }
    with patch("app.services.borrow._verify_vision", return_value=fake_match):
        kiosk.step()

    assert kiosk.state == KioskState.COMPARTMENT_ACTIVE
    assert not kiosk.lock.is_locked

    clock.advance(26)
    kiosk.step()
    assert kiosk.state == KioskState.IDLE
    assert kiosk.lock.is_locked

    item = db.get_item_by_rfid("RET_TAG", tmp_db)
    assert item["status"] == "available"

    kiosk.stop()


def test_kiosk_recovery_active_charging(tmp_db: Path) -> None:
    clock = MockClock()
    # Create an active charging session in DB
    user_id = db.create_user("REC_USER", tmp_db)
    session_id = db.create_session(user_id, "CHARGE", tmp_db)
    db.record_transaction(amount=5.0, session_id=session_id, db_path=tmp_db)

    kiosk = KioskCoordinator(simulation=True, db_path=tmp_db, clock=clock)
    kiosk.recover_active_sessions()

    # Session was created just now, so remaining seconds should be > 200
    assert kiosk.state == KioskState.CHARGING
    assert kiosk.relay.is_on
    assert kiosk.charging_ctrl.is_charging
    assert kiosk.charging_ctrl.seconds_remaining() > 0

    kiosk.stop()
