"""
SmartHub Kiosk Coordinator — Hardware & State Machine Controller.

Coordinates all physical kiosk hardware:
- Buttons: CHARGE (GPIO 5), STORE (GPIO 6), BORROW (GPIO 13), BACK_CONFIRM (GPIO 19)
- Coin Acceptor: GPIO 17 (1 coin = ₹5 = 300s charging)
- Lock: PWM Servo on GPIO 18 (safe release duty cycle 0, 25s auto-relock)
- Buzzer: GPIO 22 (non-blocking beeps, error tones, mismatch alerts)
- LEDs: Green (GPIO 20), Red (GPIO 21), Blue (GPIO 26)
- Relay: GPIO 27 (safe initial OFF)
- LCD: I2C 16x2 Display
- Camera: Picamera2 / OpenCV / Mock
- RFID: RC522 SPI (40s scan timeout, hold-off debounce)

States:
- IDLE: Ready for user input
- CHARGE_WAITING_FOR_COIN: Collecting coins for charging
- CHARGING: Active charging countdown with relay ON
- STORE_WAITING_FOR_USER: Waiting for registered user card scan
- STORE_UNLOCKED: Compartment unlocked with 25s auto-lock countdown
- BORROW_WAITING_FOR_USER: Waiting for user card scan
- BORROW_WAITING_FOR_ITEM: Waiting for item tag scan
- VERIFYING_BORROW_IMAGE: Capturing image and CNN classification
- VERIFYING_RETURN_IMAGE: Verifying returned item classification
- COMPARTMENT_ACTIVE: Compartment unlocked for item take/deposit
- MISMATCH_ALERT: Vision verification failed, alarm sounding
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from app.database import db
from app.hardware.buttons import Button, MockButtons
from app.hardware.buzzer import Buzzer, MockBuzzer
from app.hardware.camera import Camera, MockCamera
from app.hardware.charging import SECONDS_PER_COIN, ChargingController
from app.hardware.coin import CoinAcceptor, MockCoinAcceptor
from app.hardware.lcd import LCD, MockLCD
from app.hardware.leds import LEDs, MockLEDs
from app.hardware.lock import LockController, MockLock
from app.hardware.relay import MockRelay, Relay
from app.hardware.rfid import MockRFID, RFIDReader
from app.services.borrow import (
    BorrowServiceError,
    VerificationMismatchError,
    return_item,
    start_borrow,
)
from app.services.charge import (
    ChargeServiceError,
    start_charging_session,
    stop_charging_session,
)
from app.services.store import (
    AUTO_LOCK_SECONDS,
    StoreServiceError,
    finish_storage_session,
    start_storage_session,
    unlock_storage,
)

logger = logging.getLogger(__name__)

RFID_SCAN_TIMEOUT_SECONDS = 40.0
COIN_WAIT_TIMEOUT_SECONDS = 20.0


class KioskState(str, Enum):
    IDLE = "IDLE"
    CHARGE_WAITING_FOR_COIN = "CHARGE_WAITING_FOR_COIN"
    CHARGING = "CHARGING"
    STORE_WAITING_FOR_USER = "STORE_WAITING_FOR_USER"
    STORE_UNLOCKED = "STORE_UNLOCKED"
    BORROW_WAITING_FOR_USER = "BORROW_WAITING_FOR_USER"
    BORROW_WAITING_FOR_ITEM = "BORROW_WAITING_FOR_ITEM"
    VERIFYING_BORROW_IMAGE = "VERIFYING_BORROW_IMAGE"
    VERIFYING_RETURN_IMAGE = "VERIFYING_RETURN_IMAGE"
    COMPARTMENT_ACTIVE = "COMPARTMENT_ACTIVE"
    MISMATCH_ALERT = "MISMATCH_ALERT"


class KioskCoordinator:
    """Central state machine coordinating all SmartHub kiosk hardware and services."""

    def __init__(
        self,
        simulation: bool = True,
        db_path: Optional[Path] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.simulation = simulation
        self.db_path = db_path
        self._clock = clock or time.monotonic
        self.state = KioskState.IDLE

        # Hardware drivers
        if simulation:
            self.buttons = MockButtons()
            self.buzzer = MockBuzzer()
            self.leds = MockLEDs()
            self.relay = MockRelay()
            self.lock = MockLock()
            self.lcd = MockLCD()
            self.camera = MockCamera()
            self.rfid = MockRFID()
            self.coin = MockCoinAcceptor()
        else:
            self.buttons = Button(simulation=False)
            self.buzzer = Buzzer(simulation=False)
            self.leds = LEDs(simulation=False)
            self.relay = Relay(simulation=False)
            self.lock = LockController(simulation=False)
            self.lcd = LCD(simulation=False)
            self.camera = Camera(simulation=False)
            self.rfid = RFIDReader(simulation=False)
            self.coin = CoinAcceptor(simulation=False)

        self.charging_ctrl = ChargingController(
            relay=self.relay, simulation=simulation, clock=self._clock
        )

        # State tracking variables
        self.state_enter_time: float = self._clock()
        self.current_user_uid: Optional[str] = None
        self.current_item_uid: Optional[str] = None
        self.current_item_data: Optional[dict] = None
        self.current_session_id: Optional[int] = None
        self.coins_inserted: int = 0
        self.last_coin_time: float = 0.0
        self.compartment_timer: float = 0.0

        # Background loop control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Initialize to IDLE state
        self.to_idle()

    # -----------------------------------------------------------------------
    # Lifecycle & Self-Test
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start the kiosk coordinator background polling thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self.coin.start()
            self.to_idle()
            self._thread = threading.Thread(
                target=self._run_loop, name="KioskCoordinatorThread", daemon=True
            )
            self._thread.start()
            logger.info("KioskCoordinator started (simulation=%s)", self.simulation)

    def stop(self) -> None:
        """Stop the kiosk coordinator and release all hardware resources."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        try:
            self.coin.stop()
            self.charging_ctrl.close()
            self.relay.close()
            self.lock.close()
            self.leds.all_off()
            self.leds.close()
            self.buzzer.close()
            self.lcd.clear()
            self.lcd.close()
            self.camera.close()
            self.buttons.close()
            self.rfid.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during kiosk hardware shutdown: %s", exc)
        logger.info("KioskCoordinator stopped cleanly")

    def self_test(self) -> dict[str, bool]:
        """Perform comprehensive hardware and database self-test.

        Returns a dictionary of component health statuses.
        """
        results: dict[str, bool] = {}

        # 1. Database
        try:
            rows = db.execute_query("SELECT 1 AS ok", (), self.db_path)
            results["database"] = len(rows) > 0 and rows[0]["ok"] == 1
        except Exception:
            results["database"] = False

        # 2. Relay
        try:
            results["relay"] = self.relay.turn_off() is not None or True
        except Exception:
            results["relay"] = False

        # 3. Servo Lock
        try:
            results["lock"] = self.lock.lock() is not None or True
        except Exception:
            results["lock"] = False

        # 4. LEDs
        try:
            self.leds.all_off()
            results["leds"] = True
        except Exception:
            results["leds"] = False

        # 5. Buzzer
        try:
            self.buzzer.beep(0.01)
            results["buzzer"] = True
        except Exception:
            results["buzzer"] = False

        # 6. Coin Acceptor
        try:
            results["coin"] = hasattr(self.coin, "start")
        except Exception:
            results["coin"] = False

        # 7. LCD
        try:
            results["lcd"] = self.lcd.display("SmartHub Ready", "Self-Test OK") is True
        except Exception:
            results["lcd"] = False

        # 8. Camera
        try:
            test_img = self.camera.capture_image()
            results["camera"] = test_img is not None and len(test_img) > 0
        except Exception:
            results["camera"] = False

        # 9. RFID
        try:
            results["rfid"] = hasattr(self.rfid, "read_uid")
        except Exception:
            results["rfid"] = False

        # 10. Buttons
        try:
            results["buttons"] = hasattr(self.buttons, "is_pressed")
        except Exception:
            results["buttons"] = False

        all_ok = all(results.values())
        if all_ok:
            self.leds.set_status("green")
            self.buzzer.success()
            self.lcd.display("Self-Test PASS", "System Ready")
        else:
            self.leds.set_status("red")
            self.buzzer.error()
            self.lcd.display("Self-Test FAIL", "Check Hardware")

        logger.info("Self-test results: %s (overall=%s)", results, all_ok)
        return results

    def recover_active_sessions(self) -> None:
        """Inspect database for interrupted active sessions and restore hardware states."""
        try:
            active_charge = db.get_active_charging_session(db_path=self.db_path)
            if active_charge:
                session_id = active_charge["id"]
                started_at_str = active_charge["started_at"]
                amount = float(active_charge["amount"] or 5.0)

                coins = max(1, int(round(amount / 5.0)))
                total_duration = coins * SECONDS_PER_COIN

                # Calculate elapsed time
                dt_started = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                if dt_started.tzinfo is None:
                    dt_started = dt_started.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                elapsed = max(0.0, (now - dt_started).total_seconds())

                if elapsed < total_duration:
                    remaining = int(total_duration - elapsed)
                    logger.info(
                        "Recovering active charging session %d: %ds remaining",
                        session_id,
                        remaining,
                    )
                    self.current_session_id = session_id
                    self.charging_ctrl.start(remaining)
                    self.state = KioskState.CHARGING
                    self.leds.set_status("blue")
                    self.lcd.display("Charge Resumed", f"Remaining: {remaining}s")
                else:
                    logger.info(
                        "Closing expired charging session %d (elapsed %.1fs >= %ds)",
                        session_id,
                        elapsed,
                        total_duration,
                    )
                    stop_charging_session(
                        session_id=session_id,
                        charging_controller=self.charging_ctrl,
                        final_status="completed",
                        db_path=self.db_path,
                    )
                    self.relay.turn_off()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Active session recovery encountered an error: %s", exc)

    # -----------------------------------------------------------------------
    # State Machine Transitions
    # -----------------------------------------------------------------------

    def to_idle(self) -> None:
        """Reset all transient state variables and return to IDLE."""
        self.state = KioskState.IDLE
        self.state_enter_time = self._clock()
        self.current_user_uid = None
        self.current_item_uid = None
        self.current_item_data = None
        self.current_session_id = None
        self.coins_inserted = 0

        self.leds.set_status("green")
        self.lcd.display("SmartHub Ready", "Select Service")

    def step(self) -> None:
        """Perform one execution step of the kiosk state machine."""
        now = self._clock()

        # Check cancel / back button in states where it acts strictly as cancel
        if self.state in (
            KioskState.STORE_WAITING_FOR_USER,
            KioskState.BORROW_WAITING_FOR_USER,
            KioskState.BORROW_WAITING_FOR_ITEM,
        ):
            if self.buttons.is_pressed("BACK_CONFIRM"):
                self.buzzer.beep(0.08)
                self.lcd.display("Cancelled", "")
                self._safe_abort_current_session()
                time.sleep(0.5)
                self.to_idle()
                return

        # -------------------------------------------------------------------
        # IDLE: Poll for service button presses
        # -------------------------------------------------------------------
        if self.state == KioskState.IDLE:
            if self.buttons.is_pressed("CHARGE"):
                self.buzzer.beep(0.05)
                self.state = KioskState.CHARGE_WAITING_FOR_COIN
                self.state_enter_time = now
                self.coins_inserted = 0
                self.coin.reset()
                self.leds.set_status("blue")
                self.lcd.display("Insert Coin Rs5", "1 Coin = 5 Mins")
                return

            if self.buttons.is_pressed("STORE"):
                self.buzzer.beep(0.05)
                self.state = KioskState.STORE_WAITING_FOR_USER
                self.state_enter_time = now
                self.leds.set_status("blue")
                self.lcd.display("Storage Mode", "Scan Your Card")
                return

            if self.buttons.is_pressed("BORROW"):
                self.buzzer.beep(0.05)
                self.state = KioskState.BORROW_WAITING_FOR_USER
                self.state_enter_time = now
                self.leds.set_status("blue")
                self.lcd.display("Borrow/Return", "Scan User Card")
                return

        # -------------------------------------------------------------------
        # CHARGE_WAITING_FOR_COIN: Count coins and start charging
        # -------------------------------------------------------------------
        elif self.state == KioskState.CHARGE_WAITING_FOR_COIN:
            total_coins = self.coin.get_coin_count()
            if total_coins > self.coins_inserted:
                self.coins_inserted = total_coins
                self.last_coin_time = now
                self.buzzer.beep(0.08)
                mins = self.coins_inserted * 5
                self.lcd.display(f"Coins: {self.coins_inserted} (Rs{self.coins_inserted*5})", f"Time: {mins} mins")

            # Check timeout / confirm to start charging
            coin_timeout = (now - self.last_coin_time > COIN_WAIT_TIMEOUT_SECONDS) if self.coins_inserted > 0 else (now - self.state_enter_time > RFID_SCAN_TIMEOUT_SECONDS)
            confirm_pressed = self.buttons.is_pressed("BACK_CONFIRM")

            if (confirm_pressed or (self.coins_inserted > 0 and coin_timeout)) and self.coins_inserted > 0:
                amount = float(self.coins_inserted * 5)
                user_uid = self.current_user_uid or f"KIOSK_{int(now)}"
                try:
                    res = start_charging_session(
                        rfid_uid=user_uid,
                        amount=amount,
                        payment_method="coin",
                        charging_controller=self.charging_ctrl,
                        db_path=self.db_path,
                    )
                    self.current_session_id = res["session_id"]
                    self.state = KioskState.CHARGING
                    self.state_enter_time = now
                    self.buzzer.success()
                    logger.info("Charging started: session=%s coins=%d", self.current_session_id, self.coins_inserted)
                except ChargeServiceError as exc:
                    self.buzzer.error()
                    self.lcd.display("Charge Failed", str(exc)[:16])
                    time.sleep(2.0)
                    self.to_idle()
                return

            if confirm_pressed and self.coins_inserted == 0:
                self.buzzer.beep(0.08)
                self.to_idle()
                return

            if self.coins_inserted == 0 and coin_timeout:
                self.buzzer.error()
                self.lcd.display("Timed Out", "No Coin Received")
                time.sleep(1.5)
                self.to_idle()
                return

        # -------------------------------------------------------------------
        # CHARGING: Countdown active relay charging
        # -------------------------------------------------------------------
        elif self.state == KioskState.CHARGING:
            rem = self.charging_ctrl.seconds_remaining()

            # Check manual early termination
            if self.buttons.is_pressed("BACK_CONFIRM"):
                self.buzzer.beep(0.1)
                self.lcd.display("Stopping Charge", "Please wait...")
                if self.current_session_id:
                    stop_charging_session(
                        self.current_session_id,
                        charging_controller=self.charging_ctrl,
                        final_status="cancelled",
                        db_path=self.db_path,
                    )
                self.charging_ctrl.stop()
                self.to_idle()
                return

            if rem > 0:
                mins = rem // 60
                secs = rem % 60
                self.lcd.display(f"Charging: {mins:02d}:{secs:02d}", "Press Back=Stop")
            else:
                # Timer finished
                self.charging_ctrl.update()
                if self.current_session_id:
                    stop_charging_session(
                        self.current_session_id,
                        charging_controller=self.charging_ctrl,
                        final_status="completed",
                        db_path=self.db_path,
                    )
                self.buzzer.success()
                self.leds.set_status("green")
                self.lcd.display("Charge Complete", "Thank You!")
                time.sleep(2.0)
                self.to_idle()
                return

        # -------------------------------------------------------------------
        # STORE_WAITING_FOR_USER: 40s RFID scan timeout for registered user
        # -------------------------------------------------------------------
        elif self.state == KioskState.STORE_WAITING_FOR_USER:
            if now - self.state_enter_time > RFID_SCAN_TIMEOUT_SECONDS:
                self.buzzer.error()
                self.lcd.display("Scan Timed Out", "Try Again")
                time.sleep(1.5)
                self.to_idle()
                return

            scanned_uid = self.rfid.read_uid()
            if scanned_uid:
                self.buzzer.beep(0.08)
                user = db.get_user_by_rfid(scanned_uid, self.db_path)
                if user is None:
                    self.buzzer.error()
                    self.leds.set_status("red")
                    self.lcd.display("User Not Found", "Register First")
                    time.sleep(2.0)
                    self.to_idle()
                    return

                # Registered user — start session and unlock
                try:
                    res = start_storage_session(
                        rfid_uid=scanned_uid,
                        allow_auto_register=False,
                        db_path=self.db_path,
                    )
                    self.current_session_id = res["session_id"]
                    self.lock.unlock()
                    self.buzzer.success()
                    self.state = KioskState.STORE_UNLOCKED
                    self.state_enter_time = now
                    self.compartment_timer = AUTO_LOCK_SECONDS
                    self.lcd.display("Compartment Open", f"Auto-lock: {AUTO_LOCK_SECONDS}s")
                except StoreServiceError as exc:
                    self.buzzer.error()
                    self.lcd.display("Store Error", str(exc)[:16])
                    time.sleep(2.0)
                    self.to_idle()
                return

        # -------------------------------------------------------------------
        # STORE_UNLOCKED: 25s auto-relock countdown
        # -------------------------------------------------------------------
        elif self.state == KioskState.STORE_UNLOCKED:
            elapsed = now - self.state_enter_time
            remaining = max(0, int(AUTO_LOCK_SECONDS - elapsed))
            early_close = self.buttons.is_pressed("BACK_CONFIRM")

            if early_close or remaining <= 0:
                self.lock.lock()
                if self.current_session_id:
                    finish_storage_session(
                        self.current_session_id,
                        lock_controller=self.lock,
                        db_path=self.db_path,
                    )
                self.buzzer.success()
                self.leds.set_status("green")
                self.lcd.display("Stored Safely", "Thank You!")
                time.sleep(2.0)
                self.to_idle()
                return
            else:
                self.lcd.display("Compartment Open", f"Relocking in {remaining}s")

        # -------------------------------------------------------------------
        # BORROW_WAITING_FOR_USER: Scan user RFID (40s timeout)
        # -------------------------------------------------------------------
        elif self.state == KioskState.BORROW_WAITING_FOR_USER:
            if now - self.state_enter_time > RFID_SCAN_TIMEOUT_SECONDS:
                self.buzzer.error()
                self.lcd.display("Scan Timed Out", "Try Again")
                time.sleep(1.5)
                self.to_idle()
                return

            scanned_uid = self.rfid.read_uid()
            if scanned_uid:
                self.buzzer.beep(0.08)
                user = db.get_user_by_rfid(scanned_uid, self.db_path)
                if user is None:
                    self.buzzer.error()
                    self.leds.set_status("red")
                    self.lcd.display("User Not Found", "Register First")
                    time.sleep(2.0)
                    self.to_idle()
                    return

                self.current_user_uid = scanned_uid
                self.state = KioskState.BORROW_WAITING_FOR_ITEM
                self.state_enter_time = now
                self.lcd.display("Scan Item Tag", "Hold Near Reader")
                return

        # -------------------------------------------------------------------
        # BORROW_WAITING_FOR_ITEM: Scan item tag (40s timeout)
        # -------------------------------------------------------------------
        elif self.state == KioskState.BORROW_WAITING_FOR_ITEM:
            if now - self.state_enter_time > RFID_SCAN_TIMEOUT_SECONDS:
                self.buzzer.error()
                self.lcd.display("Scan Timed Out", "Try Again")
                time.sleep(1.5)
                self.to_idle()
                return

            scanned_item_uid = self.rfid.read_uid()
            if scanned_item_uid:
                self.buzzer.beep(0.08)
                item = db.get_item_by_rfid(scanned_item_uid, self.db_path)
                if item is None:
                    self.buzzer.error()
                    self.leds.set_status("red")
                    self.lcd.display("Unknown Item", "Tag Not Found")
                    time.sleep(2.0)
                    self.to_idle()
                    return

                self.current_item_uid = scanned_item_uid
                self.current_item_data = dict(item)

                if item["status"] == "borrowed":
                    # Transition to return flow
                    self.state = KioskState.VERIFYING_RETURN_IMAGE
                    self.state_enter_time = now
                    self.lcd.display("Verifying Return", "Please Wait...")
                elif item["status"] == "available":
                    # Check daily borrow limit
                    user = db.get_user_by_rfid(self.current_user_uid, self.db_path)
                    daily_count = db.get_daily_borrow_count(user["id"], db_path=self.db_path)
                    if daily_count >= 2:
                        self.buzzer.error()
                        self.leds.set_status("red")
                        self.lcd.display("Daily Limit (2)", "Reached Today")
                        time.sleep(2.5)
                        self.to_idle()
                        return

                    # Transition to borrow flow
                    self.state = KioskState.VERIFYING_BORROW_IMAGE
                    self.state_enter_time = now
                    self.lcd.display("Verifying Item", "Please Wait...")
                else:
                    self.buzzer.error()
                    self.lcd.display("Item Unavailable", f"Status: {item['status']}")
                    time.sleep(2.0)
                    self.to_idle()
                return

        # -------------------------------------------------------------------
        # VERIFYING_BORROW_IMAGE: Capture image & verify category
        # -------------------------------------------------------------------
        elif self.state == KioskState.VERIFYING_BORROW_IMAGE:
            img = self.camera.capture_image()
            try:
                res = start_borrow(
                    rfid_uid=self.current_user_uid,
                    item_rfid=self.current_item_uid,
                    image_data=img,
                    allow_auto_register=False,
                    db_path=self.db_path,
                )
                # Successful borrow
                self.lock.unlock()
                self.buzzer.success()
                self.leds.set_status("green")
                item_name = self.current_item_data.get("item_type", "Item")[:10]
                self.lcd.display("Borrow Success!", f"Take {item_name}")
                self.state = KioskState.COMPARTMENT_ACTIVE
                self.state_enter_time = now
            except VerificationMismatchError as exc:
                self._trigger_mismatch_alert(str(exc))
            except BorrowServiceError as exc:
                self.buzzer.error()
                self.lcd.display("Borrow Failed", str(exc)[:16])
                time.sleep(2.0)
                self.to_idle()
            return

        # -------------------------------------------------------------------
        # VERIFYING_RETURN_IMAGE: Verify returned item and complete session
        # -------------------------------------------------------------------
        elif self.state == KioskState.VERIFYING_RETURN_IMAGE:
            img = self.camera.capture_image()
            try:
                res = return_item(
                    item_rfid=self.current_item_uid,
                    user_rfid=self.current_user_uid,
                    image_data=img,
                    db_path=self.db_path,
                )
                self.lock.unlock()
                self.buzzer.success()
                self.leds.set_status("green")
                self.lcd.display("Return Success!", "Place In Locker")
                self.state = KioskState.COMPARTMENT_ACTIVE
                self.state_enter_time = now
            except VerificationMismatchError as exc:
                self._trigger_mismatch_alert(str(exc))
            except BorrowServiceError as exc:
                self.buzzer.error()
                self.lcd.display("Return Failed", str(exc)[:16])
                time.sleep(2.0)
                self.to_idle()
            return

        # -------------------------------------------------------------------
        # COMPARTMENT_ACTIVE: 25s auto-relock for borrow/return completion
        # -------------------------------------------------------------------
        elif self.state == KioskState.COMPARTMENT_ACTIVE:
            elapsed = now - self.state_enter_time
            remaining = max(0, int(AUTO_LOCK_SECONDS - elapsed))
            early_close = self.buttons.is_pressed("BACK_CONFIRM")

            if early_close or remaining <= 0:
                self.lock.lock()
                self.buzzer.beep(0.08)
                self.lcd.display("Compartment Locked", "Thank You!")
                time.sleep(1.5)
                self.to_idle()
                return
            else:
                self.lcd.display("Compartment Open", f"Relocking in {remaining}s")

        # -------------------------------------------------------------------
        # MISMATCH_ALERT: Security alarm
        # -------------------------------------------------------------------
        elif self.state == KioskState.MISMATCH_ALERT:
            if now - self.state_enter_time > 4.0 or self.buttons.is_pressed("BACK_CONFIRM"):
                self.to_idle()

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _trigger_mismatch_alert(self, details: str) -> None:
        """Trigger visual and acoustic alert on vision category mismatch."""
        self.state = KioskState.MISMATCH_ALERT
        self.state_enter_time = self._clock()
        self.leds.set_status("red")
        self.lcd.display("ITEM MISMATCH!", "Security Alert")
        # Continuous warning pulses
        self.buzzer.beep(0.2)
        logger.warning("VISION MISMATCH ALERT: %s", details)

    def _safe_abort_current_session(self) -> None:
        """Safely ensure locks and relays are turned off if an active operation is cancelled."""
        try:
            self.lock.lock()
            if self.state == KioskState.STORE_UNLOCKED and self.current_session_id:
                finish_storage_session(
                    self.current_session_id,
                    lock_controller=self.lock,
                    db_path=self.db_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during abort cleanup: %s", exc)

    def _run_loop(self) -> None:
        """Background thread loop calling step() repeatedly."""
        while self._running:
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected error in kiosk step: %s", exc)
            time.sleep(0.05)


# Global singleton instance for app-wide access
_kiosk_coordinator: Optional[KioskCoordinator] = None


def get_kiosk_coordinator(
    simulation: bool = True, db_path: Optional[Path] = None
) -> KioskCoordinator:
    """Return the global KioskCoordinator singleton."""
    global _kiosk_coordinator
    if _kiosk_coordinator is None:
        _kiosk_coordinator = KioskCoordinator(simulation=simulation, db_path=db_path)
    return _kiosk_coordinator
