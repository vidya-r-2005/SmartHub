"""
SmartHub Flask application factory and API routes.

Entry point for development:
    python app/main.py

Or via Flask CLI:
    set FLASK_APP=app.main
    flask run
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from app.config import Config, get_config, warn_if_insecure
from app.database import db as database

logger = logging.getLogger(__name__)

# Maximum allowed RFID UID length (matches db._MAX_RFID_LEN)
_MAX_RFID_LEN = 64


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(config: type[Config] | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    config : Config subclass | None
        Configuration class.  Resolved from SMARTHUB_ENV env var if None.
    """
    app = Flask(
        __name__,
        template_folder="dashboard/templates",
        static_folder="dashboard/static",
    )

    # Load configuration
    cfg = config or get_config()
    app.config.from_object(cfg)

    # Store db_path on app for easy access in route handlers
    app.config["DB_PATH"] = cfg.DATABASE_PATH

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Warn if SECRET_KEY is still the insecure default (no-op in testing)
    warn_if_insecure(cfg)

    # Initialise database and recover stale sessions from previous crashes
    with app.app_context():
        try:
            database.initialize_database(db_path=cfg.DATABASE_PATH)
            timeout_hours = getattr(cfg, "SESSION_TIMEOUT_HOURS", 8)
            recovered = database.timeout_stale_sessions(
                hours=timeout_hours, db_path=cfg.DATABASE_PATH
            )
            if recovered:
                logger.warning(
                    "Recovered %d stale session(s) on startup (threshold=%dh)",
                    recovered,
                    timeout_hours,
                )

            # Recover active hardware sessions
            from app.kiosk import get_kiosk_coordinator
            sim = getattr(cfg, "SIMULATION", True) or getattr(cfg, "TESTING", False)
            kiosk = get_kiosk_coordinator(simulation=sim, db_path=cfg.DATABASE_PATH)
            kiosk.recover_active_sessions()
        except Exception:
            logger.exception("Database initialisation failed — continuing anyway")

    # Register blueprints / route groups
    _register_routes(app)

    # Register centralised error handlers
    _register_error_handlers(app)

    return app


# ---------------------------------------------------------------------------
# Route registration helpers
# ---------------------------------------------------------------------------


def _db_path(app: Flask) -> Path:
    return Path(app.config["DB_PATH"])


def _validate_rfid_param(uid: str, label: str = "rfid_uid") -> tuple | None:
    """Return an error tuple if *uid* is empty or exceeds the max length."""
    if not uid:
        return _err(f"{label} is required", 400)
    if len(uid) > _MAX_RFID_LEN:
        return _err(
            f"{label} exceeds maximum length of {_MAX_RFID_LEN} characters", 400
        )
    return None


def _register_routes(app: Flask) -> None:
    """Attach all API routes to the application."""

    # --- Health / system -------------------------------------------------------

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "SmartHub"}), 200

    @app.get("/api/status")
    def status():
        try:
            summary = database.get_dashboard_summary(db_path=_db_path(app))
            return jsonify({
                "status": "ok",
                "simulation_mode": bool(app.config["SIMULATION_MODE"]),
                "charging_port_capacity": app.config["CHARGING_PORT_CAPACITY"],
                "power_bank_bay_capacity": app.config["POWER_BANK_BAY_CAPACITY"],
                "summary": summary,
            }), 200
        except Exception as exc:
            logger.error("Status check failed: %s", exc)
            return jsonify({"status": "error", "error": "Database unavailable"}), 503

    @app.get("/api/events")
    def events():
        limit = _parse_int_param("limit", default=50, min_val=1, max_val=500)
        if isinstance(limit, tuple):  # error response
            return limit
        rows = database.get_recent_system_events(limit=limit, db_path=_db_path(app))
        return jsonify([dict(r) for r in rows]), 200

    # --- Users -----------------------------------------------------------------

    @app.post("/api/users")
    def create_user():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        rfid_uid = data.get("rfid_uid", "").strip()
        err = _validate_rfid_param(rfid_uid)
        if err:
            return err
        try:
            user_id = database.create_user(rfid_uid, db_path=_db_path(app))
        except ValueError as exc:
            return _err(str(exc), 400)
        if user_id is None:
            return _err("RFID UID already registered", 409)
        return jsonify({"user_id": user_id, "rfid_uid": rfid_uid}), 201

    @app.get("/api/users/<int:user_id>")
    def get_user(user_id: int):
        user = database.get_user_by_id(user_id, db_path=_db_path(app))
        if user is None:
            return _err("User not found", 404)
        return jsonify(dict(user)), 200

    @app.get("/api/users/rfid/<string:uid>")
    def get_user_by_rfid(uid: str):
        err = _validate_rfid_param(uid, label="uid")
        if err:
            return err
        user = database.get_user_by_rfid(uid, db_path=_db_path(app))
        if user is None:
            return _err("User not found", 404)
        return jsonify(dict(user)), 200

    # --- Charge ----------------------------------------------------------------

    @app.post("/api/charge/start")
    def charge_start():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        rfid_uid = data.get("rfid_uid", "").strip()
        amount = data.get("amount", 0)
        payment_method = data.get("payment_method", "coin")
        err = _validate_rfid_param(rfid_uid)
        if err:
            return err
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            return _err("amount must be a number", 400)

        from app.services.charge import start_charging_session, ChargeServiceError

        try:
            result = start_charging_session(
                rfid_uid=rfid_uid,
                amount=amount,
                payment_method=payment_method,
                port_capacity=app.config["CHARGING_PORT_CAPACITY"],
                db_path=_db_path(app),
            )
            return jsonify(result), 201
        except ChargeServiceError as exc:
            return _err(str(exc), 400)
        except Exception as exc:
            logger.exception("Unexpected error in charge/start: %s", exc)
            return _err("Internal error", 500)

    @app.post("/api/charge/stop")
    def charge_stop():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        session_id = data.get("session_id")
        final_status = data.get("status", "completed")
        if session_id is None:
            return _err("session_id is required", 400)
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            return _err("session_id must be an integer", 400)

        from app.services.charge import stop_charging_session, ChargeServiceError

        try:
            result = stop_charging_session(
                session_id=session_id,
                final_status=final_status,
                db_path=_db_path(app),
            )
            return jsonify(result), 200
        except ChargeServiceError as exc:
            return _err(str(exc), 400)
        except Exception as exc:
            logger.exception("Unexpected error in charge/stop: %s", exc)
            return _err("Internal error", 500)

    @app.get("/api/charge/sessions")
    def charge_sessions():
        limit = _parse_int_param("limit", default=50)
        if isinstance(limit, tuple):
            return limit
        from app.services.charge import get_charge_sessions

        return jsonify(get_charge_sessions(limit=limit, db_path=_db_path(app))), 200

    @app.get("/api/charge/sessions/<int:session_id>")
    def charge_session(session_id: int):
        from app.services.charge import get_charge_session

        s = get_charge_session(session_id, db_path=_db_path(app))
        if s is None:
            return _err("Charging session not found", 404)
        return jsonify(s), 200

    @app.get("/api/charge/sessions/<int:session_id>/transactions")
    def charge_session_transactions(session_id: int):
        """Return all payment transactions for a specific charge session."""
        from app.services.charge import get_charge_transactions, get_charge_session

        # Verify the session exists and belongs to CHARGE
        s = get_charge_session(session_id, db_path=_db_path(app))
        if s is None:
            return _err("Charging session not found", 404)
        txns = get_charge_transactions(session_id, db_path=_db_path(app))
        return jsonify(txns), 200

    # --- Store -----------------------------------------------------------------

    @app.post("/api/store/start")
    def store_start():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        rfid_uid = data.get("rfid_uid", "").strip()
        err = _validate_rfid_param(rfid_uid)
        if err:
            return err

        from app.services.store import start_storage_session, StoreServiceError

        try:
            result = start_storage_session(rfid_uid=rfid_uid, db_path=_db_path(app))
            return jsonify(result), 201
        except StoreServiceError as exc:
            return _err(str(exc), 400)

    @app.post("/api/store/unlock")
    def store_unlock():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        session_id = data.get("session_id")
        if session_id is None:
            return _err("session_id is required", 400)
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            return _err("session_id must be an integer", 400)

        from app.services.store import unlock_storage, StoreServiceError

        try:
            result = unlock_storage(session_id=session_id, db_path=_db_path(app))
            return jsonify(result), 200
        except StoreServiceError as exc:
            return _err(str(exc), 400)

    @app.post("/api/store/finish")
    def store_finish():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        session_id = data.get("session_id")
        if session_id is None:
            return _err("session_id is required", 400)
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            return _err("session_id must be an integer", 400)

        from app.services.store import finish_storage_session, StoreServiceError

        try:
            result = finish_storage_session(session_id=session_id, db_path=_db_path(app))
            return jsonify(result), 200
        except StoreServiceError as exc:
            return _err(str(exc), 400)

    @app.post("/api/store/cancel")
    def store_cancel():
        """Cancel an active storage session (locks the locker and marks cancelled)."""
        data = _require_json()
        if isinstance(data, tuple):
            return data
        session_id = data.get("session_id")
        if session_id is None:
            return _err("session_id is required", 400)
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            return _err("session_id must be an integer", 400)

        from app.services.store import cancel_storage_session, StoreServiceError

        try:
            result = cancel_storage_session(session_id=session_id, db_path=_db_path(app))
            return jsonify(result), 200
        except StoreServiceError as exc:
            return _err(str(exc), 400)

    @app.get("/api/store/sessions")
    def store_sessions():
        limit = _parse_int_param("limit", default=50)
        if isinstance(limit, tuple):
            return limit
        from app.services.store import get_store_sessions

        return jsonify(get_store_sessions(limit=limit, db_path=_db_path(app))), 200

    # --- Borrow ----------------------------------------------------------------

    @app.get("/api/borrow/items")
    def borrow_items():
        from app.services.borrow import list_available_items

        return jsonify(list_available_items(db_path=_db_path(app))), 200

    @app.post("/api/borrow/start")
    def borrow_start():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        rfid_uid = data.get("rfid_uid", "").strip()
        item_rfid = data.get("item_rfid", "").strip()
        err = _validate_rfid_param(rfid_uid)
        if err:
            return err
        err = _validate_rfid_param(item_rfid, label="item_rfid")
        if err:
            return err

        from app.services.borrow import (
            start_borrow,
            BorrowServiceError,
            VerificationMismatchError,
        )

        try:
            result = start_borrow(
                rfid_uid=rfid_uid,
                item_rfid=item_rfid,
                db_path=_db_path(app),
            )
            return jsonify(result), 201
        except VerificationMismatchError as exc:
            return _err(str(exc), 422)
        except BorrowServiceError as exc:
            return _err(str(exc), 400)
        except Exception as exc:
            logger.exception("Unexpected error in borrow/start: %s", exc)
            return _err("Internal error", 500)

    @app.post("/api/borrow/return")
    def borrow_return():
        data = _require_json()
        if isinstance(data, tuple):
            return data
        item_rfid = data.get("item_rfid", "").strip()
        err = _validate_rfid_param(item_rfid, label="item_rfid")
        if err:
            return err

        from app.services.borrow import return_item, BorrowServiceError, VerificationMismatchError

        try:
            result = return_item(item_rfid=item_rfid, db_path=_db_path(app))
            return jsonify(result), 200
        except VerificationMismatchError as exc:
            return _err(str(exc), 422)
        except BorrowServiceError as exc:
            return _err(str(exc), 400)

    @app.get("/api/borrow/records")
    def borrow_records():
        limit = _parse_int_param("limit", default=50)
        if isinstance(limit, tuple):
            return limit
        from app.services.borrow import get_borrow_records

        return jsonify(get_borrow_records(limit=limit, db_path=_db_path(app))), 200

    # --- Dashboard / metrics ---------------------------------------------------

    @app.get("/api/dashboard/summary")
    def dashboard_summary():
        try:
            summary = database.get_dashboard_summary(db_path=_db_path(app))
            return jsonify(summary), 200
        except Exception as exc:
            logger.error("Dashboard summary failed: %s", exc)
            return _err("Could not retrieve summary", 503)

    @app.get("/api/metrics/usage")
    def metrics_usage():
        limit = _parse_int_param("limit", default=100, max_val=1000)
        if isinstance(limit, tuple):
            return limit
        rows = database.get_usage_metrics(limit=limit, db_path=_db_path(app))
        return jsonify([dict(r) for r in rows]), 200

    # --- ML integration --------------------------------------------------------

    @app.get("/api/ml/usage-prediction")
    def ml_usage_prediction():
        from app.ml.prediction import predict_next_hour_demand

        result = predict_next_hour_demand(db_path=str(_db_path(app)))
        status_code = 200 if result.get("available") else 503
        return jsonify(result), status_code

    @app.post("/api/ml/verify-item")
    def ml_verify_item():
        # Accept JSON body with base64 image or multipart file
        expected_category = ""
        image_data = b""
        mime_type = "image/jpeg"

        if request.content_type and "multipart" in request.content_type:
            file = request.files.get("image")
            expected_category = request.form.get("expected_category", "").strip()
            if file is None:
                return _err("image file is required", 400)
            if not expected_category:
                return _err("expected_category is required", 400)
            mime_type = file.mimetype or "image/jpeg"
            image_data = file.read()
        else:
            import base64

            body = _require_json()
            if isinstance(body, tuple):
                return body
            expected_category = body.get("expected_category", "").strip()
            if not expected_category:
                return _err("expected_category is required", 400)
            raw = body.get("image_base64", "")
            if not raw:
                return _err("image_base64 is required", 400)
            try:
                image_data = base64.b64decode(raw)
            except Exception:
                return _err("image_base64 is not valid base64", 400)
            mime_type = body.get("mime_type", "image/jpeg")

        from app.ml.vision import verify_item

        result = verify_item(image_data, expected_category, mime_type)
        status_code = 200 if result.get("available") else 503
        return jsonify(result), status_code

    # --- Kiosk State & Hardware Status -----------------------------------------

    @app.get("/api/kiosk/status")
    def kiosk_status():
        from app.kiosk import get_kiosk_coordinator
        sim = getattr(app.config, "SIMULATION", True) or app.config.get("TESTING", False)
        kiosk = get_kiosk_coordinator(simulation=sim, db_path=_db_path(app))
        return jsonify({
            "state": kiosk.state.value,
            "simulation": kiosk.simulation,
            "charging": kiosk.charging_ctrl.is_charging,
            "seconds_remaining": kiosk.charging_ctrl.seconds_remaining(),
            "current_user": kiosk.current_user_uid,
            "current_session": kiosk.current_session_id,
            "coins_inserted": kiosk.coins_inserted,
        }), 200

    @app.post("/api/kiosk/self-test")
    def kiosk_self_test():
        from app.kiosk import get_kiosk_coordinator
        sim = getattr(app.config, "SIMULATION", True) or app.config.get("TESTING", False)
        kiosk = get_kiosk_coordinator(simulation=sim, db_path=_db_path(app))
        results = kiosk.self_test()
        all_passed = all(results.values())
        return jsonify({
            "passed": all_passed,
            "results": results,
        }), 200 if all_passed else 500

    # --- Dashboard HTML --------------------------------------------------------

    @app.get("/")
    def index():
        try:
            from flask import render_template

            return render_template("index.html")
        except Exception:
            # Template not yet implemented by dashboard teammate
            return jsonify({"service": "SmartHub", "status": "ok"}), 200


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


def _register_error_handlers(app: Flask) -> None:
    @app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def payload_too_large(exc):
        return jsonify({"error": "Payload too large"}), 413

    @app.errorhandler(500)
    def internal_error(exc):
        logger.error("Unhandled exception: %s", exc)
        return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_json() -> dict | tuple:
    """Return the parsed JSON body or an error tuple."""
    if not request.is_json:
        return _err("Content-Type must be application/json", 415)
    data = request.get_json(silent=True)
    if data is None:
        return _err("Invalid JSON body", 400)
    return data


def _err(message: str, status: int) -> tuple:
    return jsonify({"error": message}), status


def _parse_int_param(
    name: str,
    default: int = 50,
    min_val: int = 1,
    max_val: int = 500,
) -> int | tuple:
    raw = request.args.get(name, str(default))
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return _err(f"'{name}' must be an integer", 400)
    if val < min_val or val > max_val:
        return _err(f"'{name}' must be between {min_val} and {max_val}", 400)
    return val


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _cfg = get_config()
    _app = create_app(_cfg)

    _app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=_cfg.DEBUG,
    )
