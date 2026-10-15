"""
Tests for Flask API routes (app/main.py).

Uses the Flask test client from the conftest fixtures.
All requests use a temporary isolated database.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Health / status endpoints
# ---------------------------------------------------------------------------


def test_health_returns_ok(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


def test_status_returns_summary(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert "summary" in data


def test_events_endpoint_returns_list(client):
    resp = client.get("/api/events")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_events_limit_out_of_range(client):
    resp = client.get("/api/events?limit=0")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


def test_create_user(client):
    resp = client.post(
        "/api/users",
        json={"rfid_uid": "TEST_RFID_A"},
        content_type="application/json",
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert "user_id" in data
    assert data["rfid_uid"] == "TEST_RFID_A"


def test_create_user_duplicate_returns_409(client):
    client.post("/api/users", json={"rfid_uid": "DUP_UID"})
    resp = client.post("/api/users", json={"rfid_uid": "DUP_UID"})
    assert resp.status_code == 409


def test_create_user_missing_rfid_returns_400(client):
    resp = client.post("/api/users", json={})
    assert resp.status_code == 400


def test_create_user_non_json_returns_415(client):
    resp = client.post("/api/users", data="notjson", content_type="text/plain")
    assert resp.status_code == 415


def test_get_user_by_id(client):
    create_resp = client.post("/api/users", json={"rfid_uid": "GET_BY_ID"})
    user_id = create_resp.get_json()["user_id"]
    resp = client.get(f"/api/users/{user_id}")
    assert resp.status_code == 200
    assert resp.get_json()["rfid_uid"] == "GET_BY_ID"


def test_get_user_by_id_not_found_returns_404(client):
    resp = client.get("/api/users/99999")
    assert resp.status_code == 404


def test_get_user_by_rfid(client):
    client.post("/api/users", json={"rfid_uid": "RFID_LOOKUP"})
    resp = client.get("/api/users/rfid/RFID_LOOKUP")
    assert resp.status_code == 200
    assert resp.get_json()["rfid_uid"] == "RFID_LOOKUP"


def test_get_user_by_rfid_not_found_returns_404(client):
    resp = client.get("/api/users/rfid/NOTREGISTERED")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Charge endpoints
# ---------------------------------------------------------------------------


def test_charge_start(client):
    resp = client.post(
        "/api/charge/start",
        json={"rfid_uid": "CHG_FLASK_1", "amount": 10.0},
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert "session_id" in data


def test_charge_start_missing_rfid_returns_400(client):
    resp = client.post("/api/charge/start", json={"amount": 5.0})
    assert resp.status_code == 400


def test_charge_start_invalid_amount_returns_400(client):
    resp = client.post(
        "/api/charge/start",
        json={"rfid_uid": "CHG_F2", "amount": "banana"},
    )
    assert resp.status_code == 400


def test_charge_stop(client):
    start = client.post(
        "/api/charge/start",
        json={"rfid_uid": "CHG_STOP_1", "amount": 5.0},
    ).get_json()
    resp = client.post("/api/charge/stop", json={"session_id": start["session_id"]})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "completed"


def test_charge_stop_missing_session_id_returns_400(client):
    resp = client.post("/api/charge/stop", json={})
    assert resp.status_code == 400


def test_charge_stop_nonexistent_session_returns_400(client):
    resp = client.post("/api/charge/stop", json={"session_id": 99999})
    assert resp.status_code == 400


def test_charge_sessions_list(client):
    resp = client.get("/api/charge/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_charge_session_by_id(client):
    start = client.post(
        "/api/charge/start",
        json={"rfid_uid": "CHG_ID_LOOKUP", "amount": 5.0},
    ).get_json()
    resp = client.get(f"/api/charge/sessions/{start['session_id']}")
    assert resp.status_code == 200


def test_charge_session_not_found_returns_404(client):
    resp = client.get("/api/charge/sessions/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Store endpoints
# ---------------------------------------------------------------------------


def test_store_start(client):
    resp = client.post("/api/store/start", json={"rfid_uid": "STORE_F1"})
    assert resp.status_code == 201
    assert "session_id" in resp.get_json()


def test_store_unlock(client):
    start = client.post("/api/store/start", json={"rfid_uid": "STORE_F2"}).get_json()
    resp = client.post("/api/store/unlock", json={"session_id": start["session_id"]})
    assert resp.status_code == 200
    assert resp.get_json()["locked"] is False


def test_store_finish(client):
    start = client.post("/api/store/start", json={"rfid_uid": "STORE_F3"}).get_json()
    sid = start["session_id"]
    client.post("/api/store/unlock", json={"session_id": sid})
    resp = client.post("/api/store/finish", json={"session_id": sid})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "completed"


def test_store_sessions_list(client):
    resp = client.get("/api/store/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


# ---------------------------------------------------------------------------
# Borrow endpoints
# ---------------------------------------------------------------------------


def test_borrow_items_list_empty(client):
    resp = client.get("/api/borrow/items")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


def test_borrow_start(client, tmp_db, app):
    from app.database import db as database

    database.create_item("FLASK_ITEM_1", "power_bank", db_path=tmp_db)

    resp = client.post(
        "/api/borrow/start",
        json={"rfid_uid": "BORROW_F1", "item_rfid": "FLASK_ITEM_1"},
    )
    assert resp.status_code == 201


def test_borrow_start_missing_rfid_returns_400(client):
    resp = client.post("/api/borrow/start", json={"item_rfid": "SOME_ITEM"})
    assert resp.status_code == 400


def test_borrow_return(client, tmp_db):
    from app.database import db as database

    database.create_item("FLASK_RET_1", "cable", db_path=tmp_db)
    client.post(
        "/api/borrow/start",
        json={"rfid_uid": "RET_F1", "item_rfid": "FLASK_RET_1"},
    )
    resp = client.post("/api/borrow/return", json={"item_rfid": "FLASK_RET_1"})
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "returned"


def test_borrow_records_list(client):
    resp = client.get("/api/borrow/records")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


# ---------------------------------------------------------------------------
# Dashboard / metrics
# ---------------------------------------------------------------------------


def test_dashboard_summary(client):
    resp = client.get("/api/dashboard/summary")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_sessions" in data
    assert "total_users" in data


def test_metrics_usage(client):
    resp = client.get("/api/metrics/usage")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), list)


# ---------------------------------------------------------------------------
# ML endpoints
# ---------------------------------------------------------------------------


def test_ml_usage_prediction_unavailable(client):
    """When no model is configured, endpoint returns 503 with clear message."""
    resp = client.get("/api/ml/usage-prediction")
    assert resp.status_code == 503
    data = resp.get_json()
    assert data["available"] is False
    assert "error" in data


def test_ml_verify_item_missing_fields_returns_400(client):
    resp = client.post(
        "/api/ml/verify-item",
        json={},
        content_type="application/json",
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 404 / error handlers
# ---------------------------------------------------------------------------


def test_unknown_route_returns_404(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404
    assert "error" in resp.get_json()

