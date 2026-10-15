# SmartHub — Secure Personal Utility Kiosk

A self-service utility kiosk providing three services:
**CHARGE** (paid device charging), **STORE** (secure temporary storage), and **BORROW** (power-bank / accessory lending).

---

## 1. Backend Architecture

```
SmartHub/
├── app/
│   ├── main.py            # Flask application factory + all API routes
│   ├── config.py          # Development / Testing / Production config classes
│   ├── database/
│   │   ├── db.py          # All database helpers (fully parameterized SQL)
│   │   └── schema.sql     # SQLite schema + service seed data
│   ├── hardware/
│   │   ├── rfid.py        # RFID reader abstraction (sim + GPIO)
│   │   ├── coin.py        # Coin acceptor (sim + GPIO pulse)
│   │   ├── charging.py    # Charging relay / MOSFET control
│   │   ├── lock.py        # Electromagnetic lock / solenoid
│   │   ├── lcd.py         # I2C character LCD display
│   │   ├── buzzer.py      # PWM buzzer
│   │   └── buttons.py     # Push-button inputs
│   ├── services/
│   │   ├── charge.py      # CHARGE business logic
│   │   ├── store.py       # STORE business logic
│   │   └── borrow.py      # BORROW business logic + vision integration
│   ├── dashboard/
│   │   ├── static/        # CSS + JavaScript
│   │   └── templates/     # Jinja2 HTML templates
│   └── ml/
│       ├── prediction.py  # Usage-prediction integration boundary
│       └── vision.py      # Item-verification integration boundary
├── data/
│   └── smarthub.db        # Production SQLite database (gitignored)
├── tests/
│   ├── conftest.py        # Shared pytest fixtures (isolated temp DB)
│   ├── test_database.py   # 38 database layer tests
│   ├── test_charge.py     # 15 CHARGE service tests
│   ├── test_store.py      # 14 STORE service tests
│   ├── test_borrow.py     # 20 BORROW service tests
│   └── test_flask.py      # 37 Flask API route tests
├── pytest.ini
├── requirements.txt
└── README.md
```

The database layer is completely isolated for tests — every test fixture
creates a fresh temporary SQLite file. The production database is never
touched during test runs.

---

## 2. Python Version

**Python 3.14.3** (tested).  Python 3.11+ is required (uses `type | None`
union syntax in type hints).

---

## 3. Environment Setup

### Windows PowerShell

```powershell
# Create a virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Raspberry Pi OS (bash)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | 3.1.1   | HTTP server and routing |
| `pytest`| 9.1.1   | Automated testing |

Raspberry Pi hardware libraries (`RPi.GPIO`, `mfrc522`, `RPLCD`) are
**not** in `requirements.txt` — they are imported lazily inside physical
driver paths only and are not needed to run the application in simulation
mode or to execute tests.

---

## 5. Configuration and Mock Mode

Configuration is selected via the `SMARTHUB_ENV` environment variable:

| Value | Class | Use |
|-------|-------|-----|
| `development` (default) | `DevelopmentConfig` | Local dev, simulation mode ON |
| `testing` | `TestingConfig` | Automated tests, temp DB |
| `production` | `ProductionConfig` | Raspberry Pi deployment |

**Simulation mode** (`SIMULATION_MODE = True`) is the default for
development and testing.  In this mode all hardware calls (RFID, coin
acceptor, charging relay, lock, LCD, buzzer) are replaced by in-process
software stubs — no GPIO or physical device is required.

To run in physical mode on a Raspberry Pi, set:

```bash
export SMARTHUB_ENV=production
export SIMULATION_MODE=   # empty = False → physical mode
```

---

## 6. Database Initialisation

```powershell
# Windows
python -c "from app.database.db import initialize_database; initialize_database()"
```

```bash
# Raspberry Pi
python3 -c "from app.database.db import initialize_database; initialize_database()"
```

This reads `app/database/schema.sql`, creates `data/smarthub.db` if it
does not exist, and inserts the three services (`CHARGE`, `STORE`,
`BORROW`) using `INSERT OR IGNORE`.  Safe to re-run on an existing
database.

---

## 7. Running the Application

```powershell
# Windows — development server
python app/main.py

# Or via Flask CLI
$env:FLASK_APP = "app.main"
flask run --host=0.0.0.0 --port=5000
```

```bash
# Raspberry Pi
python3 app/main.py
```

The server listens on `http://0.0.0.0:5000` by default.
Set the `PORT` environment variable to override.

---

## 8. Running Tests

```powershell
# Full suite
python -m pytest tests/ -v

# Collect only (no execution)
python -m pytest tests/ --collect-only -q

# Single module
python -m pytest tests/test_database.py -v
```

All tests use isolated temporary databases.  The production database
`data/smarthub.db` is **never** accessed during test runs.

**Test results (Python 3.14.3):**
```
124 passed in ~6s
```

---

## 9. API Endpoints

All endpoints accept and return `application/json`.

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check |
| `GET` | `/api/status` | Database summary |
| `GET` | `/api/events?limit=N` | Recent system events |

**`GET /api/health`** → `200 {"status": "ok", "service": "SmartHub"}`

---

### Users

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/users` | Register user by RFID |
| `GET`  | `/api/users/<id>` | Fetch user by ID |
| `GET`  | `/api/users/rfid/<uid>` | Fetch user by RFID |

**`POST /api/users`**
```json
Request:  { "rfid_uid": "AABBCCDD" }
Response 201: { "user_id": 1, "rfid_uid": "AABBCCDD" }
Response 409: { "error": "RFID UID already registered" }
```

---

### Charging

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/charge/start` | Start charging session |
| `POST` | `/api/charge/stop`  | Stop / cancel session |
| `GET`  | `/api/charge/sessions` | List CHARGE sessions |
| `GET`  | `/api/charge/sessions/<id>` | Single session |

**`POST /api/charge/start`**
```json
Request:  { "rfid_uid": "AABBCCDD", "amount": 10.0, "payment_method": "coin" }
Response 201: { "session_id": 1, "transaction_id": 1, "user_id": 1, "status": "active" }
```

**`POST /api/charge/stop`**
```json
Request:  { "session_id": 1, "status": "completed" }
Response 200: { "session_id": 1, "status": "completed", "duration_seconds": 42 }
```

---

### Storage

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/store/start`  | Start storage session |
| `POST` | `/api/store/unlock` | Disengage lock |
| `POST` | `/api/store/finish` | Re-lock + end session |
| `GET`  | `/api/store/sessions` | List STORE sessions |

**Workflow:** `start` → `unlock` → user places/retrieves item → `finish`

```json
POST /api/store/start
Request:  { "rfid_uid": "AABBCCDD" }
Response 201: { "session_id": 2, "user_id": 1, "status": "active" }

POST /api/store/unlock
Request:  { "session_id": 2 }
Response 200: { "session_id": 2, "locked": false }

POST /api/store/finish
Request:  { "session_id": 2 }
Response 200: { "session_id": 2, "status": "completed", "duration_seconds": 15 }
```

---

### Borrowing

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/borrow/items` | Available items |
| `POST` | `/api/borrow/start` | Borrow an item |
| `POST` | `/api/borrow/return` | Return an item |
| `GET`  | `/api/borrow/records` | Borrow history |

```json
POST /api/borrow/start
Request:  { "rfid_uid": "AABBCCDD", "item_rfid": "ITEM001" }
Response 201: { "session_id": 3, "borrow_record_id": 1, "user_id": 1,
                "item_id": 1, "item_type": "power_bank", "vision_result": null }

POST /api/borrow/return
Request:  { "item_rfid": "ITEM001" }
Response 200: { "item_id": 1, "item_type": "power_bank",
                "status": "returned", "vision_result": null }
```

---

### Dashboard / Metrics

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/dashboard/summary` | Aggregate counts |
| `GET` | `/api/metrics/usage?limit=N` | Recent completed sessions |

**`GET /api/dashboard/summary`**
```json
{
  "total_sessions": 42,
  "active_sessions": 3,
  "total_users": 15,
  "total_revenue": 210.50,
  "available_items": 5,
  "borrowed_items": 2,
  "service_counts": {"CHARGE": 20, "STORE": 12, "BORROW": 10}
}
```

---

### ML Integration

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/ml/usage-prediction` | Next-hour demand forecast |
| `POST` | `/api/ml/verify-item` | Vision item verification |

`GET /api/ml/usage-prediction` returns `503` when the model is not yet
configured:
```json
{ "available": false, "predicted_demand": null, "error": "Prediction model not yet configured" }
```

`POST /api/ml/verify-item` accepts either:
- **JSON body**: `{ "expected_category": "power_bank", "image_base64": "<base64>", "mime_type": "image/jpeg" }`
- **Multipart form**: `image` file + `expected_category` field

Returns `503` when the vision model is not configured.

---

## 10. Database Tables

| Table | Purpose |
|-------|---------|
| `users` | RFID-identified users |
| `services` | CHARGE, STORE, BORROW service registry |
| `sessions` | Every service session with timestamps and status |
| `transactions` | Payment records linked to sessions |
| `items` | Borrowable items with RFID UIDs |
| `borrow_records` | Borrow / return history |
| `system_events` | Audit log for all service events |

Foreign keys are enforced via `PRAGMA foreign_keys = ON` on every connection.

---

## 11. Hardware Integration Status

All hardware drivers have been implemented with a **simulation / physical**
dual-mode design. The following table shows the current readiness:

| Hardware | Simulation | Physical driver |
|----------|-----------|----------------|
| RFID reader (MFRC522) | ✅ | ✅ (deferred import) |
| Coin acceptor (pulse) | ✅ | ✅ (deferred GPIO import) |
| Charging relay | ✅ | ✅ (deferred GPIO import) |
| Lock solenoid | ✅ | ✅ (deferred GPIO import) |
| I2C LCD (16×2) | ✅ | ✅ (deferred RPLCD import) |
| PWM Buzzer | ✅ | ✅ (deferred GPIO import) |
| Push-buttons | ✅ | ✅ (deferred GPIO import) |
| Camera (vision) | ✅ (mock) | ⚠️ Model not yet trained |

> **Note:** Physical drivers have not been tested on actual Raspberry Pi
> hardware.  Only simulation mode has been exercised.  Physical pin
> numbers must be confirmed against your hardware wiring before deployment.

---

## 12. ML Integration Interfaces

### Usage Prediction (`app/ml/prediction.py`)

- **Function**: `predict_next_hour_demand(historical_data, db_path) → dict`
- **Input**: Historical CHARGE session records (list of dicts with `started_at`, `duration_seconds`)
- **Output**: `{ "available": bool, "predicted_demand": float|None, "confidence": float|None, "model": str, "error": str|None }`
- **Intended model**: Chronos-2 or similar time-series forecasting model
- **Status**: Integration boundary defined. Model implementation (`app/ml/_prediction_model.py`) pending.

### Item Verification (`app/ml/vision.py`)

- **Function**: `verify_item(image_data: bytes, expected_category: str, mime_type: str) → dict`
- **Input**: Raw image bytes (JPEG/PNG/WebP, ≤4 MB) + expected item category string
- **Output**: `{ "available": bool, "match": bool|None, "category": str|None, "confidence": float|None, "expected": str, "error": str|None }`
- **Intended model**: CNN-based object classifier
- **Status**: Integration boundary defined. Model implementation (`app/ml/_vision_model.py`) pending.

Both modules fail gracefully when their respective implementations are
absent — they return `{"available": false, ...}` rather than raising.

---

## 13. Known Limitations and Remaining Work

| Area | Status | Notes |
|------|--------|-------|
| Database schema | ✅ Complete | No changes needed |
| DB helper functions | ✅ Complete | All functions with isolation support |
| CHARGE service | ✅ Complete | Simulation tested |
| STORE service | ✅ Complete | Simulation tested |
| BORROW service | ✅ Complete | Vision mock tested |
| Flask API | ✅ Complete | All routes implemented |
| Hardware sim | ✅ Complete | All drivers simulated |
| Hardware physical | ⚠️ Untested | Requires Raspberry Pi + wiring |
| ML prediction model | ⚠️ Pending | Stub boundary only |
| ML vision model | ⚠️ Pending | Stub boundary only |
| Dashboard HTML/JS/CSS | ⚠️ Pending | Backend API ready for integration |
| Production secrets | ⚠️ Action needed | Set SECRET_KEY env var before production |
| Camera image upload | ✅ Boundary defined | Vision model required for real use |

---

## 14. Next Commands to Run

```powershell
# Install dependencies (if not already done)
pip install -r requirements.txt

# Initialise the database
python -c "from app.database.db import initialize_database; initialize_database()"

# Run the full test suite
python -m pytest tests/ -v

# Start the development server
python app/main.py
```

---

## 15. Git Status

Working branch: **`backend-dev`**

No commits or pushes have been made by the automated implementation. All
changes are unstaged working-tree modifications. Run `git diff` to review
before committing.

## Dashboard frontend

A responsive Flask-rendered dashboard is served at `/`. It includes a live overview of session totals, active sessions, item availability, service counts, recent events, system health, and the next-hour CHARGE forecast when configured. CHARGE supports session start/stop, STORE supports start/unlock/finish with confirmations, and BORROW lists available items and submits borrow/return requests. API failures and unconfigured ML services show an honest unavailable state.

The current STORE API does not return locker assignments or locker inventory, so the UI does not invent a locker number. The BORROW start and return APIs accept RFID fields only; they do not accept an image. `POST /api/ml/verify-item` remains a separate endpoint (multipart `image` plus `expected_category`, or JSON with `image_base64`, `mime_type`, and `expected_category`) and is not wired into borrowing.

Frontend files: `app/dashboard/templates/index.html`, `app/dashboard/static/style.css`, and `app/dashboard/static/script.js`. Dashboard integrations use `GET /api/dashboard/summary`, `/api/metrics/usage`, `/api/status`, `/api/events`, `/api/health`, `/api/ml/usage-prediction`; CHARGE uses `POST /api/charge/start` and `/api/charge/stop`; STORE uses `POST /api/store/start`, `/api/store/unlock`, and `/api/store/finish`; BORROW uses `GET /api/borrow/items`, `POST /api/borrow/start`, and `/api/borrow/return`.

## Dashboard setup and use

Python 3.11 or newer is required. Install and run with:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/ -v
python app/main.py
```

Open <http://127.0.0.1:5000/>. `SMARTHUB_ENV` selects development (default), testing, or production configuration. Set a private `SECRET_KEY` for production; `SESSION_TIMEOUT_HOURS` controls stale-session recovery (default 8), `PORT` changes the development server port (default 5000), and production `SIMULATION_MODE` selects simulated hardware. Do not commit secret values.

Hardware simulation supports local development. Physical RFID, coin, charging, and lock operations require compatible Raspberry Pi hardware and verified wiring. Forecast and image verification models are optional and report unavailable until configured. CHARGE records amounts and payment methods but does not process card payments. Tests use isolated temporary databases and require no hardware, money, or trained ML models. Run the complete suite with `python -m pytest tests/ -v`.
