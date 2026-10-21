# SmartHub — Secure Personal Utility Kiosk

SmartHub is a small-scale, Raspberry Pi-based college project designed to manage a shared campus charging and peripheral lending kiosk.

The application integrates three primary self-service utilities:
- **CHARGE**: Paid and timed personal device charging sessions (managing port availability and coin/payment records).
- **STORE**: Secure temporary locker storage (RFID-associated session lifecycle with lock/unlock controls).
- **BORROW**: Lending and return of shared accessories (power banks, charging cables, and charger adapters) tracked by RFID and verified by camera vision.

---

## 1. Verified Implementation Status

As of October 2026, the complete repository audit and multi-branch integration on `hardware-dashboard-integration` confirms:

- **Full Backend & REST API**: Flask application with SQLite database persistence, WAL mode, foreign key enforcement, atomic database transactions, session lifecycle management, and simulation mode.
- **Hardware Abstraction Layer**: Dual-mode controllers for RFID reader (MFRC522), coin acceptor (pulse interrupt), charging relay/MOSFETs, electromagnetic locker lock, character LCD display, buzzer, and physical buttons. Full simulation mode enables complete end-to-end execution on Windows, macOS, or Linux without physical hardware.
- **ML Feature 1 (Usage Prediction & Resource Allocation)**: Chronos-2 foundation time-series forecaster and historical previous-hour baseline. Models kiosk charging session demand (target: `charge_sessions`) bounded by physical kiosk capacity (`CHARGING_PORT_CAPACITY` default 16, `POWER_BANK_BAY_CAPACITY` default 12) rather than industrial EV metrics.
- **ML Feature 2 (Camera-Based Device Verification)**: Deep convolutional neural network (MobileNetV2 transfer learning) trained on 424 Roboflow multi-class images (`charger`, `charging_cable`, `power_bank`). PyTorch checkpoint `app/ml/device_model.pth` (31 MB) is verified for CPU inference, integrated with fail-closed safety into the BORROW service.
- **Service Dashboard**: Responsive campus kiosk UI (`index.html`, `script.js`, `style.css`) serving live session counts, system status, active session controls, inventory lists, and demand forecasts.
- **Automated Test Suite**: 157 tests passing across database, charge, store, borrow, Flask API, dashboard, usage prediction, and vision classification (`157 passed in 8.87s`).

---

## 2. Architecture & Repository Structure

```
SmartHub/
├── app/
│   ├── config.py              # Dev, Testing, and Production configuration classes
│   ├── main.py                # Flask application factory and REST API endpoints
│   ├── dashboard/             # Web dashboard
│   │   ├── static/            # Frontend assets (script.js, style.css)
│   │   └── templates/         # HTML templates (index.html)
│   ├── database/              # Persistence layer
│   │   ├── db.py              # SQLite helper layer (parameterized SQL, FK, WAL)
│   │   └── schema.sql         # Database schema & initial service seed data
│   ├── hardware/              # Dual-mode (Simulation / GPIO) hardware abstractions
│   │   ├── buttons.py         # Push-button inputs
│   │   ├── buzzer.py          # Buzzer controller
│   │   ├── charging.py        # Charging relay / port controller
│   │   ├── coin.py            # Coin acceptor pulse reader
│   │   ├── lcd.py             # I2C character LCD interface
│   │   ├── lock.py            # Electromagnetic solenoid lock driver
│   │   └── rfid.py            # MFRC522 RFID reader driver
│   ├── ml/                    # Machine learning models & pipelines
│   │   ├── chronos_forecaster.py  # Chronos-2 time-series forecaster
│   │   ├── device_classifier.py   # MobileNetV2 / ResNet18 PyTorch architecture
│   │   ├── device_dataset.py      # Vision dataset loader & augmentations
│   │   ├── device_model.pth       # Trained MobileNetV2 PyTorch weights
│   │   ├── evaluate.py            # Baseline evaluation & error metrics
│   │   ├── prediction.py          # Kiosk demand prediction & allocation advice
│   │   ├── preprocess.py          # Chronological preprocessing pipeline
│   │   ├── resource_allocation.py # Capacity-bounded advice generator
│   │   ├── train_vision.py        # Vision classifier training script
│   │   └── vision.py              # Unified camera verification & fail-closed wrapper
│   └── services/              # Business logic services
│       ├── borrow.py          # Accessory lending & return with vision check
│       ├── charge.py          # Timed charging lifecycle & port capacity enforcement
│       └── store.py           # Locker storage session lifecycle
├── data/
│   ├── dataset/
│   │   └── charge_usage.csv   # Native SmartHub kiosk session dataset
│   ├── devices/               # Roboflow peripheral device image dataset (424 images)
│   │   ├── train/             # 309 training images with _classes.csv
│   │   ├── valid/             # 85 validation images with _classes.csv
│   │   └── test/              # 30 test images with _classes.csv
│   ├── processed/             # Processed datasets, metrics, and training charts
│   └── smarthub.db            # SQLite database file (generated/local)
├── tests/
│   ├── conftest.py            # Pytest fixtures (isolated temporary databases)
│   ├── test_borrow.py         # 20 BORROW service tests
│   ├── test_charge.py         # 16 CHARGE service tests
│   ├── test_dashboard.py      # 3 Dashboard static & template tests
│   ├── test_database.py       # 38 Database access layer tests
│   ├── test_flask.py          # 35 REST API route tests
│   ├── test_prediction.py     # 22 Forecaster, allocation & metric tests
│   ├── test_store.py          # 14 STORE service tests
│   └── test_vision.py         # 7 Deep learning vision & transform tests
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Unified Python dependencies
└── README.md                  # Complete documentation
```

---

## 3. Scope Clarification: Kiosk vs Industrial EV

> [!IMPORTANT]
> Early exploratory documentation in the repository referenced megawatt-scale EV plazas, transformer loading, kilowatt-hour tariffs, and battery energy storage systems (BESS).
> **SmartHub is not an industrial EV charging plaza.** It is an automated self-service college kiosk designed for smartphones, laptops, power banks, and personal electronics.
> The UrbanEV dataset (`hub_595_hourly.csv`, `volume_kwh`) is retained strictly as an external high-volume benchmark for time-series foundation model transfer research. All operational kiosk prediction and resource allocation logic strictly targets `charge_sessions` and physical kiosk capacities (`CHARGING_PORT_CAPACITY = 16`, `POWER_BANK_BAY_CAPACITY = 12`).

---

## 4. Setup & Installation

### Requirements
- **Python**: 3.11 or 3.12 (Python 3.12.5 verified)
- **OS**: Windows, macOS, or Raspberry Pi OS (Linux)

### Virtual Environment Setup

**Windows PowerShell:**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.env\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

**Raspberry Pi / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Running the Application

### Development / Simulation Mode (Default)
In simulation mode, hardware drivers simulate RFID card reads, coin pulses, LCD rendering, and lock controls in software. No physical hardware is required.

```powershell
python app/main.py
```
Or with Flask CLI:
```powershell
$env:FLASK_APP = "app.main"
flask run --host=0.0.0.0 --port=5000
```
Open `http://localhost:5000` in your web browser to access the SmartHub dashboard.

### Physical Raspberry Pi Deployment
On a configured Raspberry Pi with physical sensors connected:
```bash
export SMARTHUB_ENV=production
export SIMULATION_MODE=false
python3 app/main.py
```

---

## 6. Running Tests

The test suite runs against temporary, isolated SQLite databases. The production database is never modified during tests.

```powershell
# Run the complete test suite
python -m pytest

# Run with verbose output
python -m pytest -v

# Run individual test modules
python -m pytest tests/test_database.py
python -m pytest tests/test_vision.py
python -m pytest tests/test_prediction.py
python -m pytest tests/test_dashboard.py
```

**Verified Test Result:**
```
157 passed in 8.87s
```

---

## 7. Machine Learning Features

### Feature 1: Usage Forecasting & Smart Resource Allocation
- **Architecture**: Supports zero-shot time-series forecasting via Chronos-2 (`autogluon/chronos-2-small`) and an in-process chronological baseline (`previous-hour baseline`).
- **Target Variable**: Expected kiosk charging sessions (`charge_sessions`), bounded to non-negative counts.
- **Resource Recommendations**:
  - `HIGH` demand (≥85% port utilization): Alert busy window, recommend staging power banks as walk-away alternatives.
  - `MODERATE` demand (50%–84% port utilization): Standard rotation in charging bays.
  - `LOW` demand (<50% utilization): Initiate recharge cycles on depleted returned power banks inside kiosk bays.
- **Kiosk Capacities**: Configurable via `CHARGING_PORT_CAPACITY` (default 16 ports) and `POWER_BANK_BAY_CAPACITY` (default 12 bays).

### Feature 2: Peripheral Device Vision Classifier
- **Model Architecture**: MobileNetV2 with custom linear classification head (`Linear(1280, 256) -> BatchNorm1d -> ReLU -> Dropout -> Linear(256, 3)`).
- **Classes**:
  1. `charger` (Charger Adapter)
  2. `charging_cable` (Charging Cable)
  3. `power_bank` (Power Bank)
- **Dataset**: 424 images sourced from Roboflow:
  - Training split: 309 images
  - Validation split: 85 images
  - Test split: 30 images (21 Charger Adapters, 5 Charging Cables, 4 Power Banks)
- **Checkpoint**: `app/ml/device_model.pth` (31.0 MB) verified for CPU inference.
- **Performance Evaluation**: Evaluated test accuracy on the 30-image test set is 100% (1.0 Macro F1).
  > [!NOTE]
  > A 100% score on a small 30-image test set is a known limitation of small college datasets and does not indicate production perfection under varying campus lighting and background conditions.
- **Fail-Closed Security**: When camera verification is requested during borrow/return, an unavailable vision model or confidence below threshold (`0.60`) refuses the operation to prevent unauthorized item swaps.

---

## 8. Hardware Abstraction & Pin Mapping

Hardware abstractions in `app/hardware/` automatically detect whether physical GPIO is available. In simulation mode, operations return successful mock state and log diagnostic events.

When deployed on a Raspberry Pi:
| Component | Interface / Pins | Physical Driver |
|-----------|------------------|-----------------|
| RFID Reader | SPI (MISO, MOSI, SCK, CS: GPIO 8, RST: GPIO 25) | `mfrc522` / `SimpleMFRC522` |
| LCD Display | I2C (SDA: GPIO 2, SCL: GPIO 3, Addr: `0x27`) | `smbus2` / `RPLCD` |
| Coin Acceptor | GPIO 17 (Interrupt pulse counter) | `RPi.GPIO` |
| Locker Solenoid | GPIO 22 (Active-High Relay / MOSFET) | `RPi.GPIO` |
| Charging Relay | GPIO 27 (Active-High Relay) | `RPi.GPIO` |
| Buzzer | GPIO 18 (PWM frequency output) | `RPi.GPIO` |
| Push Buttons | GPIO 23, 24 (Internal pull-up) | `RPi.GPIO` |

---

## 9. API Reference

All API routes communicate using `application/json` with consistent HTTP status codes.

| Service | Method | Route | Description |
|---------|--------|-------|-------------|
| System | `GET` | `/api/health` | Liveness and service check |
| System | `GET` | `/api/status` | Database totals and simulation mode status |
| System | `GET` | `/api/events` | Recent audit events log |
| Users | `POST` | `/api/users` | Register user by RFID UID |
| Users | `GET` | `/api/users/<id>` | Lookup user by ID |
| Users | `GET` | `/api/users/rfid/<uid>` | Lookup user by RFID UID |
| Charge | `POST` | `/api/charge/start` | Start session and record payment transaction |
| Charge | `POST` | `/api/charge/stop` | End charging session and calculate duration |
| Charge | `GET` | `/api/charge/sessions` | List charging sessions |
| Store | `POST` | `/api/store/start` | Start locker storage session |
| Store | `POST` | `/api/store/unlock` | Disengage electronic locker lock |
| Store | `POST` | `/api/store/finish` | Re-engage electronic lock and complete session |
| Borrow | `GET` | `/api/borrow/items` | List available shared inventory |
| Borrow | `POST` | `/api/borrow/start` | Borrow item with optional camera image check |
| Borrow | `POST` | `/api/borrow/return` | Return item with atomic inventory update |
| Dashboard | `GET` | `/api/dashboard/summary` | Live metric cards for kiosk dashboard |
| Dashboard | `GET` | `/` | Web dashboard single-page interface |
| ML | `GET` | `/api/ml/usage-prediction` | Next-hour demand forecast and resource plan |
| ML | `POST` | `/api/ml/verify-item` | Multipart/form image verification |

---

## 10. Git Branch & Merge Guidance

The repository development history includes:
- `main` / `origin/main` (`9e83f8b`): Production base branch containing initial commits and merged hardware PR #1.
- `origin/feature/raspberry-pi-hardware` (`6b43006`): Early hardware and coin credit prototypes.
- `origin/backend-dev` (`1b7c6a7`): Completed Flask API, database layer, hardware abstractions, and initial 124 tests.
- `origin/feature/frontend-dashboard` (`6df5e14`): Completed web dashboard HTML, CSS, JavaScript, and asset tests.
- `origin/dashboard-ml-dev` (`e5f4e6f`): Completed ML vision classifier (MobileNetV2, 424 images, `device_model.pth`), Chronos-2 forecaster, and datasets.
- `hardware-dashboard-integration`: **Unified integration branch** uniting backend, hardware abstractions, dashboard UI, and both ML features with 157 passing tests.
