# SmartHub

SmartHub is an intelligent device charging and peripheral lending platform designed for kiosks and charging hubs.

## Feature 1: Usage Prediction & Kiosk Resource Allocation using Chronos-2

This feature provides zero-shot time-series forecasting of charging demand using the **Chronos-2** foundation model (`autogluon/chronos-2-small` / `amazon/chronos-2`). Predictions directly inform SmartHub kiosk operational decisions:
- **Next-Hour Charging-Session Demand:** Forecasted device-charging arrivals.
- **Charging-Port Availability:** Port congestion management across physical kiosk charging ports (default: 16 ports).
- **Power-Bank Preparation:** Dynamic staging of fully charged power banks in the dispenser to serve overflow users during high-demand peak windows.

### Target Definitions & Data Sources

1. **Native SmartHub Kiosk Sessions: `charge_sessions` (Primary Operational Target)**
   - **Source:** [`data/dataset/charge_usage.csv`](file:///c:/Users/2005v/OneDrive/Desktop/SmartHub/data/dataset/charge_usage.csv)
   - **Metric:** Discrete user device-charging sessions per hour (operating range: 3–22 sessions/hr, mean: ~12.4).
   - **Units:** `sessions`
   - **Operational Purpose:** Kiosk port capacity planning, dispenser power-bank staging, and queue mitigation.

2. **External Transfer Benchmark: `volume_kwh` & `occupancy_piles` (Proxy Dataset)**
   - **Source:** UrbanEV benchmark ([`data/processed/hub_595_hourly.csv`](file:///c:/Users/2005v/OneDrive/Desktop/SmartHub/data/processed/hub_595_hourly.csv))
   - **Units:** `kWh` (electrical energy load) and `piles` (active commercial EV bays).
   - **Proxy Limitations:** UrbanEV measures automotive charging plaza load. It is used strictly as a high-resolution time-series transfer benchmark for Chronos-2 foundation models; automotive energy and pile metrics are not conflated with SmartHub kiosk device-charging sessions.

---

## Environment Setup

PyTorch Windows wheels and dependencies are managed inside a dedicated virtual environment (`.venv-chronos`):

### 1. Activate Virtual Environment
```powershell
.\.venv-chronos\Scripts\Activate.ps1
```

### 2. Dependencies
```powershell
# Chronos-2, PyTorch CPU, and Computer Vision packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate chronos-forecasting pandas numpy scipy scikit-learn pillow matplotlib
```

---

## Usage Instructions

### 1. Predict Kiosk Demand & Generate Resource Plan (Python)
```python
from app.ml.prediction import predict_usage

# Forecast next-hour charging sessions on native kiosk data
result = predict_usage(target="charge_sessions", total_ports=16, total_power_banks=12)

print(result["forecast"]["prediction"])  # Expected next-hour sessions (e.g. 14.5)
print(result["resource_plan"]["demand_level"])  # "HIGH", "MODERATE", or "LOW"
print(result["resource_plan"]["power_bank_preparation"])
# {
#   "status": "DISPENSE_READY",
#   "recommended_staged_units": 4,
#   "action": "Stage 4 fully charged power banks in kiosk dispenser..."
# }
```

### 2. External Benchmark Evaluation (Holdout Test)
Evaluates Chronos-2 on the 869-hour UrbanEV transfer holdout set (`2023-01-23 19:00` to `2023-02-28 23:00`):
```powershell
.\.venv-chronos\Scripts\python.exe app/ml/evaluate.py
```

#### Holdout Test Benchmark Results (869 Hours)
| Model / Benchmark | MAE (kWh) | RMSE (kWh) | WAPE (%) |
|---|---|---|---|
| **Chronos-2 (`autogluon/chronos-2-small`)** | **462.02** | **622.50** | **8.12%** |
| Seasonal Naive (24h lag) | 747.32 | 964.86 | 13.14% |
| Lag-1 Persistence | 860.19 | 1,201.24 | 15.13% |
| Rolling 3h Moving Average | 1,298.62 | 1,767.90 | 22.83% |
| Seasonal Naive (168h lag) | 1,574.18 | 2,326.01 | 27.68% |

### 3. Automated Unit Tests
```powershell
.\.venv-chronos\Scripts\python.exe -m unittest tests/test_prediction.py
```
*(All 19 unit tests pass)*

---


## Feature 2: Peripheral Device Verification & Classification (Vision)

This feature provides high-accuracy automated computer vision verification and classification for SmartHub hardware peripherals (`charger`, `charging_cable`, `power_bank`). It is used in charging bay automated inspection, kiosk item borrowing/return verification, and inventory auditing.

### Dataset Overview
- **Source:** Roboflow multiclass classification crops dataset (`data/devices/`)
- **Total Images:** 424 images (512x512 RGB)
  - **Train split:** 309 images (`charger`: 146, `charging_cable`: 134, `power_bank`: 29)
  - **Validation split:** 85 images (`charger`: 41, `charging_cable`: 36, `power_bank`: 8)
  - **Test split (holdout):** 30 images (`charger`: 21, `charging_cable`: 5, `power_bank`: 4)
- **Class Imbalance Strategy:** Loss-weighted Cross-Entropy ($w_{power\_bank} \approx 2.12$, $w_{cable} \approx 0.46$, $w_{charger} \approx 0.42$) with label smoothing.

### Model Architecture
- **Backbone:** Pretrained `MobileNetV2` with ImageNet transfer learning (fast, lightweight ~14MB, ideal for edge and CPU inference).
- **Classification Head:**
  ```text
  Dropout(0.3) -> Linear(1280, 256) -> BatchNorm1d(256) -> ReLU() -> Dropout(0.2) -> Linear(256, 3)
  ```
- **Checkpoint Location:** `app/ml/device_model.pth`

### Holdout Test Benchmark Results (30 Holdout Images)
| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Charger Adapter (`charger`)** | 1.00 | 1.00 | 1.00 | 21 |
| **Charging Cable (`charging_cable`)** | 1.00 | 1.00 | 1.00 | 5 |
| **Power Bank (`power_bank`)** | 1.00 | 1.00 | 1.00 | 4 |
| **Overall Accuracy** | | | **100.00%** | 30 |
| **Macro Average F1** | 1.00 | 1.00 | **1.0000** | 30 |

Generated plots saved in `data/processed/`:
- `data/processed/device_confusion_matrix.png`
- `data/processed/device_training_history.png`
- `data/processed/device_vision_metrics.json`

### Usage Instructions

#### 1. Train or Re-train the Model
```powershell
.\.venv-chronos\Scripts\python.exe app/ml/train_vision.py --epochs 12 --batch-size 16
```

#### 2. Classify an Image (Python)
```python
from app.ml.vision import classify_device

result = classify_device("data/devices/test/sample.jpg")
print(result)
# {
#   "status": "success",
#   "predicted_class": "power_bank",
#   "class_label": "Power Bank",
#   "confidence": 0.982,
#   "probabilities": {"charger": 0.005, "charging_cable": 0.013, "power_bank": 0.982},
#   "inference_latency_ms": 14.8
# }
```

#### 3. Verify Returned or Borrowed Device (Python)
```python
from app.ml.vision import verify_item

# Returns structured verification with confidence threshold check (default 0.60)
verification = verify_item("data/devices/test/sample.jpg", expected_item="powerbank")
print(verification["verified"])  # True / False
print(verification["message"])   # "Item verified successfully as Power Bank (Confidence: 98.2%)"
```

#### 4. Automated Tests
```powershell
.\.venv-chronos\Scripts\python.exe -m unittest tests/test_vision.py
```
*(All 7 unit tests pass)*