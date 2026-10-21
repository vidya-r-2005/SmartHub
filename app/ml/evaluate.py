"""
Evaluation Module for SmartHub Forecasting.

Evaluates Chronos-2 and non-ML statistical baselines on chronological holdout sets,
reporting MAE, RMSE, and WAPE without future-data leakage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.chronos_forecaster import ChronosForecaster, DEFAULT_CONTEXT_LENGTH, get_forecaster

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "hub_595_hourly.csv"
EVAL_RESULTS_PATH = BASE_DIR / "data" / "processed" / "evaluation_results.json"


def compute_metrics(
    actual: Union[pd.Series, np.ndarray, List[float]],
    predicted: Union[pd.Series, np.ndarray, List[float]],
) -> Dict[str, float]:
    """Compute standard forecasting accuracy metrics: MAE, RMSE, WAPE."""
    y_true = np.array(actual, dtype=float)
    y_pred = np.array(predicted, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} actual vs {len(y_pred)} predicted")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    sum_true = float(np.sum(y_true))
    wape = float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100) if sum_true > 0 else 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape_percent": round(wape, 2),
        "sample_count": len(y_true),
    }


def evaluate_non_ml_baselines(
    df: pd.DataFrame,
    target_column: str = "volume_kwh",
) -> Dict[str, Dict[str, float]]:
    """Evaluate standard non-ML statistical benchmarks on holdout test set.
    
    Benchmarks:
    1. Lag-1 Persistence (previous hour): y_hat(t) = y(t-1)
    2. Seasonal Naive 24h (same hour yesterday): y_hat(t) = y(t-24)
    3. Seasonal Naive 168h (same hour last week): y_hat(t) = y(t-168)
    4. Moving Average (last 3 hours): y_hat(t) = mean(y(t-1), y(t-2), y(t-3))
    """
    test_mask = df["split"] == "test"
    test_indices = df.index[test_mask]

    series = df[target_column]

    predictions: Dict[str, List[float]] = {
        "lag_1_persistence": [],
        "seasonal_naive_24h": [],
        "seasonal_naive_168h": [],
        "rolling_3h_mean": [],
    }
    actuals: List[float] = []

    for idx in test_indices:
        actual_val = series.iloc[idx]
        actuals.append(actual_val)

        # 1. Lag-1
        predictions["lag_1_persistence"].append(series.iloc[idx - 1])

        # 2. 24h Lag
        lag_24 = series.iloc[idx - 24] if idx >= 24 else series.iloc[idx - 1]
        predictions["seasonal_naive_24h"].append(lag_24)

        # 3. 168h Lag
        lag_168 = series.iloc[idx - 168] if idx >= 168 else lag_24
        predictions["seasonal_naive_168h"].append(lag_168)

        # 4. 3h Moving Average
        ma3 = series.iloc[idx - 3 : idx].mean()
        predictions["rolling_3h_mean"].append(ma3)

    results: Dict[str, Dict[str, float]] = {}
    for name, preds in predictions.items():
        results[name] = compute_metrics(actuals, preds)

    return results


def evaluate_chronos_holdout(
    df: pd.DataFrame,
    forecaster: Optional[ChronosForecaster] = None,
    target_column: str = "volume_kwh",
    num_eval_steps: Optional[int] = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluate Chronos-2 on the holdout test set using rolling 1-step-ahead evaluation."""
    if forecaster is None:
        forecaster = get_forecaster()

    test_indices = df.index[df["split"] == "test"].tolist()
    if not test_indices:
        raise ValueError("No test split found in dataframe.")

    if num_eval_steps is not None and num_eval_steps > 0:
        test_indices = test_indices[:num_eval_steps]

    series = df[target_column]
    actuals: List[float] = [float(series.iloc[idx]) for idx in test_indices]
    predictions: List[float] = []

    pipeline = forecaster.load_model()

    # Check if native batched predict_quantiles is available
    if callable(getattr(pipeline, "predict_quantiles", None)):
        contexts: List[np.ndarray] = []
        for target_idx in test_indices:
            context_start = max(0, target_idx - DEFAULT_CONTEXT_LENGTH)
            c = series.iloc[context_start:target_idx].to_numpy(dtype=np.float32)
            contexts.append(c)

        # Run batched inference
        for b_start in range(0, len(contexts), batch_size):
            batch_contexts = contexts[b_start : b_start + batch_size]
            try:
                q_list, _ = pipeline.predict_quantiles(
                    batch_contexts,
                    prediction_length=1,
                    quantile_levels=[0.1, 0.5, 0.9],
                )
                for q_tensor in q_list:
                    if hasattr(q_tensor, "cpu"):
                        p50 = float(q_tensor[0, 0, 1].cpu().item())
                    else:
                        p50 = float(q_tensor[0, 0, 1])
                    predictions.append(max(0.0, p50))
            except Exception:
                # If batching fails on edge case, fall back to sequential for this batch
                for single_c in batch_contexts:
                    res = forecaster.predict_next_hour(
                        context=single_c,
                        target=target_column,
                        units="kWh" if "kwh" in target_column else "piles",
                        prediction_length=1,
                    )
                    predictions.append(res["prediction"])
    else:
        # Fallback to sequential inference (e.g. mock pipelines)
        for target_idx in test_indices:
            context_start = max(0, target_idx - DEFAULT_CONTEXT_LENGTH)
            context_slice = series.iloc[context_start:target_idx].to_numpy()
            res = forecaster.predict_next_hour(
                context=context_slice,
                target=target_column,
                units="kWh" if "kwh" in target_column else "piles",
                prediction_length=1,
            )
            predictions.append(res["prediction"])

    metrics = compute_metrics(actuals, predictions)

    return {
        "model_name": forecaster.model_name,
        "target": target_column,
        "eval_steps": len(test_indices),
        "metrics": metrics,
    }


def run_full_evaluation(
    data_path: Union[str, Path] = PROCESSED_DATA_PATH,
    run_chronos: bool = True,
    chronos_forecaster: Optional[ChronosForecaster] = None,
    num_eval_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete evaluation workflow comparing non-ML baselines and Chronos-2."""
    df = pd.read_csv(data_path)

    print(f"Loaded processed dataset from {data_path} (Total rows: {len(df)})")
    print(f"Evaluating non-ML baselines on test set (869 holdout hours)...")
    baseline_results = evaluate_non_ml_baselines(df, target_column="volume_kwh")

    output: Dict[str, Any] = {
        "dataset": Path(data_path).name,
        "target": "volume_kwh",
        "units": "kWh",
        "baselines": baseline_results,
    }

    if run_chronos:
        print("Evaluating Chronos-2 model on holdout set...")
        chronos_results = evaluate_chronos_holdout(
            df,
            forecaster=chronos_forecaster,
            target_column="volume_kwh",
            num_eval_steps=num_eval_steps,
        )
        output["chronos_2"] = chronos_results

    # Persist evaluation results
    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SmartHub forecasting models")
    parser.add_argument("--skip-chronos", action="store_true", help="Skip Chronos-2 evaluation")
    parser.add_argument("--steps", type=int, default=None, help="Limit number of test steps to evaluate")
    args = parser.parse_args()

    results = run_full_evaluation(run_chronos=not args.skip_chronos, num_eval_steps=args.steps)
    print("\n" + "=" * 70)
    print("HOLD-OUT EVALUATION BENCHMARK (869 Hours: 2023-01-23 to 2023-02-28):")
    print("=" * 70)
    for model_name, metrics in results["baselines"].items():
        print(
            f"{model_name:28} | MAE: {metrics['mae']:8.2f} kWh | "
            f"RMSE: {metrics['rmse']:8.2f} kWh | WAPE: {metrics['wape_percent']:5.2f}%"
        )
    if "chronos_2" in results:
        c_metrics = results["chronos_2"]["metrics"]
        print("-" * 70)
        print(
            f"{'Chronos-2 (autogluon/small)':28} | MAE: {c_metrics['mae']:8.2f} kWh | "
            f"RMSE: {c_metrics['rmse']:8.2f} kWh | WAPE: {c_metrics['wape_percent']:5.2f}%"
        )
    print("=" * 70)
    print(f"Results saved to {EVAL_RESULTS_PATH}")
