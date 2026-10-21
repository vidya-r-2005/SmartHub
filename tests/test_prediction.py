"""
Automated Unit Tests for SmartHub Chronos-2 Forecasting Pipeline.

These tests run without downloading foundation model weights or requiring GPU hardware,
using unit testing best practices and mocked model inference.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from pathlib import Path

from app.ml.chronos_forecaster import ChronosForecaster
from app.ml.evaluate import compute_metrics, evaluate_non_ml_baselines
from app.ml.prediction import (
    DEFAULT_KIOSK_CHARGING_PORTS,
    DEFAULT_KIOSK_POWER_BANKS,
    PROXY_DATA_LIMITATION_NOTE,
    compute_resource_allocation,
    predict_usage,
)



class TestChronosForecasterInputValidation(unittest.TestCase):
    """Test validation, missing data handling, and type normalization."""

    def setUp(self):
        self.forecaster = ChronosForecaster(model_name="autogluon/chronos-2-small")

    def test_valid_series_input(self):
        data = pd.Series([100.0 + i for i in range(50)])
        result = self.forecaster.validate_and_prepare_context(data, min_context_length=24)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 50)
        self.assertEqual(result[0], 100.0)

    def test_valid_dataframe_input(self):
        df = pd.DataFrame({
            "volume_kwh": [50.0 + i for i in range(40)],
            "other_col": [1.0] * 40,
        })
        result = self.forecaster.validate_and_prepare_context(df, target_column="volume_kwh", min_context_length=24)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 40)
        self.assertEqual(result[0], 50.0)

    def test_missing_column_dataframe_raises(self):
        df = pd.DataFrame({"some_metric": [1.0] * 30})
        with self.assertRaises(ValueError):
            self.forecaster.validate_and_prepare_context(df, target_column="volume_kwh")

    def test_insufficient_context_length_raises(self):
        short_series = pd.Series([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError) as ctx:
            self.forecaster.validate_and_prepare_context(short_series, min_context_length=24)
        self.assertIn("Insufficient context length", str(ctx.exception))

    def test_nan_handling_imputation(self):
        data_with_nans = pd.Series([10.0] * 10 + [np.nan] * 5 + [20.0] * 15)
        clean = self.forecaster.validate_and_prepare_context(data_with_nans, min_context_length=24)
        self.assertFalse(np.isnan(clean).any())
        self.assertEqual(len(clean), 30)
        # NaN elements should be filled
        self.assertEqual(clean[10], 10.0)

    def test_non_numeric_values_raises(self):
        corrupted_series = pd.Series([1.0] * 25 + ["corrupted_string"])
        with self.assertRaises(ValueError):
            self.forecaster.validate_and_prepare_context(corrupted_series, min_context_length=24)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            self.forecaster.validate_and_prepare_context("not_a_series")  # type: ignore


class TestChronosMockInference(unittest.TestCase):
    """Test forecasting function and structured response with mocked pipeline."""

    def test_structured_forecast_output(self):
        # Create a mock Chronos pipeline returning simulated sample predictions
        mock_pipeline = MagicMock()
        # Simulated forecast tensor: [batch_size=1, num_samples=20, prediction_length=1]
        simulated_samples = np.random.normal(loc=5500.0, scale=300.0, size=(1, 20, 1))
        mock_pipeline.predict.return_value = simulated_samples

        forecaster = ChronosForecaster(
            model_name="autogluon/chronos-2-small",
            pipeline=mock_pipeline,
        )

        history = [4000.0 + (i % 24) * 100 for i in range(168)]
        res = forecaster.predict_next_hour(
            context=history,
            target="volume_kwh",
            units="kWh",
            prediction_length=1,
            num_samples=20,
        )

        # Verify structured output schema
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["model_name"], "autogluon/chronos-2-small")
        self.assertEqual(res["target"], "volume_kwh")
        self.assertEqual(res["units"], "kWh")
        self.assertEqual(res["forecast_period"], "next_hour")
        self.assertEqual(res["prediction_length"], 1)
        self.assertGreater(res["prediction"], 0.0)
        self.assertGreaterEqual(res["prediction_q90"], res["prediction_q10"])
        self.assertIn("generated_at", res)
        self.assertIn("historical_context_length", res)
        self.assertEqual(res["historical_context_length"], 168)

    def test_negative_predictions_are_clipped_to_zero(self):
        # Physical energy volume cannot be negative
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([[[-50.0], [-10.0], [-20.0]]])

        forecaster = ChronosForecaster(pipeline=mock_pipeline)
        history = [10.0] * 50
        res = forecaster.predict_next_hour(history, num_samples=3)

        self.assertEqual(res["prediction"], 0.0)
        self.assertEqual(res["prediction_q10"], 0.0)


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation calculation functions and non-ML baselines."""

    def test_compute_metrics_exact(self):
        y_true = [100.0, 200.0, 300.0]
        y_pred = [110.0, 190.0, 300.0]

        # Errors: +10, -10, 0 -> MAE = (10+10+0)/3 = 6.67
        # MSE = (100 + 100 + 0)/3 = 66.67 -> RMSE = 8.16
        # WAPE = 20 / 600 * 100 = 3.33%
        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["mae"], 6.67)
        self.assertEqual(metrics["rmse"], 8.16)
        self.assertEqual(metrics["wape_percent"], 3.33)
        self.assertEqual(metrics["sample_count"], 3)

    def test_compute_metrics_length_mismatch(self):
        with self.assertRaises(ValueError):
            compute_metrics([1.0, 2.0], [1.0])

    def test_non_ml_baselines_execution(self):
        # Create minimal synthetic dataframe with train and test splits
        dates = pd.date_range("2023-01-01", periods=200, freq="1h")
        df = pd.DataFrame({
            "timestamp": dates,
            "volume_kwh": [100.0 + (i % 24) * 10 for i in range(200)],
            "split": ["train"] * 170 + ["test"] * 30,
        })

        results = evaluate_non_ml_baselines(df, target_column="volume_kwh")
        self.assertIn("lag_1_persistence", results)
        self.assertIn("seasonal_naive_24h", results)
        self.assertIn("seasonal_naive_168h", results)
        self.assertIn("rolling_3h_mean", results)
        self.assertGreater(results["lag_1_persistence"]["sample_count"], 0)


class TestProcessedDataSchema(unittest.TestCase):
    """Verify integrity of preprocessed dataset."""

    def test_hub_595_processed_file(self):
        data_path = Path("data/processed/hub_595_hourly.csv")
        self.assertTrue(data_path.exists(), "Processed hub file does not exist")

        df = pd.read_csv(data_path)
        self.assertEqual(len(df), 4344)
        required_cols = ["timestamp", "volume_kwh", "occupancy_piles", "duration_hours", "e_price", "s_price", "split"]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Check no missing values
        self.assertFalse(df[required_cols].isna().any().any())

        # Check split proportions
        train_count = (df["split"] == "train").sum()
        test_count = (df["split"] == "test").sum()
        self.assertEqual(train_count, 3475)
        self.assertEqual(test_count, 869)


class TestSmartHubResourceAllocation(unittest.TestCase):
    """Test SmartHub kiosk resource allocation logic, port availability, and power-bank prep."""

    def test_high_demand_kiosk_allocation(self):
        plan = compute_resource_allocation(
            predicted_sessions=15.0,
            total_ports=16,
            total_power_banks=12,
        )
        self.assertEqual(plan["demand_level"], "HIGH")
        self.assertEqual(plan["status_color"], "red")
        self.assertGreaterEqual(plan["estimated_port_utilization_percent"], 85.0)
        self.assertLessEqual(plan["available_ports_expected"], 2)

        # Power bank preparation
        pb = plan["power_bank_preparation"]
        self.assertEqual(pb["status"], "DISPENSE_READY")
        self.assertGreater(pb["recommended_staged_units"], 0)
        self.assertLessEqual(pb["recommended_staged_units"], 12)

        # Ensure no EV grid / BESS terms appear in recommendations
        all_recs = " ".join(plan["recommendations"]).lower()
        self.assertNotIn("bess", all_recs)
        self.assertNotIn("substation", all_recs)
        self.assertNotIn("transformer", all_recs)
        self.assertNotIn("tariff", all_recs)
        self.assertIn("power bank", all_recs)

    def test_moderate_demand_kiosk_allocation(self):
        plan = compute_resource_allocation(
            predicted_sessions=10.0,
            total_ports=16,
            total_power_banks=12,
        )
        self.assertEqual(plan["demand_level"], "MODERATE")
        self.assertEqual(plan["status_color"], "amber")
        self.assertTrue(50.0 <= plan["estimated_port_utilization_percent"] < 85.0)
        self.assertEqual(plan["available_ports_expected"], 6)
        self.assertEqual(plan["power_bank_preparation"]["status"], "STANDARD_ROTATION")

    def test_low_demand_kiosk_allocation(self):
        plan = compute_resource_allocation(
            predicted_sessions=4.0,
            total_ports=16,
            total_power_banks=12,
        )
        self.assertEqual(plan["demand_level"], "LOW")
        self.assertEqual(plan["status_color"], "green")
        self.assertLess(plan["estimated_port_utilization_percent"], 50.0)
        self.assertEqual(plan["available_ports_expected"], 12)

        # Low demand should trigger recharging depleted returned units
        pb = plan["power_bank_preparation"]
        self.assertEqual(pb["status"], "CYCLE_RECHARGE")
        self.assertEqual(pb["recommended_staged_units"], 0)

    def test_respects_actual_kiosk_capacity(self):
        # 18 sessions on a 16-port kiosk (over-capacity)
        plan = compute_resource_allocation(
            predicted_sessions=18.0,
            total_ports=16,
            total_power_banks=12,
        )
        # Available ports cannot be negative
        self.assertEqual(plan["available_ports_expected"], 0)
        self.assertEqual(plan["total_charging_ports"], 16)
        # Power bank staging capped at total power bank capacity
        self.assertLessEqual(plan["power_bank_preparation"]["recommended_staged_units"], 12)

    def test_proxy_benchmark_mode_includes_caveat(self):
        plan = compute_resource_allocation(
            predicted_volume_kwh=6500.0,
            total_ports=16,
        )
        self.assertTrue(plan["is_proxy_data"])
        self.assertIsNotNone(plan["proxy_data_limitations"])
        self.assertIn("UrbanEV", plan["proxy_data_limitations"])
        # Should not equate 6500 kWh with 6500 phone sessions
        self.assertLessEqual(plan["predicted_charging_sessions"], 16)

    def test_predict_usage_with_mock_forecaster(self):
        mock_pipeline = MagicMock()
        mock_pipeline.predict.return_value = np.array([[[14.0], [14.0], [14.0]]])
        forecaster = ChronosForecaster(pipeline=mock_pipeline)

        context = [10.0] * 50
        res = predict_usage(
            context_data=context,
            target="charge_sessions",
            total_ports=16,
            total_power_banks=12,
            forecaster=forecaster,
        )
        self.assertIn("forecast", res)
        self.assertIn("resource_plan", res)
        self.assertEqual(res["resource_plan"]["demand_level"], "HIGH")
        self.assertEqual(res["resource_plan"]["power_bank_preparation"]["status"], "DISPENSE_READY")


if __name__ == "__main__":
    unittest.main()



# ---------------------------------------------------------------------------
# Direct integration tests for baseline demand and resource allocation
# ---------------------------------------------------------------------------

from app.ml.prediction import predict_next_hour_demand
from app.ml.resource_allocation import recommend_resources


def test_previous_hour_baseline_is_one_hour_and_capacity_bounded():
    result = predict_next_hour_demand([
        {"started_at": "2026-10-21T09:05:00+00:00", "duration_seconds": 60},
        {"started_at": "2026-10-21T10:05:00+00:00", "duration_seconds": 60},
        {"started_at": "2026-10-21T10:25:00+00:00", "duration_seconds": 60},
    ])
    assert result["available"] is True
    assert result["model"] == "previous-hour baseline"
    assert result["forecast_horizon_hours"] == 1
    assert result["predicted_demand"] == 2.0
    assert result["prediction_interval"] is None
    assert result["resource_allocation"]["expected_vacant_ports"] == 14


def test_invalid_or_insufficient_history_is_unavailable():
    result = predict_next_hour_demand([{"started_at": "not-a-date"}])
    assert result["available"] is False


def test_resource_recommendation_never_exceeds_capacity():
    result = recommend_resources(30, charging_port_capacity=16, power_bank_bay_capacity=12)
    assert result["expected_occupied_ports"] == 16
    assert result["expected_vacant_ports"] == 0
    assert result["recommended_ready_power_banks"] == 12
