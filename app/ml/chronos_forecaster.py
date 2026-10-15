"""
Modular Chronos-2 Forecasting Module for SmartHub.

Provides zero-shot time-series forecasting for charging volume (kWh) and occupancy (piles)
using Chronos-2 foundation models.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# Default Configuration
DEFAULT_MODEL_NAME = "autogluon/chronos-2-small"
DEFAULT_TARGET = "volume_kwh"
DEFAULT_UNITS = "kWh"
DEFAULT_CONTEXT_LENGTH = 168  # 1 week of hourly context


class ChronosForecaster:
    """Modular forecaster for SmartHub utilizing Chronos-2 time-series models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device_map: str = "cpu",
        pipeline: Optional[Any] = None,
    ) -> None:
        """Initialize the forecaster.
        
        Args:
            model_name: Hugging Face model repository identifier.
            device_map: Execution device ('cpu' or 'cuda').
            pipeline: Optional pre-loaded pipeline instance (useful for dependency injection and mocking).
        """
        self.model_name = model_name
        self.device_map = device_map
        self._pipeline = pipeline
        self._model_loaded: bool = pipeline is not None

    def load_model(self) -> Any:
        """Load model pipeline once. Reuses loaded instance where practical."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            # Import dynamically to allow mock testing without hard torch/chronos import errors
            from chronos import BaseChronosPipeline  # type: ignore
            import torch  # type: ignore

            torch_dtype = torch.float32 if self.device_map == "cpu" else torch.bfloat16
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
            )
            self._model_loaded = True
            return self._pipeline
        except ImportError as err:
            raise RuntimeError(
                f"Required dependency not installed for Chronos-2: {err}. "
                "Ensure torch, transformers, and chronos-forecasting are installed in .venv-chronos."
            ) from err
        except Exception as err:
            raise RuntimeError(
                f"Failed to load Chronos-2 weights '{self.model_name}': {err}. "
                "Check network connection or Hugging Face cache."
            ) from err

    @property
    def is_loaded(self) -> bool:
        """Return True if model pipeline is already loaded into memory."""
        return self._model_loaded

    def validate_and_prepare_context(
        self,
        context_data: Union[pd.Series, pd.DataFrame, List[float], np.ndarray],
        target_column: Optional[str] = None,
        min_context_length: int = 24,
    ) -> np.ndarray:
        """Validate input time series, handle missing values, and return clean 1D float array."""
        if isinstance(context_data, pd.DataFrame):
            col = target_column or DEFAULT_TARGET
            if col not in context_data.columns:
                raise ValueError(
                    f"Target column '{col}' not found in input DataFrame columns: {list(context_data.columns)}"
                )
            series = context_data[col].copy()
        elif isinstance(context_data, pd.Series):
            series = context_data.copy()
        elif isinstance(context_data, list):
            series = pd.Series(context_data, dtype=float)
        elif isinstance(context_data, np.ndarray):
            series = pd.Series(context_data.flatten().tolist(), dtype=float)
        else:
            raise TypeError(
                f"Unsupported context data type: {type(context_data)}. "
                "Expected pandas Series, DataFrame, numpy array, or list of numbers."
            )

        if len(series) < min_context_length:
            raise ValueError(
                f"Insufficient context length ({len(series)} steps). "
                f"At least {min_context_length} hourly steps required for reliable forecasting."
            )

        # Handle missing values via forward/backward fill
        if series.isna().any():
            series = series.ffill().bfill()
            if series.isna().any():
                raise ValueError("Context data contains unresolvable NaN values.")

        values: np.ndarray = np.asarray(pd.to_numeric(series, errors="coerce"), dtype=float)
        if np.isnan(values).any():
            raise ValueError("Context data contains non-numeric values that could not be parsed.")

        return values


    def predict_next_hour(
        self,
        context: Union[pd.Series, pd.DataFrame, List[float], np.ndarray],
        target: str = DEFAULT_TARGET,
        units: str = DEFAULT_UNITS,
        prediction_length: int = 1,
        num_samples: int = 20,
    ) -> Dict[str, Any]:
        """Generate structured forecast for the next hour(s) using Chronos-2.
        
        Args:
            context: Historical observations up to the forecast origin.
            target: Target variable name (e.g. 'volume_kwh' or 'occupancy_piles').
            units: Unit of measurement ('kWh' or 'piles').
            prediction_length: Forecast horizon (default 1 for next-hour).
            num_samples: Number of trajectory samples for quantile estimation.

        Returns:
            Structured dictionary with forecast period, prediction, quantiles, and metadata.
        """
        if prediction_length < 1:
            raise ValueError(f"prediction_length must be >= 1, got {prediction_length}")

        clean_context = self.validate_and_prepare_context(context, target_column=target)
        
        # Use recent context window to prevent unnecessary memory consumption
        recent_context = clean_context[-DEFAULT_CONTEXT_LENGTH:]

        pipeline = self.load_model()

        p10: float = 0.0
        p50: float = 0.0
        p90: float = 0.0
        mean_pred: float = 0.0

        try:
            used_quantiles = False

            if callable(getattr(pipeline, "predict_quantiles", None)):
                context_input = [np.asarray(recent_context, dtype=np.float32)]
                try:
                    res = pipeline.predict_quantiles(
                        context_input,
                        prediction_length=prediction_length,
                        quantile_levels=[0.1, 0.5, 0.9],
                    )
                    if isinstance(res, (tuple, list)) and len(res) == 2:
                        quantiles_list, mean_list = res
                        q_tensor = quantiles_list[0]
                        mean_tensor = mean_list[0]

                        if hasattr(q_tensor, "cpu"):
                            q_arr = q_tensor.cpu().numpy()
                            mean_arr = mean_tensor.cpu().numpy()
                        else:
                            q_arr = np.asarray(q_tensor)
                            mean_arr = np.asarray(mean_tensor)

                        # q_arr shape: (n_variates, prediction_length, n_quantiles)
                        p10 = float(q_arr[0, 0, 0])
                        p50 = float(q_arr[0, 0, 1])
                        p90 = float(q_arr[0, 0, 2])
                        # mean_arr shape: (n_variates, prediction_length)
                        mean_pred = float(mean_arr[0, 0]) if mean_arr.ndim > 1 else float(mean_arr[0])
                        used_quantiles = True
                except (TypeError, ValueError):
                    used_quantiles = False

            if not used_quantiles:
                if hasattr(pipeline, "predict"):
                    # Sample-based Chronos-1 or mock interface
                    try:
                        import torch  # type: ignore
                        context_tensor = torch.tensor(recent_context, dtype=torch.float32).unsqueeze(0)
                        has_torch = True
                    except ImportError:
                        context_tensor = np.expand_dims(recent_context, axis=0)
                        has_torch = False

                    try:
                        forecast_samples = pipeline.predict(
                            context_tensor,
                            prediction_length=prediction_length,
                            num_samples=num_samples,
                        )
                    except TypeError:
                        forecast_samples = pipeline.predict(
                            [np.asarray(recent_context, dtype=np.float32)],
                            prediction_length=prediction_length,
                        )

                    if has_torch and hasattr(forecast_samples, "cpu"):
                        samples_arr = forecast_samples[0].cpu().numpy()
                    elif isinstance(forecast_samples, np.ndarray):
                        samples_arr = forecast_samples[0]
                    else:
                        samples_arr = np.asarray(forecast_samples)[0]

                    p10 = float(np.percentile(samples_arr[:, 0], 10))
                    p50 = float(np.percentile(samples_arr[:, 0], 50))
                    p90 = float(np.percentile(samples_arr[:, 0], 90))
                    mean_pred = float(np.mean(samples_arr[:, 0]))
                else:
                    raise AttributeError("Pipeline object does not expose a predict_quantiles() or predict() method.")

            # Post-process: Physical demand (volume/occupancy) cannot be negative
            point_prediction = max(0.0, p50)
            p10 = max(0.0, p10)
            p90 = max(0.0, p90)

        except Exception as inference_err:
            raise RuntimeError(
                f"Inference error during Chronos-2 prediction: {inference_err}"
            ) from inference_err

        return {
            "status": "success",
            "model_name": self.model_name,
            "target": target,
            "units": units,
            "forecast_period": "next_hour" if prediction_length == 1 else f"next_{prediction_length}_hours",
            "prediction_length": prediction_length,
            "prediction": round(point_prediction, 2),
            "prediction_mean": round(mean_pred, 2),
            "prediction_q10": round(p10, 2),
            "prediction_q90": round(p90, 2),
            "historical_context_length": len(recent_context),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# Module-level cached instance for efficient reuse
_GLOBAL_FORECASTER: Optional[ChronosForecaster] = None


def get_forecaster(
    model_name: str = DEFAULT_MODEL_NAME,
    device_map: str = "cpu",
) -> ChronosForecaster:
    """Retrieve or create the global cached ChronosForecaster instance."""
    global _GLOBAL_FORECASTER
    if _GLOBAL_FORECASTER is None or _GLOBAL_FORECASTER.model_name != model_name:
        _GLOBAL_FORECASTER = ChronosForecaster(model_name=model_name, device_map=device_map)
    return _GLOBAL_FORECASTER


def predict_next_hour_demand(
    context: Union[pd.Series, pd.DataFrame, List[float], np.ndarray],
    target: str = DEFAULT_TARGET,
    units: str = DEFAULT_UNITS,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    """Convenience helper to predict next-hour demand with structured output."""
    forecaster = get_forecaster(model_name=model_name)
    return forecaster.predict_next_hour(context=context, target=target, units=units)
