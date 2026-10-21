"""
UrbanEV Dataset Preprocessing Module for SmartHub.

Processes raw UrbanEV hourly aggregated data into clean chronological time series
for Chronos-2 forecasting and smart resource allocation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd


# Default project directories
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = (
    BASE_DIR
    / "data"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "charge_1hour"
)
ZONE_INFO_PATH = (
    BASE_DIR
    / "data"
    / "UrbanEVDataset"
    / "20220901-20230228_zone-cleaned-aggregated"
    / "zone-information.csv"
)
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Constants & Metadata
PRIMARY_ZONE = "595"  # High-activity major charging hub (274 piles)
PRIMARY_TARGET = "volume_kwh"
SECONDARY_METRIC = "occupancy_piles"

TARGET_DOCUMENTATION = {
    "dataset_role": "External automotive charging benchmark used as transfer proxy for time-series forecasting. Does NOT represent local SmartHub kiosk device-charging sessions.",
    "primary_target": {
        "name": "volume_kwh",
        "description": "Total electric energy dispensed to vehicles during the 1-hour window.",
        "units": "Kilowatt-hours (kWh)",
        "interpretation": "Electrical energy demand / grid load. Does NOT measure SmartHub device session counts.",
        "limitations": "Automotive EV plaza data. Influenced by commercial EV charger power capacities (fast DC vs slow AC).",
    },
    "secondary_metric": {
        "name": "occupancy_piles",
        "description": "Average number of concurrently occupied charging piles during the 1-hour window.",
        "units": "Active occupied chargers (piles)",
        "interpretation": "Infrastructure bay utilization. Does NOT measure SmartHub device session arrival counts.",
        "limitations": "Saturates at maximum zone pile capacity (274 piles for Zone 595).",
    },
    "auxiliary_features": {
        "duration_hours": "Cumulative vehicle plug-in duration (hours) across active piles.",
        "e_price": "Electricity tariff during the hour (Yuan/kWh).",
        "s_price": "Charging service fee during the hour (Yuan/kWh).",
    },
}



def load_raw_hourly_tables(
    raw_dir: Union[str, Path] = RAW_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """Load the 5 hourly CSV tables from the UrbanEV aggregated dataset."""
    raw_path = Path(raw_dir)
    required_files = {
        "volume": "volume.csv",
        "occupancy": "occupancy.csv",
        "duration": "duration.csv",
        "e_price": "e_price.csv",
        "s_price": "s_price.csv",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for key, filename in required_files.items():
        filepath = raw_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required UrbanEV table not found: {filepath}. "
                f"Verify that UrbanEV dataset exists under data/UrbanEVDataset/."
            )
        df = pd.read_csv(filepath)
        if "time" not in df.columns:
            raise ValueError(f"Missing 'time' column in {filepath}")
        tables[key] = df

    return tables


def validate_timestamps(time_series: pd.Series) -> pd.DatetimeIndex:
    """Validate that the timestamp series is sorted, has consistent 1-hour intervals, and has no duplicates."""
    dt_index = pd.DatetimeIndex(pd.to_datetime(time_series))

    if not dt_index.is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly monotonically increasing.")

    if dt_index.duplicated().any():
        raise ValueError("Duplicate timestamps found in dataset.")

    expected_range = pd.date_range(
        start=dt_index.min(),
        end=dt_index.max(),
        freq="1h",
    )
    if len(expected_range) != len(dt_index):
        raise ValueError(
            f"Timestamp continuity mismatch: expected {len(expected_range)} steps, "
            f"found {len(dt_index)} steps. Interval is not consistently 1h."
        )

    return dt_index



def build_zone_series(
    tables: Dict[str, pd.DataFrame],
    zone_id: str = PRIMARY_ZONE,
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """Build a clean chronological time series for a specific zone/hub.
    
    Ensures zero future leakage by performing a strict chronological split.
    """
    zone_str = str(zone_id)
    for name, df in tables.items():
        if zone_str not in df.columns:
            raise KeyError(f"Zone ID '{zone_str}' not found in {name}.csv columns.")

    timestamps = validate_timestamps(tables["volume"]["time"])

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "volume_kwh": tables["volume"][zone_str].astype(float),
            "occupancy_piles": tables["occupancy"][zone_str].astype(float),
            "duration_hours": tables["duration"][zone_str].astype(float),
            "e_price": tables["e_price"][zone_str].astype(float),
            "s_price": tables["s_price"][zone_str].astype(float),
        }
    )

    # Add temporal calendar covariates
    df["hour"] = timestamps.hour
    df["day_of_week"] = timestamps.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


    # Missing value verification & handling
    numeric_cols = [
        "volume_kwh",
        "occupancy_piles",
        "duration_hours",
        "e_price",
        "s_price",
    ]
    if df[numeric_cols].isna().any().any():
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Chronological train/test holdout assignment (no future leakage)
    split_idx = int(len(df) * train_ratio)
    df["split"] = ["train"] * split_idx + ["test"] * (len(df) - split_idx)

    return df


def build_citywide_series(
    tables: Dict[str, pd.DataFrame],
    train_ratio: float = 0.8,
) -> pd.DataFrame:
    """Build a clean chronological citywide aggregated time series across all 275 zones."""
    timestamps = validate_timestamps(tables["volume"]["time"])
    zone_cols = [c for c in tables["volume"].columns if c != "time"]

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "volume_kwh": tables["volume"][zone_cols].astype(float).sum(axis=1),
            "occupancy_piles": tables["occupancy"][zone_cols].astype(float).sum(axis=1),
            "duration_hours": tables["duration"][zone_cols].astype(float).sum(axis=1),
            "e_price": tables["e_price"][zone_cols].astype(float).mean(axis=1),
            "s_price": tables["s_price"][zone_cols].astype(float).mean(axis=1),
        }
    )

    df["hour"] = timestamps.hour
    df["day_of_week"] = timestamps.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


    split_idx = int(len(df) * train_ratio)
    df["split"] = ["train"] * split_idx + ["test"] * (len(df) - split_idx)

    return df


def preprocess_and_save(
    raw_dir: Union[str, Path] = RAW_DATA_DIR,
    output_dir: Union[str, Path] = PROCESSED_DIR,
    zone_id: str = PRIMARY_ZONE,
    train_ratio: float = 0.8,
) -> Tuple[Path, Path]:
    """Run full preprocessing pipeline and persist clean CSVs and metadata."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    tables = load_raw_hourly_tables(raw_dir)

    # 1. Primary Hub Series
    hub_df = build_zone_series(tables, zone_id=zone_id, train_ratio=train_ratio)
    hub_file = out_path / f"hub_{zone_id}_hourly.csv"
    hub_df.to_csv(hub_file, index=False)

    # 2. Citywide Aggregated Series
    city_df = build_citywide_series(tables, train_ratio=train_ratio)
    city_file = out_path / "citywide_hourly.csv"
    city_df.to_csv(city_file, index=False)

    # 3. Save Metadata
    split_idx = int(len(hub_df) * train_ratio)
    split_timestamp = hub_df.iloc[split_idx]["timestamp"].isoformat()

    metadata = {
        "dataset_name": "UrbanEV Preprocessed Hourly Benchmark",
        "source": "UrbanEV 20220901-20230228_zone-cleaned-aggregated",
        "primary_zone_id": zone_id,
        "total_records": len(hub_df),
        "start_time": hub_df["timestamp"].iloc[0].isoformat(),
        "end_time": hub_df["timestamp"].iloc[-1].isoformat(),
        "sampling_frequency": "1 hour",
        "chronological_split": {
            "train_ratio": train_ratio,
            "train_records": split_idx,
            "test_records": len(hub_df) - split_idx,
            "split_timestamp": split_timestamp,
        },
        "target_definitions": TARGET_DOCUMENTATION,
        "files_generated": [hub_file.name, city_file.name],
    }

    meta_file = out_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return hub_file, city_file


if __name__ == "__main__":
    print("=" * 60)
    print("SMARTHUB - URBANEV DATASET PREPROCESSING")
    print("=" * 60)
    hub_path, city_path = preprocess_and_save()
    print(f"Primary hub processed file saved to: {hub_path}")
    print(f"Citywide processed file saved to:   {city_path}")
    print("Metadata written to:                data/processed/metadata.json")
    print("Preprocessing completed successfully.")
