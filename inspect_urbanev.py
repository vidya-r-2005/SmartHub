
from pathlib import Path
import pandas as pd

# Find the cleaned UrbanEV dataset automatically
project_dir = Path(__file__).resolve().parent
dataset_dir = project_dir / "data" / "UrbanEVDataset"

folders = list(
    dataset_dir.glob("*zone-cleaned-aggregated/charge_1hour")
)

if not folders:
    print("Hourly folder not found. Check the dataset location.")
else:
    hourly_dir = folders[0]
    print("Hourly data folder:", hourly_dir)

    for file_path in sorted(hourly_dir.glob("*.csv")):
        print("\n" + "=" * 60)
        print("FILE:", file_path.name)

        try:
            df = pd.read_csv(file_path)

            print("\nFirst 5 rows:")
            print(df.head().to_string(index=False))

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nShape:", df.shape)
            print("\nData types:")
            print(df.dtypes.to_string())

        except Exception as error:
            print("Could not read file:", error)
