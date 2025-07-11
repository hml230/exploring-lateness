"""Preprocessing pipeline for ChronosT5 fine-tuning using official training scripts"""
from pathlib import Path

import yaml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from tqdm import tqdm


# Load configuration
CONFIG_PATH = Path.cwd().parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Load global variables
D_PATH = config["processed_path"] + "sample_50k.csv" # Your data file
COLS = ["route_variant", "suburb", "timetable_time",
        "lateness_minutes", "standing_capacity", "seated_capacity"]


# Load data via Spark
spark = SparkSession.builder.appName("ChronosPreprocessing").getOrCreate() # type: ignore
df = spark.read.csv(D_PATH, header=True, inferSchema=True)
df = df.select(*COLS).dropna()


# Convert to pandas
pd_df = (
    df.withColumn("timetable_time", F.date_format("timetable_time", "yyyy-MM-dd HH:mm:ss"))
    .toPandas()
)


# Process timestamps and create item_id
pd_df["timetable_time"] = pd.to_datetime(pd_df["timetable_time"])
pd_df["item_id"] = pd_df["route_variant"].astype(str) + "_" + pd_df["suburb"].astype(str)
pd_df.drop(["route_variant", "suburb"], axis=1, inplace=True)

# Sort by item_id and time
pd_df = pd_df.sort_values(["item_id", "timetable_time"])


# Create training records
records = []

for item_id, group in tqdm(pd_df.groupby("item_id"), desc="Chunked resampling"):
    # Set time index and resample
    group = group.set_index("timetable_time")

    try:
        # Resample to 15-minute intervals
        resampled = group["lateness_minutes"].resample("15min").mean()

        # Skip if insufficient data (lowered threshold)
        if resampled.notna().sum() < 10:
            continue

        # Fill missing values using correct pandas syntax
        target_values = resampled.ffill().fillna(0).tolist()

        records.append({
            "item_id": item_id,
            "start": resampled.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "target": target_values,
            "freq": "15min"
        })

    except Exception as e:
        continue


# Assemble DataFrame
chronos_df = pd.DataFrame.from_dict(records)  # type: ignore


# Convert start times to datetime for splitting
chronos_df['start_dt'] = pd.to_datetime(chronos_df['start'])

# Define split date (e.g., 80% for training)
split_date = chronos_df['start_dt'].quantile(0.8)

# Split the data
train_df = chronos_df[chronos_df['start_dt'] <= split_date].copy()
test_df = chronos_df[chronos_df['start_dt'] > split_date].copy()

# Remove the helper column
train_df = train_df.drop('start_dt', axis=1)
test_df = test_df.drop('start_dt', axis=1)

# Save split as Parquet
train_table = pa.Table.from_pandas(train_df)
test_table = pa.Table.from_pandas(test_df)

pq.write_table(train_table, "./chronos_train.parquet")
pq.write_table(test_table, "./chronos_test.parquet")


# Save CSV for inspection
chronos_df.to_csv("./chronos_data.csv", index=False)


print(f"Created {len(records)} time series")
spark.stop()
