"""Zero-shot inference on unprocessed .csv data, use only for prototyping"""
from pathlib import Path

import yaml
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from sklearn.preprocessing import StandardScaler


CONFIG_PATH = Path.cwd().parent / "config.yaml"


with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)


D_PATH =  config["processed_path"] + "sample_50k.csv"
df = pd.read_csv(D_PATH)
df = df.sort_values(by="timetable_time")


# Normalise lateness_minutes
scaler = StandardScaler()
lateness = scaler.fit_transform(df[["lateness_minutes"]].values).flatten()


# Take the most recent sequence (last 96 timesteps for context)
CONTEXT_LEN = 96
CONTEXT_SERIES = lateness[-CONTEXT_LEN:]
CONTEXT_TENSOR = torch.tensor(CONTEXT_SERIES, dtype=torch.float32).unsqueeze(0)


# Load
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda",
    torch_dtype=torch.float32,
)

# Predict
quantiles, mean = pipeline.predict_quantiles(
    context=CONTEXT_TENSOR,
    prediction_length=12,
    quantile_levels=[0.1, 0.5, 0.9],
)

# Inverse transform to original scale
mean_pred = scaler.inverse_transform(mean.squeeze(0).cpu().numpy().reshape(-1, 1)).flatten()


# Display prediction
with open("zero_shot_results.txt", "w", encoding="utf-8") as f:
    f.writelines(["Mean prediction (lateness_minutes) for next 12 trips: \n", str(mean_pred)])
