"""Extract predictions from the pre-trained models"""
import ast

import pandas as pd
import torch
from chronos import BaseChronosPipeline


MAX_CONTEXT = 48  # 72, 48, etc.

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)


df = pd.read_csv(
    "./chronos_data.csv"
)


df['target_parsed'] = df['target'].apply(ast.literal_eval)
contexts = [torch.tensor(row[-MAX_CONTEXT:], dtype=torch.float32) for row in df['target_parsed']]


quantiles, mean = pipeline.predict_quantiles(
    context=contexts,
    prediction_length=12,  # predict next 12 intervals (3 hours)
    quantile_levels=[0.1, 0.5, 0.9],
)


# Build DataFrame
rows = []
for i in range(10):
    item_id = df['item_id'].iloc[i]
    for t in range(12):
        rows.append({
            "item_id": item_id,
            "step": t + 1,
            "q10": float(quantiles[i, t, 0]),
            "q50": float(quantiles[i, t, 1]),
            "q90": float(quantiles[i, t, 2]),
        })

forecast_df = pd.DataFrame(rows)

# Save to CSV
forecast_df.to_csv("zero-shot-forsecast.csv", index=False)
