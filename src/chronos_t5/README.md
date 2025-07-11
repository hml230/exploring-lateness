# ChronosT5 Fine-tuning for Bus Lateness Prediction

This directory contains a preprocessing pipeline for fine-tuning ChronosT5 on bus lateness prediction data.

## Overview

ChronosT5-tiny is an open-source time series forecasting model that can be fine-tuned on domain-specific data. This directory preprocesses bus schedule and lateness data into the format required for ChronosT5 training.

## Dataset Format

The pipeline transforms raw bus data into time series format suitable for ChronosT5:

### Input Data Structure

```text
route_variant, suburb, timetable_time, lateness_minutes, standing_capacity, seated_capacity
```

### Output Data Structure

```text
item_id, start, target, freq
0037-7_raymond terrace, 2016-08-09 15:30:00, [12.0, 8.5, 3.0, ...], 15min
```

## Pipeline Components

### Data Processing Steps

1. **Load**: Loads CSV data using Spark for efficient processing of large datasets

2. **Preprocessing**: Cleans data by removing null values and formatting timestamps

3. **Time Series Creation**: Groups data by route-suburb combinations to create individual time series

4. **Resampling**: Converts unqueal bus schedule observations to 15-minute intervals

5. **Output Generation**: Exports data in Parquet format for ChronosT5 training

## Configuration

Create a `config.yaml` file in your root directory for paths to data:

```yaml
processed_path: "/path/to/your/data/"
```

## Usage

### Running the Pipeline

```bash
python chronos_preprocessing.py
```

### Output Files

- `chronos_train.parquet`: Main training data in Parquet format

- `chronos_test.parquet`: Main test data in Parquet format

- `chronos_data.csv`: Human-readable CSV for inspection

## Data Schema

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | String | Unique identifier combining route and suburb |
| `start` | String | Start timestamp in ISO format |
| `target` | List[Float] | Time series values (lateness in minutes) |
| `freq` | String | Frequency of observations (15min) |

## Fine-tuning Parameters

Create `training_config.yaml` and `test_config.yaml` files in your root directory for paths to datasets and model parameters:

```yaml
processed_path: "/path/to/your/data/"
```

### Minimum Data Requirements

- Minimum 10 non-null observations per time series

- 15-minute sampling frequency

- Forward-fill gap handling

## Training with ChronosT5

After running the preprocessing pipeline, use the generated `chronos_train.parquet` file with ChronosT5 training scripts provided on the official repo:

```bash
python chronos-forecasting/scripts/evaluation/evaluate.py --config chronos_eval_config.yaml
# Example ChronosT5 training command
python train_chronos.py \
    --data-path ./chronos_data.parquet \
    --model-size tiny \
    --batch-size 32 \
    --learning-rate 1e-4
```

## Data Characteristics

### Time Series Properties

- **Frequency**: 15-minute intervals

- **Domain**: Bus lateness (minutes)

- **Granularity**: Route-suburb combinations

### Expected Output Volume

- Processing 50k raw records typically yields hundreds of time series

- Each time series contains variable length sequences

- Average sequence length depends on data temporal span

## Training and Evaluating

First, clone into the official [chronos-repo](https://github.com/amazon-science/chronos-forecasting). The repo will contain scripts that can be used out of the box for training.

After clonning, execute the training script:

```bash
python chronos-forecasting/scripts/train.py --config path/to/train_config.yaml \
                                            --model-id amazon/chronos-t5-tiny
```
For evaluation, I added some code to the `load_and_split_dataset()` function since my test set was hosted locally. If your dataset is hosted on Hugging Face, the evaluate.py script can be used directly. 

```python
elif "path" in backtest_config:
        df = pd.read_parquet(backtest_config["path"])
        frequency = backtest_config["frequency"]
        
        gts_dataset = []
        for item_id in df['item_id'].unique():
            series_data = df[df['item_id'] == item_id]
            gts_dataset.append({
                "start": pd.Period(series_data['start'].iloc[0], freq=frequency),
                "target": series_data['target'].iloc[0]
            })

        # Set default
        offset = backtest_config.get("offset", -prediction_length)
        num_rolls = backtest_config.get("num_rolls", 1)
        
        _, test_template = split(gts_dataset, offset=offset)
        test_data = test_template.generate_instances(prediction_length, windows=num_rolls)
#
```

## References

- [ChronosT5 Documentation](https://github.com/amazon-science/chronos-forecasting)
