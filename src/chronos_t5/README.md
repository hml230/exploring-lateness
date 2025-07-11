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

## Troubleshooting

### Common Issues

#### "Created 0 time series"

- Check if minimum data threshold is too high

- Verify timestamp parsing is working correctly

- Ensure data contains sufficient non-null values

#### Memory Issues

- Increase Spark memory allocation

- Process data in smaller chunks

- Consider using Spark's built-in partitioning

## References

- [ChronosT5 Documentation](https://github.com/amazon-science/chronos-forecasting)
