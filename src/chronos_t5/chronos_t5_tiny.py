from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from chronos import ChronosPipeline
import warnings
warnings.filterwarnings("ignore")


dpath = Path.cwd().parent.parent / "cleaned_data/sample_50k.csv"


class BusLatenessPredictor:
    def __init__(self, model_name="amazon/chronos-t5-tiny"):
        self.model_name = model_name
        self.pipeline = None
        self.time_series = None
        self.time_index = None
        self.time_window = None
        
    def load_model(self):
        """Load Chronos model"""
        print(f"Loading {self.model_name}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded!")
        
    def prepare_data(self, data_path=dpath, route_filter=None, time_window="15min"):
        """Convert bus data to regular time series"""
        print("Preparing data...")
        
        self.time_window = time_window
        
        # Load and process data
        data = pd.read_csv(data_path)
        data["timetable_time"] = pd.to_datetime(data["timetable_time"], errors='coerce', utc=True)
        data["calendar_date"] = pd.to_datetime(data["calendar_date"], errors='coerce')
        
        # Filter routes if specified
        if route_filter:
            data = data[data["route"].isin(route_filter)]
        
        # Data quality check
        print(f"Lateness range: {data['lateness_minutes'].min():.1f} to {data['lateness_minutes'].max():.1f} minutes")
        
        # Create time-windowed series (15min, 30min, 1H, etc.)
        data["time_window"] = data["timetable_time"].dt.floor(time_window)
        windowed_data = data.groupby("time_window")["lateness_minutes"].agg(['mean', 'count']).reset_index()
        
        # Filter out windows with very few observations
        min_observations = 3
        windowed_data = windowed_data[windowed_data['count'] >= min_observations]
        
        # Create complete time range
        full_range = pd.date_range(
            start=windowed_data["time_window"].min(),
            end=windowed_data["time_window"].max(),
            freq=time_window
        )
        
        # Reindex and interpolate
        windowed_data = windowed_data.set_index("time_window")["mean"].reindex(full_range)
        windowed_data = windowed_data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        self.time_series = windowed_data.values
        self.time_index = windowed_data.index
        
        print(f"Created time series with {len(self.time_series)} {time_window} windows")
        print(f"Final lateness range: {self.time_series.min():.1f} to {self.time_series.max():.1f} minutes")
        return self.time_series
    
    def predict(self, prediction_length=12, num_samples=100):
        """Generate predictions using Chronos"""
        print(f"Predicting next {prediction_length} intervals...")
        
        # Convert to tensor
        context = torch.tensor(self.time_series, dtype=torch.float32).unsqueeze(0)
        
        # Generate forecast
        forecast = self.pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples
        )
        
        # Get median prediction
        predictions = forecast.median(dim=1).values.squeeze().numpy()
        
        # Create future timestamps
        future_times = pd.date_range(
            start=self.time_index[-1] + pd.Timedelta(self.time_window),
            periods=prediction_length,
            freq=self.time_window  # Use time window frequency
        )
        
        return pd.DataFrame({
            "time": future_times,
            "predicted_lateness": predictions
        })
    
    def evaluate(self, test_steps=12):
        """Evaluate model on recent data"""
        if len(self.time_series) < test_steps + 12:
            print("Not enough data for evaluation")
            return None
            
        # Split data
        train_data = self.time_series[:-test_steps]
        actual = self.time_series[-test_steps:]
        
        # Predict on training data
        context = torch.tensor(train_data, dtype=torch.float32).unsqueeze(0)
        forecast = self.pipeline.predict(context, prediction_length=test_steps)
        predicted = forecast.median(dim=1).values.squeeze().numpy()
        
        # Calculate metrics
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        
        print(f"\nEvaluation Results:")
        print(f"MAE: {mae:.2f} minutes")
        print(f"RMSE: {np.sqrt(mse):.2f} minutes")
        
        return {"mae": mae, "rmse": np.sqrt(mse), "actual": actual, "predicted": predicted}
    
    def plot_results(self, predictions, eval_results=None):
        """Plot predictions and evaluation"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot recent history and predictions
        recent_hours = min(72, len(self.time_series))
        recent_data = self.time_series[-recent_hours:]
        recent_times = self.time_index[-recent_hours:]
        
        axes[0].plot(recent_times, recent_data, label="Historical", color="blue")
        axes[0].plot(predictions["time"], predictions["predicted_lateness"], 
                    label="Predicted", color="red", linestyle="--")
        axes[0].set_title("Bus Lateness Forecast")
        axes[0].set_ylabel("Lateness (minutes)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot evaluation if available
        if eval_results:
            axes[1].plot(eval_results["actual"], label="Actual", color="blue")
            axes[1].plot(eval_results["predicted"], label="Predicted", color="red")
            axes[1].set_title("Model Evaluation")
            axes[1].set_ylabel("Lateness (minutes)")
            axes[1].set_xlabel("Hours")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].hist(self.time_series, bins=30, alpha=0.7, color="skyblue")
            axes[1].set_title("Lateness Distribution")
            axes[1].set_xlabel("Lateness (minutes)")
            axes[1].set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig("results/chronos_output.png")
        plt.show()

def main():
    """Simple pipeline execution"""
    print("=== Simple Bus Lateness Prediction ===\n")
    
    # Initialize and load model
    predictor = BusLatenessPredictor()
    predictor.load_model()
    
    # Prepare data with 15-minute windows
    time_series = predictor.prepare_data(time_window="15min")
    
    # Generate predictions (12 hours = 48 15 minutes intervals)
    predictions = predictor.predict(prediction_length=48)
    print(f"\nNext 12 hours predictions (15-min intervals):")
    print(predictions)
    
    # Evaluate model
    eval_results = predictor.evaluate(test_steps=48)
    
    # Plot results
    predictor.plot_results(predictions, eval_results)
    
    return predictor, predictions

if __name__ == "__main__":
    predictor, predictions = main()