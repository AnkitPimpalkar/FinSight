import os
import numpy as np
import pandas as pd
from tensorflow import keras
import mlflow
import mlflow.keras
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from finance_ml.entity.config_entity import ModelEvaluationConfig
from finance_ml.utils.common import save_json


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE as a percentage."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def directional_accuracy(y_true, y_pred):
    """Calculates the percentage of times the model correctly predicts the price direction (up or down)."""
    y_true_diff = np.diff(y_true.flatten())
    y_pred_diff = np.diff(y_pred.flatten())
    
    # Correct direction if the signs are the same
    correct_direction = np.sign(y_true_diff) == np.sign(y_pred_diff)
    return np.mean(correct_direction) * 100


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def _inverse_transform(self, y_scaled, scaler):
        """Helper to inverse transform data."""
        # The scaler expects a 2D array with the same number of features it was trained on.
        # We create a dummy array of the right shape, put our data in the first column, and then inverse transform.
        dummy_array = np.zeros((len(y_scaled), scaler.n_features_in_))
        dummy_array[:, 0] = y_scaled.flatten()
        return scaler.inverse_transform(dummy_array)[:, 0]

    def _create_and_log_visualization(self, y_true, y_pred):
        """Creates and logs a plot of actual vs. predicted values."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 7))
        
        ax.plot(y_true, color='#00BFFF', label='Actual Price', linewidth=2)
        ax.plot(y_pred, color='#FF4500', label='Predicted Price', linestyle='--')
        
        ax.set_title('Stock Price Prediction: Actual vs. Predicted', fontsize=16)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Price (Rupees)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        
        # Create a dedicated directory for visualizations
        viz_dir = Path(self.config.metrics_file_name).parent / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_path = viz_dir / "prediction_visualization.png"

        fig.savefig(plot_path)
        plt.close(fig)
        
        mlflow.log_artifact(str(plot_path), "visualizations")

    def evaluate(self):
        """
        Evaluates the model, calculates comprehensive metrics, logs them to MLflow,
        and saves a visualization.
        """
        # Use config paths for model and data, ensuring they are Path objects
        model_path = Path(self.config.model_path)
        X_test_path = Path(self.config.X_test_path)
        y_test_path = Path(self.config.y_test_path)
        # Path to the scaler object, assuming it's saved in the transformation artifact directory
        scaler_path = X_test_path.parent / "scaler.joblib"

        if not all(p.exists() for p in [model_path, X_test_path, y_test_path, scaler_path]):
            raise FileNotFoundError(f"Model, test data, or scaler not found. Checked paths:\n- {model_path}\n- {X_test_path}\n- {y_test_path}\n- {scaler_path}")

        # Load model, data, and scaler
        model = keras.models.load_model(model_path)
        X_test = np.load(X_test_path)
        y_test_scaled = np.load(y_test_path)
        scaler = joblib.load(scaler_path)

        # Start MLflow run
        with mlflow.start_run():
            # Log model parameters from config if available
            if hasattr(self.config, 'all_params'):
                mlflow.log_params(self.config.all_params)

            # Make predictions
            y_pred_scaled = model.predict(X_test)

            # Inverse transform to get actual rupee values
            y_test_rupees = self._inverse_transform(y_test_scaled, scaler)
            y_pred_rupees = self._inverse_transform(y_pred_scaled, scaler)

            # 1. Calculate error in Rupees (MAE, RMSE)
            mae_rupees = mean_absolute_error(y_test_rupees, y_pred_rupees)
            rmse_rupees = np.sqrt(mean_squared_error(y_test_rupees, y_pred_rupees))

            # 2. Calculate error in Percentage (MAPE) and Accuracy
            mape = mean_absolute_percentage_error(y_test_rupees, y_pred_rupees)
            accuracy_percentage = 100 - mape

            # 3. Calculate Directional Accuracy
            dir_accuracy = directional_accuracy(y_test_rupees, y_pred_rupees)
            
            # Log metrics to MLflow
            metrics = {
                "mae_in_rupees": float(mae_rupees),
                "rmse_in_rupees": float(rmse_rupees),
                "mape_percentage": float(mape),
                "accuracy_percentage": float(accuracy_percentage),
                "directional_accuracy_percentage": float(dir_accuracy)
            }
            mlflow.log_metrics(metrics)

            # 4. & 5. Generate and log visualization
            self._create_and_log_visualization(y_test_rupees, y_pred_rupees)

            # Save metrics to a local JSON file
            save_json(path=Path(self.config.metrics_file_name), data=metrics)
            
            print("Model evaluation complete. Metrics and visualization logged to MLflow and saved locally.")