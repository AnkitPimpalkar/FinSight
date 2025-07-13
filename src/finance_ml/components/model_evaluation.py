import os
import numpy as np
from tensorflow import keras
import mlflow
import mlflow.keras  
from pathlib import Path

from finance_ml.entity.config_entity import ModelEvaluationConfig
from finance_ml.utils.common import save_json

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        # Enable MLflow autologging for Keras
        mlflow.keras.autolog()

        # Use config paths for model and data
        model_path = getattr(self.config, "model_path", os.path.join("artifacts", "model_training", "lstm_model.h5"))
        X_test_path = getattr(self.config, "X_test_path", os.path.join("artifacts", "data_transformation", "X_test.npy"))
        y_test_path = getattr(self.config, "y_test_path", os.path.join("artifacts", "data_transformation", "y_test.npy"))

        if not (os.path.exists(model_path) and os.path.exists(X_test_path) and os.path.exists(y_test_path)):
            raise FileNotFoundError("Model or test data files not found. Please check the paths.")

        model = keras.models.load_model(model_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        # Evaluate the model
        results = model.evaluate(X_test, y_test, verbose=0)
        if isinstance(results, list) and len(results) >= 2:
            loss, rmse = results[:2]
        else:
            raise ValueError("Model evaluation did not return expected metrics (loss, rmse).")

        # Log metrics to MLflow (autologging will handle this, but you can keep manual logging for custom metrics)
        try:
            with mlflow.start_run():
                mlflow.log_metric("test_loss_mae", float(loss))
                mlflow.log_metric("test_rmse", float(rmse))
        except Exception as e:
            import logging
            logging.warning(f"MLflow logging failed: {e}. Continuing without MLflow tracking.")

        # Save metrics to a JSON file
        metrics = {
            "test_loss_mae": float(loss),
            "test_rmse": float(rmse)
        }
        metrics_file_path = Path(self.config.root_dir) / self.config.metrics_file_name
        save_json(metrics_file_path, metrics)