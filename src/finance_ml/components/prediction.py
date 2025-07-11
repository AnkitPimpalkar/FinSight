import os
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path
from joblib import load

from finance_ml.entity.config_entity import ModelPredictionConfig

class ModelPrediction:
    def __init__(self, config: ModelPredictionConfig):
        self.config = config

    def predict(self):
        # Load the trained model
        model = keras.models.load_model(self.config.trained_model_path)

        # Load the input data for prediction
        data = pd.read_csv(self.config.input_data_path)

        # --- Data Preprocessing for Prediction ---
        # Select the 'close' column
        stock_close = data.filter(["Close"])
        dataset = stock_close.values # Convert to numpy array

        # Load the fitted scaler
        scaler = load("artifacts/data_transformation/scaler.joblib")

        # Use the loaded scaler to transform your prediction data
        scaled_data = scaler.transform(dataset) 

        
        # the prediction is for the next step based on the last 'time_steps' data points
        # Need to determine the 'time_steps' used during training data
        time_steps = (self.config.lookback)

        # Use the last 'time_steps' data points from the scaled data
        X_predict = scaled_data[-time_steps:].reshape(1, time_steps, 1) # Reshape for LSTM: (samples, time_steps, features)


        # Make the prediction
        predicted_price_scaled = model.predict(X_predict)

        # Inverse transform the prediction to get the actual price
        predicted_price = scaler.inverse_transform(predicted_price_scaled)



        # ---Save the Prediction with Date ---
        # Get the last date from the input data
        last_date = pd.to_datetime(data['Datetime'].iloc[-1])
        # Predict for the next day (adjust if your data is not daily)
        predicted_date = last_date + pd.Timedelta(days=1)

        # Save both date and predicted price
        predictions_df = pd.DataFrame({
            'predicted_date': [predicted_date.strftime('%Y-%m-%d')],
            'predicted_close_price': [predicted_price[0][0]]
        })

        predictions_file_path = Path(self.config.root_dir) / self.config.predictions_file_name
        predictions_df.to_csv(predictions_file_path, index=False)  # Use pandas directly

        # --- Append to Prediction History ---
        # Extract ticker from input data
        if 'Ticker' in data.columns:
            ticker = data['Ticker'].iloc[-1]
        else:
            ticker = 'UNKNOWN'
        history_file_path = Path(self.config.root_dir) / 'prediction_history.csv'
        # Prepare historical record
        historical_record = pd.DataFrame({
            'ticker': [ticker],
            'prediction_generated_on': [pd.Timestamp.now().strftime('%Y-%m-%d')],
            'predicted_for_date': [predicted_date.strftime('%Y-%m-%d')],
            'predicted_close_price': [predicted_price[0][0]]
        })
        # Append or create
        if history_file_path.exists():
            history_df = pd.read_csv(history_file_path)
            updated_history = pd.concat([history_df, historical_record], ignore_index=True)
        else:
            updated_history = historical_record
        updated_history.to_csv(history_file_path, index=False)
        
        print(f"Prediction history updated at: {history_file_path}")
        print(f" <<<<<<<<<<<<<<<Predicted next closing price for {predicted_date.strftime('%Y-%m-%d')}: {predicted_price[0][0]}>>>>>>>>>>>>>>>>")
        print(f"Predictions saved to: {predictions_file_path}")
        print(f"Scaler path used: artifacts/data_transformation/scaler.joblib")
        print(f"Data used for prediction (last 5):\n{stock_close.tail()}")
        print(f"Model loaded from: {self.config.trained_model_path}")
        model.summary()
        print(f"Input data path used: {self.config.input_data_path}")
        print("First 2 rows of input data:")
        print(data.head(2))
        print("Ticker column:", data['Ticker'].unique())
        print(f"Final prediction file path: {predictions_file_path}")
        print("Predicted date and price:\n", predictions_df)

