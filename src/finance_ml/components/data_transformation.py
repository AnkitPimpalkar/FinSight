from finance_ml.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import traceback
from joblib import dump # Import dump to save the scaler
from finance_ml import logger # Import logger for logging within the component

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def transform_and_save_data(self, feature="Close", lookback=60, split_ratio=0.95):
        """
        Loads raw data, transforms it, and saves the transformed data and scaler.

        Args:
            feature (str): The feature column to use for transformation (default 'close').
            lookback (int): Number of previous time steps to use for prediction (default 60).
            split_ratio (float): Ratio for splitting data into training and testing sets (default 0.95).
        """
        raw_data_path = self.config.raw_data_file

        df = pd.read_csv(raw_data_path)
        logger.info(f"Raw data loaded from: {raw_data_path}")

        df['Datetime'] = pd.to_datetime(df['Datetime'])
        data = df[[feature]].copy()
        values = data.values

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(values)

        training_data_len = int(len(scaled_data) * split_ratio)

        train_data = scaled_data[:training_data_len]
        test_data = scaled_data[training_data_len - lookback:]

        def create_sequences(data):
            X, y = [], []
            for i in range(lookback, len(data)):
                X.append(data[i - lookback:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        try:
        # Save transformed data and scaler
            np.save(os.path.join(self.config.root_dir, 'X_train.npy'), X_train)
            np.save(os.path.join(self.config.root_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.config.root_dir, 'X_test.npy'), X_test)
            np.save(os.path.join(self.config.root_dir, 'y_test.npy'), y_test)
            dump(scaler, os.path.join(self.config.root_dir, 'scaler.joblib')) # Save the scaler

            logger.info("Transformed data and scaler saved.")
        except Exception as e:
            logger.error(f"Error saving transformed data: {e}")
            logger.error(traceback.format_exc())