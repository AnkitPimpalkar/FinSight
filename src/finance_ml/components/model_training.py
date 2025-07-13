from finance_ml.entity.config_entity import ModelTrainingConfig, ModelTrainingParams 
from tensorflow import keras
import numpy as np
import os
import tensorflow as tf
import mlflow.keras  # Added for autologging

class ModelTraining:
    def __init__(self, config: ModelTrainingConfig, params: ModelTrainingParams): 
        self.config = config
        self.params = params

    def train_model(self):
        # Enable MLflow autologging
        mlflow.keras.autolog()
        # Load the transformed data
        X_train = np.load(os.path.join("artifacts", "data_transformation", "X_train.npy"))
        y_train = np.load(os.path.join("artifacts", "data_transformation", "y_train.npy"))
        X_test = np.load(os.path.join("artifacts", "data_transformation", "X_test.npy"))
        y_test = np.load(os.path.join("artifacts", "data_transformation", "y_test.npy"))

        # Build the Model using parameters
        model = keras.models.Sequential()

        model.add(keras.layers.LSTM(self.params.lstm_units_1, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(keras.layers.LSTM(self.params.lstm_units_2, return_sequences=False))
        model.add(keras.layers.Dense(self.params.dense_units_1, activation="relu"))
        model.add(keras.layers.Dropout(self.params.dropout_rate))
        model.add(keras.layers.Dense(1))

        model.compile(optimizer="adam",
                      loss="mae",
                      metrics=[keras.metrics.RootMeanSquaredError()])

        # Add early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )

        # Train the model using parameters
        model.fit(
            X_train, y_train, 
            epochs=self.params.epochs, 
            batch_size=self.params.batch_size,
            callbacks=[early_stopping]
        )

        # Save the trained model
        model.save(os.path.join(self.config.root_dir, self.config.trained_model_name))

        return model
