import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import StandardScaler

class ModelTraining:
    def __init__(self, data_manager):
        self.scaler = StandardScaler()
        self.data_manager = data_manager 

    def load_model(self, folder_path, model_filename):
        model_path = os.path.join(folder_path, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'El archivo {model_path} no existe.')
        return load_model(model_path)

    def get_predictions(self, final_model, df_combined, city_name):
        df_combined = df_combined.dropna(axis=1, how='all')
        features = ['tmax', 'tmin', 'tavg', 'pres']
        self.scaler.fit(df_combined[features])
        data = self.scaler.transform(df_combined[features])

        window_size = 30
        # Obtener la última secuencia de datos
        last_sequence = data[-window_size:]
        last_sequence = np.expand_dims(last_sequence, axis=0)

        # Realizar la predicción
        predicted = final_model.predict(last_sequence)

        # Si la predicción tiene 3 elementos
        if predicted.shape[1] == 3:
            predicted = np.concatenate([predicted, np.zeros((predicted.shape[0], 1))], axis=1)

        predicted = self.scaler.inverse_transform(predicted)

        # Eliminar el valor agregado
        if predicted.shape[1] == 4:
            predicted = predicted[:, :-1]

        predicted = np.round(predicted, decimals=1)

        # Obtener el ID de la ciudad
        city_id = self.data_manager.get_city_id(city_name)

        if city_id is None:
            raise ValueError(f"No se pudo obtener el ID de la ciudad '{city_name}'")

        df_predictions = pd.DataFrame({
            'id': [city_id],  # Utiliza el ID de la ciudad obtenido
            'time': [pd.Timestamp(df_combined['time'].iloc[-1]) + pd.DateOffset(days=1)],
            'tmax': [predicted[0, 0]],
            'tmin': [predicted[0, 1]],
            'tavg': [predicted[0, 2]]
        })

        return df_predictions
