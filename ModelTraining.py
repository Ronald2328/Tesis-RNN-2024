import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
from sklearn.preprocessing import StandardScaler

class ModelTraining:
    def __init__(self, data_manager):
        """
        Inicializa la clase ModelTraining con un escalador y un gestor de datos.

        Args:
            data_manager: Instancia de la clase DatabaseManager utilizada para interactuar con la base de datos.
        """
        self.scaler = StandardScaler()
        self.data_manager = data_manager

    def load_model(self, folder_path, model_filename):
        """
        Carga un modelo de aprendizaje profundo desde un archivo.

        Args:
            folder_path (str): Ruta a la carpeta donde se encuentra el archivo del modelo.
            model_filename (str): Nombre del archivo del modelo.

        Returns:
            Modelo de Keras cargado.

        Raises:
            FileNotFoundError: Si el archivo del modelo no existe en la ruta especificada.
        """
        model_path = os.path.join(folder_path, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'El archivo {model_path} no existe.')
        return load_model(model_path)

    def get_predictions(self, final_model, df_combined, city_name):
        """
        Realiza predicciones basadas en un modelo entrenado y los datos de entrada.

        Args:
            final_model: Modelo de Keras utilizado para hacer predicciones.
            df_combined (pd.DataFrame): DataFrame que contiene los datos de entrada con características relevantes.
            city_name (str): Nombre de la ciudad para obtener su ID de la base de datos.

        Returns:
            pd.DataFrame: DataFrame con las predicciones, incluyendo el ID de la ciudad, la fecha de predicción y las temperaturas.

        Raises:
            ValueError: Si no se puede obtener el ID de la ciudad.
        """
        # Eliminar columnas que estén completamente vacías
        df_combined = df_combined.dropna(axis=1, how='all')
        
        # Definir las características a utilizar y escalar los datos
        features = ['tmax', 'tmin', 'tavg', 'pres']
        self.scaler.fit(df_combined[features])
        data = self.scaler.transform(df_combined[features])

        # Definir el tamaño de la ventana para la secuencia de entrada al modelo
        window_size = 30
        last_sequence = data[-window_size:]  # Obtener la última secuencia de datos
        last_sequence = np.expand_dims(last_sequence, axis=0)  # Expandir dimensiones para la predicción

        # Realizar la predicción
        predicted = final_model.predict(last_sequence)

        # Ajustar la forma de la predicción si es necesario
        if predicted.shape[1] == 3:
            predicted = np.concatenate([predicted, np.zeros((predicted.shape[0], 1))], axis=1)

        # Inversa la transformación de estandarización
        predicted = self.scaler.inverse_transform(predicted)

        # Eliminar la columna extra añadida si existe
        if predicted.shape[1] == 4:
            predicted = predicted[:, :-1]

        # Redondear los valores predichos
        predicted = np.round(predicted, decimals=1)

        # Obtener el ID de la ciudad desde la base de datos
        city_id = self.data_manager.get_city_id(city_name)
        if city_id is None:
            raise ValueError(f"No se pudo obtener el ID de la ciudad '{city_name}'")

        # Crear un DataFrame con las predicciones
        df_predictions = pd.DataFrame({
            'id': [city_id],  # ID de la ciudad
            'time': [pd.Timestamp(df_combined['time'].iloc[-1]) + pd.DateOffset(days=1)],  # Fecha de predicción
            'tmax': [predicted[0, 0]],  # Temperatura máxima
            'tmin': [predicted[0, 1]],  # Temperatura mínima
            'tavg': [predicted[0, 2]]   # Temperatura promedio
        })

        return df_predictions
