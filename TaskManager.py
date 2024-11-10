import os
import numpy as np
import pandas as pd
import logging
from Get_data import DatabaseManager
from ModelTraining import ModelTraining

# Configuración del logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, data_updater: DatabaseManager, model_trainer: ModelTraining, 
                 folder_path: str, model_filename: str, city: str = 'Piura'):
        """
        Inicializa la clase TaskManager.
        
        Args:
            data_updater (DatabaseManager): Instancia del gestor de base de datos.
            model_trainer (ModelTraining): Instancia del gestor de entrenamiento de modelos.
            folder_path (str): Ruta del directorio que contiene el modelo.
            model_filename (str): Nombre del archivo del modelo.
        """
        self.data_updater = data_updater
        self.model_trainer = model_trainer
        self.folder_path = folder_path
        self.model_filename = model_filename
        self.city = city  

    @staticmethod
    def convert_to_dataframe(sequence, columns):
        """
        Convierte una secuencia a un DataFrame de pandas.

        Args:
            sequence (array-like): La secuencia de datos a convertir.
            columns (list): Lista de nombres de las columnas del DataFrame.

        Returns:
            pd.DataFrame: DataFrame creado a partir de la secuencia.
        """
        return pd.DataFrame(sequence, columns=columns)

    def run(self):
        """
        Ejecuta el proceso principal de gestión de tareas.
        
        Actualiza la base de datos, obtiene el modelo y realiza predicciones según la diferencia de días 
        entre la fecha más reciente de los datos y la fecha de la última predicción.
        """
        try:
            self.data_updater.load_or_update_weather_data(self.city)
            df_combined = self.data_updater.get_combined_dataframe(self.city)
            final_model = self.model_trainer.load_model(self.folder_path, self.model_filename)

            latest_date, latest_date_prediction = self.get_dates()
            day_difference = (latest_date - latest_date_prediction).days

            if day_difference > 1:
                self.process_future_predictions(df_combined, final_model, day_difference)
            else:
                self.process_current_predictions(df_combined, final_model) if day_difference == 0 else self.print_next_prediction()

        except Exception as e:
            logger.error(f"An error occurred while running the TaskManager: {e}")

    def get_dates(self):
        """
        Obtiene las fechas más recientes de los datos y de la última predicción.

        Returns:
            tuple: Una tupla que contiene la fecha más reciente de los datos y la fecha de la última predicción.
        """
        return self.data_updater.get_latest_date(), self.data_updater.get_latest_date_prediction()

    def process_future_predictions(self, df_combined, final_model, day_difference, window_size=30):
        """
        Procesa las predicciones futuras basadas en los datos más recientes.

        Args:
            df_combined (pd.DataFrame): DataFrame que contiene los datos combinados.
            final_model: El modelo final cargado para realizar predicciones.
            day_difference (int): Diferencia en días entre las fechas más recientes y la última predicción.
        """
        try:
            last_data = self.prepare_last_data(df_combined, window_size, day_difference)
            sequences = self.extract_sequences(last_data, window_size, day_difference)

            for seq in sequences:
                self.process_sequence(seq, final_model)
            
            self.run()
        except Exception as e:
            logger.error(f"Error processing future predictions: {e}")

    def process_sequence(self, seq, final_model):
        """
        Procesa una secuencia de datos para obtener predicciones.

        Args:
            seq (array-like): Secuencia de datos para realizar la predicción.
            final_model: El modelo utilizado para realizar predicciones.
        """
        try:
            features = ['id', 'time', 'tmax', 'tmin', 'tavg', 'pres']
            modified_sequence = np.hstack((seq[:, 0:5], seq[:, 10].reshape(-1, 1)))
            sequence_df = self.convert_to_dataframe(modified_sequence, features)
            df_predictions = self.model_trainer.get_predictions(final_model, sequence_df)

            # Renombrar columnas
            df_predictions.rename(columns={
                    'id': 'ciudad',
                    'time': 'dia',
                    'tmax': 'temp_max',
                    'tmin': 'temp_min',
                    'tavg': 'avg_temp'}, inplace=True)
            
            self.data_updater.upload_data_to_database(df_predictions, table='prediccion')
            self.print_next_prediction()
        except Exception as e:
            logger.error(f"Error processing sequence: {e}")

    def prepare_last_data(self, df_combined, window_size, day_difference):
        """
        Prepara los últimos datos necesarios para las predicciones.

        Args:
            df_combined (pd.DataFrame): DataFrame que contiene todos los datos combinados.
            window_size (int): Tamaño de la ventana para la secuencia.
            day_difference (int): Diferencia en días para ajustar los datos.

        Returns:
            pd.DataFrame: DataFrame con los últimos datos preparados.
        """
        return df_combined.iloc[-(window_size + day_difference):]

    def extract_sequences(self, last_data, window_size, day_difference):
        """
        Extrae las secuencias de datos de los últimos 30 días.

        Args:
            last_data (pd.DataFrame): DataFrame con los últimos 30 días de datos.
            window_size (int): Tamaño de la ventana para la secuencia.
            day_difference (int): Diferencia en días para ajustar las secuencias.

        Returns:
            list: Lista de secuencias extraídas de los datos.
        """
        sequences = []
        for i in range(day_difference):
            start_index = i
            end_index = i + window_size
            if end_index <= len(last_data):
                sequences.append(last_data.iloc[start_index:end_index].values)
        return sequences

    def process_current_predictions(self, df_combined, final_model):
        """
        Procesa las predicciones actuales basadas en los datos disponibles.

        Args:
            df_combined (pd.DataFrame): DataFrame que contiene los datos combinados.
            final_model: El modelo utilizado para realizar predicciones.
        """
        try:
            features = ['id', 'time', 'tmax', 'tmin', 'tavg', 'pres']
            df_combined = df_combined[['id_ciudad', 'dia', 'temp_max', 'temp_min', 'avg_temp', 'presion']]
            df_combined.columns = features
            print(df_combined) 
            df_predictions = self.model_trainer.get_predictions(final_model, df_combined.iloc[:, :6])
            print(df_predictions)

            # Renombrar columnas
            df_predictions.rename(columns={
                        'id': 'ciudad',
                        'time': 'dia',
                        'tmax': 'temp_max',
                        'tmin': 'temp_min',
                        'tavg': 'avg_temp'}, inplace=True)
            
            self.data_updater.upload_data_to_database(df_predictions, table='prediccion')
        except Exception as e:
            logger.error(f"Error processing current predictions: {e}")

    def print_next_prediction(self):
        """
        Imprime la siguiente predicción basada en los datos más recientes.
        """
        try:
            latest_date_prediction = self.data_updater.get_latest_date_prediction()
            prediction = self.data_updater.next_prediction()
            logger.info(f"\nPrediction for the next day ({latest_date_prediction}):\n{prediction}\n")
        except Exception as e:
            logger.error(f"Could not retrieve the next prediction: {e}")

    @staticmethod
    def clear_terminal():
        """
        Limpia la terminal según el sistema operativo.
        """
        os.system('cls' if os.name == 'nt' else 'clear')