import os
import numpy as np
import pandas as pd
import logging
from Get_data import DatabaseManager
from ModelTraining import ModelTraining

# Configuración del logger para capturar mensajes de información y errores
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, data_updater: DatabaseManager, model_trainer: ModelTraining, 
                 folder_path: str, model_filename: str, city: str = 'Piura'):
        """
        Inicializa la clase TaskManager.

        Args:
            data_updater (DatabaseManager): Instancia para actualizar y gestionar la base de datos.
            model_trainer (ModelTraining): Instancia para cargar y gestionar modelos de entrenamiento.
            folder_path (str): Ruta al directorio donde se encuentra el archivo del modelo.
            model_filename (str): Nombre del archivo del modelo.
            city (str, opcional): Nombre de la ciudad. Por defecto es 'Piura'.
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
            sequence (array-like): Secuencia de datos para convertir.
            columns (list): Lista de nombres de las columnas del DataFrame.

        Returns:
            pd.DataFrame: DataFrame creado a partir de la secuencia.
        """
        return pd.DataFrame(sequence, columns=columns)

    def run(self):
        """
        Ejecuta el proceso principal de gestión de tareas:
        - Actualiza la base de datos.
        - Carga el modelo y realiza predicciones.
        - Compara fechas y decide si se necesita hacer predicciones futuras o actuales.
        """
        try:
            # Actualiza los datos meteorológicos en la base de datos
            self.data_updater.load_or_update_weather_data(self.city)
            # Obtiene los datos combinados para la ciudad
            df_combined = self.data_updater.get_combined_dataframe(self.city)
            # Carga el modelo desde la ruta especificada
            final_model = self.model_trainer.load_model(self.folder_path, self.model_filename)

            # Obtiene las fechas de los datos más recientes y de la última predicción
            latest_date, latest_date_prediction = self.get_dates()
            day_difference = (latest_date - latest_date_prediction).days

            # Decide el proceso de predicción basado en la diferencia de días
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
            tuple: Fecha más reciente de los datos y fecha de la última predicción.
        """
        return self.data_updater.get_latest_date(self.city), self.data_updater.get_latest_date_prediction(self.city)

    def process_future_predictions(self, df_combined, final_model, day_difference, window_size=30):
        """
        Procesa predicciones futuras basadas en los datos actuales.

        Args:
            df_combined (pd.DataFrame): DataFrame con datos combinados.
            final_model: Modelo de Keras para predicciones.
            day_difference (int): Diferencia en días para calcular futuras predicciones.
            window_size (int, opcional): Tamaño de la ventana de secuencia. Por defecto es 30.
        """
        try:
            # Prepara los datos más recientes
            last_data = self.prepare_last_data(df_combined, window_size, day_difference)
            # Extrae secuencias de los datos
            sequences = self.extract_sequences(last_data, window_size, day_difference)

            # Procesa cada secuencia para generar predicciones
            for seq in sequences:
                self.process_sequence(seq, final_model)

            # Repite el proceso para verificar si se necesitan más predicciones
            self.run()
        except Exception as e:
            logger.error(f"Error processing future predictions: {e}")

    def process_sequence(self, seq, final_model):
        """
        Procesa una secuencia de datos y realiza predicciones.

        Args:
            seq (array-like): Secuencia de datos de entrada.
            final_model: Modelo de Keras para predicciones.
        """
        try:
            # Modifica la secuencia para incluir solo las características relevantes
            features = ['id', 'time', 'tmax', 'tmin', 'tavg', 'pres']
            modified_sequence = np.hstack((seq[:, 0:5], seq[:, 10].reshape(-1, 1)))
            sequence_df = self.convert_to_dataframe(modified_sequence, features)

            # Obtiene las predicciones usando el modelo
            df_predictions = self.model_trainer.get_predictions(final_model, sequence_df, self.city)

            # Renombra las columnas de las predicciones para cargar a la base de datos
            df_predictions.rename(columns={
                'id': 'ciudad',
                'time': 'dia',
                'tmax': 'temp_max',
                'tmin': 'temp_min',
                'tavg': 'avg_temp'}, inplace=True)

            # Sube las predicciones a la base de datos
            self.data_updater.upload_data_to_database(df_predictions, table='predic')
            self.print_next_prediction()
        except Exception as e:
            logger.error(f"Error processing sequence: {e}")

    def prepare_last_data(self, df_combined, window_size, day_difference):
        """
        Prepara los datos necesarios para la predicción de acuerdo con la ventana de tiempo.

        Args:
            df_combined (pd.DataFrame): DataFrame con todos los datos combinados.
            window_size (int): Tamaño de la ventana de secuencia.
            day_difference (int): Diferencia en días para ajustar los datos.

        Returns:
            pd.DataFrame: DataFrame con los datos preparados.
        """
        return df_combined.iloc[-(window_size + day_difference):]

    def extract_sequences(self, last_data, window_size, day_difference):
        """
        Extrae secuencias de datos para hacer predicciones futuras.

        Args:
            last_data (pd.DataFrame): DataFrame con los datos más recientes.
            window_size (int): Tamaño de la ventana de secuencia.
            day_difference (int): Diferencia en días para crear las secuencias.

        Returns:
            list: Lista de secuencias de datos.
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
        Procesa las predicciones basadas en los datos actuales.

        Args:
            df_combined (pd.DataFrame): DataFrame con datos combinados.
            final_model: Modelo de Keras utilizado para las predicciones.
        """
        try:
            # Reestructura las columnas para el modelo
            features = ['id', 'time', 'tmax', 'tmin', 'tavg', 'pres']
            df_combined = df_combined[['id_ciudad', 'dia', 'temp_max', 'temp_min', 'avg_temp', 'presion']]
            df_combined.columns = features

            # Realiza la predicción y muestra el resultado
            df_predictions = self.model_trainer.get_predictions(final_model, df_combined.iloc[:, :6],self.city)

            # Renombra las columnas para cargar en la base de datos
            df_predictions.rename(columns={
                'id': 'ciudad',
                'time': 'dia',
                'tmax': 'temp_max',
                'tmin': 'temp_min',
                'tavg': 'avg_temp'}, inplace=True)
            
            # Sube los resultados a la base de datos
            self.data_updater.upload_data_to_database(df_predictions, table='predic')
        except Exception as e:
            logger.error(f"Error processing current predictions: {e}")

    def print_next_prediction(self):
        """
        Imprime la próxima predicción basada en la fecha más reciente.
        """
        try:
            latest_date_prediction = self.data_updater.get_latest_date_prediction(self.city)
            prediction = self.data_updater.next_prediction(self.city)
            logger.info(f"\nPrediction for the next day ({latest_date_prediction}):\n{prediction}\n")
        except Exception as e:
            logger.error(f"Could not retrieve the next prediction: {e}")

    @staticmethod
    def clear_terminal():
        """
        Limpia la terminal de acuerdo al sistema operativo.
        """
        os.system('cls' if os.name == 'nt' else 'clear')