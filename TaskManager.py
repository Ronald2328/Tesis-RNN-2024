import os
import numpy as np
import pandas as pd
import logging
from Get_data import DatabaseManager
from ModelTraining import ModelTraining

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, data_updater: DatabaseManager, model_trainer: ModelTraining, 
                 folder_path: str, model_filename: str, city: str = 'Piura') -> None:
        self.data_updater = data_updater
        self.model_trainer = model_trainer
        self.folder_path = folder_path
        self.model_filename = model_filename
        self.city = city  

    def run(self) -> None:
        try:
            self.data_updater.update_climate_data(self.city)
            id_city = self.data_updater.get_city_id_by_name(self.city)
            df_combined = self.data_updater.get_dataframe('clima', id_city)
            final_model = self.model_trainer.load_model(self.folder_path, self.model_filename)

            latest_date, latest_date_prediction = self.data_updater.get_latest_date_for_city(self.city), \
                                                    self.data_updater.get_latest_prediction_date_for_city(self.city)
            day_difference = (latest_date - latest_date_prediction).days

            if day_difference > 0:
                window_size = 20
                self.process_predictions(df_combined, final_model, day_difference, window_size)
            else:
                self.print_next_prediction()

        except Exception as e:
            logger.error(f"Ocurrió un error al ejecutar el TaskManager: {e}\n")

    def process_predictions(self, df_combined: pd.DataFrame, final_model: object, 
                            day_difference: int, window_size: int) -> None:
        try:
            columns = df_combined.columns
            last_data = df_combined.iloc[-(window_size + day_difference):]
            if day_difference == 1:
                sequences = [last_data.iloc[-window_size:].values]
            else:
                sequences = [last_data.iloc[i:i + window_size].values for i in range(day_difference)]
            
            for seq in sequences:
                sequence_df = pd.DataFrame(seq, columns=columns)
                df_predictions = self.model_trainer.get_predictions(final_model, sequence_df, self.city, window_size)
                self.data_updater.insert_data('predic', df_predictions)
                self.print_next_prediction()

        except Exception as e:
            logger.error(f"Error al procesar las predicciones: {e}\n")

    def print_next_prediction(self) -> None:
        try:
            latest_date_prediction = self.data_updater.get_latest_prediction_date_for_city(self.city, False)
            prediction = self.data_updater.next_prediction(self.city)
            prediction_df = pd.DataFrame([prediction])
            logger.info(f"\nPredicción para el próximo día ({latest_date_prediction}):\n{prediction_df}\n")
        except Exception as e:
            logger.error(f"No se pudo obtener la siguiente predicción: {e}\n")

    @staticmethod
    def clear_terminal() -> None:
        os.system('cls' if os.name == 'nt' else 'clear')