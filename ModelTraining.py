import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import torch
from Get_data import DatabaseManager
import logging

class ModelTraining:
    def __init__(self, data_manager: DatabaseManager, scaler_path='escalado.pkl'):
        self.scaler = joblib.load(scaler_path)
        self.data_manager = data_manager
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_model(self, folder_path: str, model_filename: str, model=None) -> torch.nn.Module:
        model_path = os.path.join(folder_path, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'El archivo {model_path} no existe.\n')

        if model_filename.endswith('.pth'):
            model = torch.load(model_path)
            model.eval()
            self.logger.info(f"Modelo PyTorch cargado exitosamente desde {model_path}.\n")
            return model
        else:
            raise ValueError("Formato de modelo no soportado. Asegúrese de que el archivo sea .pth (PyTorch) o .h5/.keras (Keras).\n")

    def prepare_data(self, df: pd.DataFrame, add_month_columns: bool=True) -> pd.DataFrame:
        df['time'] = pd.to_datetime(df['time'])
        df['mes'] = df['time'].dt.month
        
        if add_month_columns:
            meses_dummies = pd.get_dummies(df['mes'], prefix='month').astype(int)
            all_months = [f'month_{i}' for i in range(1, 13)]
            meses_dummies = meses_dummies.reindex(columns=all_months, fill_value=0)
            df = pd.concat([df, meses_dummies], axis=1)
        
        df.drop('mes', axis=1, inplace=True)
        return df

    def get_predictions(self, final_model: torch.nn.Module, df_combined: pd.DataFrame, city_name: str, 
                        window_size: int, features: list=None, add_month_columns: bool=True) -> pd.DataFrame:
        df_combined = df_combined.dropna(axis=1, how='all')
        df_combined = self.prepare_data(df_combined, add_month_columns)

        features = [col for col in df_combined.columns if col not in ['id_ciudad', 'time','snow', 'wpgt', 'tsun']]
        
        data = df_combined[features].values
        data = self.scaler.transform(data)

        last_sequence = data[-window_size:]
        last_sequence = np.expand_dims(last_sequence, axis=0)
        last_sequence = torch.tensor(last_sequence, dtype=torch.float32)

        final_model.eval()
        with torch.no_grad():
            predicted = final_model(last_sequence)

        predicted = predicted.squeeze().cpu().numpy()
        predicted = np.round(predicted, decimals=1)

        city_id = self.data_manager.get_city_id_by_name(city_name)
        if city_id is None:
            raise ValueError(f"No se pudo obtener el ID de la ciudad '{city_name}'.\n")

        df_predictions = pd.DataFrame({
            'id': [city_id],
            'time': [pd.Timestamp(df_combined['time'].iloc[-1]) + pd.DateOffset(days=1)],
            'tmax': [predicted[0]],
            'tmin': [predicted[1]],
            'tavg': [predicted[2]]
        })

        print(f"Predicción realizada para la ciudad {city_name}:\n{df_predictions}\n")
        return df_predictions
