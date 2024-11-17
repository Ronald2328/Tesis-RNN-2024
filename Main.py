import os
from Get_data import DatabaseManager
from ModelTraining import ModelTraining
from TaskManager import TaskManager
from LSTM_model import LSTM

def main():
    # Define la ciudad y las columnas de entrada
    ciudad = 'Piura'

    # Directorio donde se encuentra el modelo y el archivo de escalado
    folder_path = 'C:\\Users\\Ronaldo Olivares\\Desktop\\DOCUMENTO DE TESIS\\Codigos'
    model_filename = 'modelo_completo.pth'
    scaler_path = 'escalado.pkl'  # Ruta a tu archivo de escalado

    # Inicialización de los gestores de base de datos y modelo
    data_manager = DatabaseManager(host='localhost', 
                                   user='root', 
                                   passwd='root', 
                                   database='meteorology')
    
    model_trainer = ModelTraining(data_manager, scaler_path)

    # Inicializa el TaskManager con los datos necesarios
    task_manager = TaskManager(data_updater=data_manager, 
                               model_trainer=model_trainer, 
                               folder_path=folder_path, 
                               model_filename=model_filename, 
                               city=ciudad)

    # Ejecuta el proceso
    task_manager.clear_terminal()
    task_manager.run()

if __name__ == "__main__":
    main()
