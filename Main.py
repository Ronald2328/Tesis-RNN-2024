import warnings
from Get_data import DatabaseManager
from ModelTraining import ModelTraining
from TaskManager import TaskManager

warnings.filterwarnings("ignore")

def main():
    city = 'Piura'
    folder_path = 'C:\\Users\\Ronaldo Olivares\\Desktop\\DOCUMENTO DE TESIS\\Codigos'
    model_filename = 'Modelo_Entrenado.keras'
    
    data_updater = DatabaseManager()
    model_trainer = ModelTraining(data_updater)  # Pasar la instancia de DatabaseManager
    task_manager = TaskManager(data_updater, model_trainer, folder_path, model_filename, city)
    task_manager.clear_terminal()
    task_manager.run()

if __name__ == "__main__":
    main()
