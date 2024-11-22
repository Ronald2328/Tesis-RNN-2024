import os
import subprocess
import sys
from LSTM_model import LSTM

def check_and_install_requirements(requirements_file="requirements.txt"):
    """
    Verifica si los paquetes en requirements.txt están instalados y los instala si no lo están.
    """
    try:
        # Verifica si pip está disponible
        subprocess.check_call([sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL)
    except Exception:
        print("Pip no está instalado. Asegúrate de tener pip disponible.")
        sys.exit(1)

    # Verifica si requirements.txt existe
    if not os.path.exists(requirements_file):
        print(f"El archivo {requirements_file} no se encontró.")
        sys.exit(1)

    # Leer dependencias desde el archivo requirements.txt
    with open(requirements_file, "r") as f:
        dependencies = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Verifica qué paquetes ya están instalados
    installed_packages_output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    installed_packages = installed_packages_output.decode("utf-8").split("\n")
    
    # Normaliza el formato de los paquetes instalados
    installed_packages_dict = {}
    for pkg in installed_packages:
        if "==" in pkg:
            name, version = pkg.split("==")
            installed_packages_dict[name.lower().strip()] = version.strip()

    # Lista de paquetes que faltan o tienen una versión diferente
    packages_to_install = []

    for dep in dependencies:
        if "==" in dep:
            pkg, version = dep.split("==")
            pkg = pkg.lower().strip()
            version = version.strip()
            # Verifica si el paquete no está instalado o si la versión instalada es diferente
            if pkg not in installed_packages_dict or installed_packages_dict[pkg] != version:
                packages_to_install.append(dep)
        else:
            pkg = dep.lower().strip()
            if pkg not in installed_packages_dict:
                packages_to_install.append(dep)

    # Instala solo los paquetes que no están instalados o tienen una versión diferente
    if packages_to_install:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)
        except Exception as e:
            print(f"Error al instalar dependencias: {e}")
            sys.exit(1)



def main():
    # Llama a la función para verificar e instalar requisitos
    requirements_file = "requirements.txt"  # Asegúrate de tener este archivo en el mismo directorio
    check_and_install_requirements(requirements_file)

    # Define la ciudad y las columnas de entrada
    ciudad = 'Piura'

    # Directorio donde se encuentra el modelo y el archivo de escalado
    folder_path = os.path.join(os.getcwd())
    model_filename = 'modelo_completo.pth'
    scaler_path = 'escalado.pkl'  # Ruta a tu archivo de escalado

    # Inicialización de los gestores de base de datos y modelo
    from Get_data import DatabaseManager
    from ModelTraining import ModelTraining
    from TaskManager import TaskManager

    try:
        from LSTM_model import LSTM
        print("La clase LSTM se importó correctamente.")
    except ImportError as e:
        print(f"Error al importar la clase LSTM: {e}")
        sys.exit(1)


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