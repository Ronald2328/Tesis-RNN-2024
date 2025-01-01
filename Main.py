import os
from modelo_lstm import LSTM
from gestor_requisitos import verificar_e_instalar_requisitos
from configuracion import Configuracion

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia las advertencias de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactiva las operaciones oneDNN para evitar errores

def main():
    verificar_e_instalar_requisitos()

    configurador = Configuracion() # Cargar configuraciones desde el archivo JSON

    from gestor_datos_climaticos import GestorDatosClimaticos
    from gestor_predicciones import GestorPredicciones
    from gestor_tareas import GestorTareas

    data_manager = GestorDatosClimaticos(servidor=configurador.obtener_servidor(), 
                                         usuario=configurador.obtener_usuario(), 
                                         contrasena=configurador.obtener_contrasena(), 
                                         base_datos=configurador.obtener_base_datos())
    
    model_trainer = GestorPredicciones(gestor_datos=data_manager, 
                                       ruta_escalador='escalado.pkl')

    task_manager = GestorTareas(actualizador_datos=data_manager, 
                                entrenador_modelo=model_trainer, 
                                ruta_carpeta=os.path.join(os.getcwd()), 
                                nombre_modelo='modelo_completo.pth', 
                                ciudad=configurador.obtener_ciudad())

    task_manager.limpiar_terminal()
    task_manager.ejecutar()

if __name__ == "__main__":
    main()