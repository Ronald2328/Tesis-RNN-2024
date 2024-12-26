import os
from modelo_lstm import LSTM
from gestor_requisitos import verificar_e_instalar_requisitos

def main():
    verificar_e_instalar_requisitos()

    from gestor_datos_climaticos import GestorDatosClimaticos
    from gestor_predicciones import GestorPredicciones
    from gestor_tareas import GestorTareas

    data_manager = GestorDatosClimaticos(servidor='localhost', 
                                   usuario='root', 
                                   contrasena='root', 
                                   base_datos='meteorology')
    
    model_trainer = GestorPredicciones(gestor_datos=data_manager, 
                                       ruta_escalador='escalado.pkl')

    task_manager = GestorTareas(actualizador_datos=data_manager, 
                               entrenador_modelo=model_trainer, 
                               ruta_carpeta=os.path.join(os.getcwd()), 
                               nombre_modelo='modelo_completo.pth', 
                               ciudad='Piura')

    task_manager.limpiar_terminal()
    task_manager.ejecutar()

if __name__ == "__main__":
    main()