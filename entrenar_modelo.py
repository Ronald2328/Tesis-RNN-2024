import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch.nn as nn
import torch

class EntrenadorModelo:
    """"
    Esta clase se encarga de la creación, entrenamiento y evaluación de un modelo de 
    predicción de temperaturas utilizando redes neuronales. 
    """
    def __init__(self, modelo: nn.Module, 
                 tasa_aprendizaje: float) -> None:
        """
        Inicializa el entrenador de modelo.

        :param modelo: El modelo de red neuronal a entrenar.
        :param tasa_aprendizaje: Tasa de aprendizaje para el optimizador.
        :return: None
        """
        self.modelo = modelo
        self.tasa_aprendizaje = tasa_aprendizaje
        self.criterio = nn.MSELoss()
        self.optimizador = torch.optim.SGD(
            modelo.parameters(), 
            lr=tasa_aprendizaje, 
            momentum=0.9, 
            weight_decay=1e-4
        )
    
    def entrenar(self, x_entrenamiento: torch.Tensor, 
                 y_entrenamiento: torch.Tensor, 
                 x_prueba: torch.Tensor, 
                 y_prueba: torch.Tensor, 
                 epocas: int = 200) -> dict:
        """
        Entrena el modelo y registra métricas de rendimiento.

        :param x_entrenamiento: Datos de entrenamiento (entradas).
        :param y_entrenamiento: Datos de entrenamiento (salidas).
        :param x_prueba: Datos de validación (entradas).
        :param y_prueba: Datos de validación (salidas).
        :param epocas: Número de épocas para entrenar el modelo (valor predeterminado 200).
        :return: Diccionario con el historial de métricas (pérdida, MAE, RMSE, R²).
        """
        historial = self._inicializar_historial(epocas)
        
        for epoca in range(epocas):
            self._ejecutar_epoca_entrenamiento(
                epoca, 
                x_entrenamiento, 
                y_entrenamiento, 
                x_prueba, 
                y_prueba, 
                historial
            )
            
            if self._debe_mostrar_progreso(epoca):
                self._mostrar_progreso(epoca, epocas, historial)
        
        return historial
    
    def _inicializar_historial(self, epoca: int) -> dict:
        """
        Inicializa un diccionario para almacenar métricas de rendimiento.

        :param epoca: Número de épocas para definir el tamaño de las métricas.
        :return: Diccionario vacío con las claves de las métricas y sus valores iniciales en cero.
        """
        return {
            "loss": np.zeros(epoca),
            "val_loss": np.zeros(epoca),
            "mae": np.zeros(epoca), 
            "val_mae": np.zeros(epoca),
            "rmse": np.zeros(epoca), 
            "val_rmse": np.zeros(epoca),
            "r2": np.zeros(epoca),
            "val_r2": np.zeros(epoca)
        }
    
    def _ejecutar_epoca_entrenamiento(self, epoca: int, 
                                      x_entrenamiento: torch.Tensor, 
                                      y_entrenamiento: torch.Tensor, 
                                      x_prueba: torch.Tensor, 
                                      y_prueba: torch.Tensor, 
                                      historial: dict) -> None:
        """
        Ejecuta una época de entrenamiento, realizando el cálculo de la pérdida, la retropropagación
        y la optimización del modelo.

        :param epoca: Número de la época actual.
        :param x_entrenamiento: Datos de entrenamiento (entradas).
        :param y_entrenamiento: Etiquetas de entrenamiento (salidas).
        :param x_prueba: Datos de validación (entradas).
        :param y_prueba: Etiquetas de validación (salidas).
        :param historial: Historial donde se almacenan las métricas de cada época.
        :return: None
        """
        self.optimizador.zero_grad()
        predicciones = self.modelo(x_entrenamiento)
        perdida = self.criterio(predicciones, y_entrenamiento)
        perdida.backward()
        self.optimizador.step()
        
        self._actualizar_metricas_entrenamiento(
            epoca, historial, perdida, y_entrenamiento, predicciones
        )
        self._actualizar_metricas_validacion(
            epoca, historial, x_prueba, y_prueba
        )
    
    def _actualizar_metricas_entrenamiento(self, epoca: int, 
                                           historial: dict, 
                                           perdida: torch.Tensor, 
                                           y_real: torch.Tensor, 
                                           predicciones: torch.Tensor) -> None:
        """
        Actualiza las métricas de rendimiento para la época de entrenamiento actual.

        :param epoca: Número de la época actual.
        :param historial: Diccionario que contiene las métricas de todas las épocas.
        :param perdida: Pérdida calculada para la época actual.
        :param y_real: Etiquetas reales de los datos de entrenamiento.
        :param predicciones: Predicciones generadas por el modelo para los datos de entrenamiento.
        :return: None
        """
        y_real_np = y_real.detach().cpu().numpy()
        pred_np = predicciones.detach().cpu().numpy()
        
        historial["loss"][epoca] = perdida.item()
        historial["mae"][epoca] = mean_absolute_error(y_real_np, pred_np)
        historial["rmse"][epoca] = np.sqrt(mean_squared_error(y_real_np, pred_np))
        historial["r2"][epoca] = r2_score(y_real_np, pred_np)
    
    def _actualizar_metricas_validacion(self, epoca: int, 
                                        historial: dict, 
                                        x_prueba: torch.Tensor, 
                                        y_prueba: torch.Tensor) -> None:
        """
        Actualiza las métricas de rendimiento para la época de validación actual.

        :param epoca: Número de la época actual.
        :param historial: Diccionario que contiene las métricas de todas las épocas.
        :param x_prueba: Datos de validación (entradas).
        :param y_prueba: Etiquetas de validación (salidas).
        :return: None
        """
        with torch.no_grad():
            predicciones = self.modelo(x_prueba)
            perdida = self.criterio(predicciones, y_prueba)
            
            y_prueba_np = y_prueba.detach().cpu().numpy()
            pred_np = predicciones.detach().cpu().numpy()
            
            historial["val_loss"][epoca] = perdida.item()
            historial["val_mae"][epoca] = mean_absolute_error(y_prueba_np, pred_np)
            historial["val_rmse"][epoca] = np.sqrt(mean_squared_error(y_prueba_np, pred_np))
            historial["val_r2"][epoca] = r2_score(y_prueba_np, pred_np)
    
    def _debe_mostrar_progreso(self, epoca: int, 
                               intervalo: int = 50) -> bool:
        """
        Determina si se debe mostrar el progreso de entrenamiento.

        :param epoca: Número de la época actual.
        :param intervalo: Intervalo de épocas para mostrar el progreso (por defecto es cada 50).
        :return: True si se debe mostrar el progreso, False en caso contrario.
        """
        return (epoca + 1) % intervalo == 0
    
    def _mostrar_progreso(self, epoca: int, 
                          epocas_totales: int, 
                          historial: dict) -> None:
        """
        Muestra el progreso de entrenamiento en la consola.

        :param epoca: Número de la época actual.
        :param epocas_totales: Total de épocas de entrenamiento.
        :param historial: Diccionario con el historial de las métricas de entrenamiento.
        :return: None
        """
        print(
            f'Epoch {epoca + 1}/{epocas_totales}, '
            f'Loss: {historial["loss"][epoca]:.3f}, '
            f'Val Loss: {historial["val_loss"][epoca]:.3f}, '
            f'MAE: {historial["mae"][epoca]:.3f}, '
            f'Val MAE: {historial["val_mae"][epoca]:.3f}, '
            f'RMSE: {historial["rmse"][epoca]:.3f}, '
            f'Val RMSE: {historial["val_rmse"][epoca]:.3f}, '
            f'R²: {historial["r2"][epoca]:.3f}, '
            f'Val R²: {historial["val_r2"][epoca]:.3f}'
        )