from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
from typing import Dict, Any

class VisualizadorResultados:
    """"
    Clase para visualizar los resultados de un modelo de predicción de temperaturas.
    Incluye funciones para graficar la evolución de métricas de entrenamiento,
    errores de predicción y comparaciones de temperaturas reales versus predicciones.
    """
    def __init__(self):
        """
        Inicializa el diccionario de colores utilizados para las gráficas.
        """
        self.colores = {
            'linea_principal': 'blue',
            'linea_prediccion': 'red',
            'dispersion': 'red',
            'histograma': 'red'
        }
    
    def graficar_historial_entrenamiento(self, historial: Dict[str, Any], 
                                         epoca_inicio: int = 0, 
                                         epoca_fin: int = None, 
                                         tamanio_anotacion: float = 0.01) -> None:
        """
        Grafica el historial de entrenamiento de las métricas de pérdida (loss), error absoluto medio (mae),
        y raíz del error cuadrático medio (rmse) a lo largo de las épocas.

        :param historial: Diccionario que contiene las métricas de entrenamiento y validación.
        :param epoca_inicio: Época a partir de la cual se graficarán las métricas.
        :param epoca_fin: Época hasta la cual se graficarán las métricas.
        :param tamanio_anotacion: Tamaño de la anotación en la gráfica.
        :return: None
        """
        if epoca_fin is None:
            epoca_fin = len(historial['loss'])
        
        fig, ejes = plt.subplots(1, 3, figsize=(18, 5))
        self._graficar_metrica(ejes[0], historial, 'loss', epoca_inicio, 
                              epoca_fin, tamanio_anotacion)
        self._graficar_metrica(ejes[1], historial, 'mae', epoca_inicio, 
                              epoca_fin, tamanio_anotacion)
        self._graficar_metrica(ejes[2], historial, 'rmse', epoca_inicio, 
                              epoca_fin, tamanio_anotacion)
        
        plt.tight_layout()
        plt.show()
    
    def graficar_errores_temperatura(self, y_real: np.ndarray, 
                                     y_prediccion: np.ndarray) -> None:
        """
        Grafica los errores de predicción de las temperaturas (máxima, mínima y media), incluyendo dispersión,
        distribución de errores y comparación temporal entre valores reales y predicciones.

        :param y_real: Valores reales de las temperaturas (máxima, mínima y media).
        :param y_prediccion: Valores predichos por el modelo para las temperaturas (máxima, mínima y media).
        :return: None
        """
        errores = self._calcular_errores(y_real, y_prediccion)
        metricas = self._calcular_metricas(y_real, y_prediccion)
        
        fig, ejes = plt.subplots(3, 3, figsize=(18, 20))
        
        self._graficar_dispersion_temperaturas(ejes[0], y_real, y_prediccion, metricas)
        self._graficar_distribucion_errores(ejes[1], errores)
        self._graficar_comparacion_temporal(ejes[2], y_real, y_prediccion)
        
        plt.tight_layout()
        plt.show()
    
    def _graficar_metrica(self, eje: plt.Axes, 
                          historial: Dict[str, Any], 
                          metrica: str, 
                          epoca_inicio: int, 
                          epoca_fin: int, 
                          tamanio_anotacion: float) -> None:
        """
        Grafica una métrica de entrenamiento (loss, mae, rmse) a lo largo de las épocas.

        :param eje: Eje de la gráfica donde se dibujará la métrica.
        :param historial: Diccionario que contiene las métricas de entrenamiento y validación.
        :param metrica: Nombre de la métrica a graficar.
        :param epoca_inicio: Época a partir de la cual se graficarán las métricas.
        :param epoca_fin: Época hasta la cual se graficarán las métricas.
        :param tamanio_anotacion: Tamaño de la anotación en la gráfica.
        :return: None
        """
        epocas = range(epoca_inicio, epoca_fin)
        eje.plot(epocas, historial[metrica][epoca_inicio:epoca_fin], 
                label=f'{metrica.capitalize()} Entrenamiento')
        
        eje.plot(epocas, historial[f'val_{metrica}'][epoca_inicio:epoca_fin], 
                label=f'{metrica.capitalize()} Validación')
        
        self._agregar_anotaciones(eje, historial, metrica, epoca_inicio, 
                                epoca_fin, tamanio_anotacion)
        
        eje.set_title(f'Evolución de {metrica.capitalize()}')
        eje.set_xlabel('Épocas')
        eje.set_ylabel(metrica.capitalize())
        eje.legend()
        eje.grid(True)

    def _agregar_anotaciones(self, eje: plt.Axes, 
                             historial: Dict[str, Any], 
                             metrica: str, 
                             epoca_inicio: int, 
                             epoca_fin: int, 
                             tamanio_anotacion: float) -> None:
        """
        Agrega anotaciones de los valores inicial y final de una métrica en el gráfico.

        :param eje: Eje de la gráfica donde se dibujará la anotación.
        :param historial: Diccionario que contiene las métricas de entrenamiento y validación.
        :param metrica: Nombre de la métrica a graficar.
        :param epoca_inicio: Época a partir de la cual se graficarán las métricas.
        :param epoca_fin: Época hasta la cual se graficarán las métricas.
        :param tamanio_anotacion: Tamaño de la anotación en la gráfica.
        :return
        """
        eje.annotate(
            f'Inicial: {historial[metrica][epoca_inicio]:.4f}',
            xy=(epoca_inicio, historial[metrica][epoca_inicio]),
            xytext=(epoca_inicio, historial[metrica][epoca_inicio] + tamanio_anotacion),
            arrowprops=dict(arrowstyle='->', color='black')
        )
        eje.annotate(
            f'Final: {historial[metrica][epoca_fin - 1]:.4f}',
            xy=(epoca_fin - 1, historial[metrica][epoca_fin - 1]),
            xytext=(epoca_fin - 1, historial[metrica][epoca_fin - 1] + tamanio_anotacion),
            arrowprops=dict(arrowstyle='->', color='black')
        )

    def _calcular_errores(self, y_real: np.ndarray, 
                          y_prediccion: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calcula los errores entre las temperaturas reales y las predicciones (máxima, mínima y media).
        
        :param y_real: Valores reales de las temperaturas (máxima, mínima y media).
        :param y_prediccion: Valores predichos por el modelo para las temperaturas (máxima, mínima y media).
        :return: Diccionario con los errores de predicción para cada tipo de temperatura.
        """
        return {
            'maxima': y_real[:, 0] - y_prediccion[:, 0],
            'minima': y_real[:, 1] - y_prediccion[:, 1],
            'media': y_real[:, 2] - y_prediccion[:, 2]
        }
    
    def _calcular_metricas(self, y_real: np.ndarray, 
                           y_prediccion: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calcula las métricas (MAE, RMSE, R, R2) para las temperaturas máximas, mínimas y medias.

        :param y_real: Valores reales de las temperaturas (máxima, mínima y media).
        :param y_prediccion: Valores predichos por el modelo para las temperaturas (máxima, mínima y media).
        :return: Diccionario con las métricas de rendimiento para cada tipo de temperatura.
        """
        metricas = {}
        for i, tipo in enumerate(['maxima', 'minima', 'media']):
            metricas[tipo] = {
                'mae': mean_absolute_error(y_real[:, i], y_prediccion[:, i]),
                'rmse': np.sqrt(mean_squared_error(y_real[:, i], y_prediccion[:, i])),
                'r2': r2_score(y_real[:, i], y_prediccion[:, i]),
                'r': np.corrcoef(y_real[:, i], y_prediccion[:, i])[0, 1]
            }
        return metricas
    
    def _graficar_dispersion_temperaturas(self, ejes: np.ndarray, 
                                          y_real: np.ndarray, 
                                          y_prediccion: np.ndarray, 
                                          metricas: Dict[str, Dict[str, float]]) -> None:
        """
        Grafica la dispersión de las temperaturas reales versus las predicciones.

        :param ejes: Ejes de la gráfica donde se dibujarán las dispersiones.
        :param y_real: Valores reales de las temperaturas (máxima, mínima y media).
        :param y_prediccion: Valores predichos por el modelo para las temperaturas (máxima, mínima y media).
        :param metricas: Diccionario con las métricas de rendimiento para cada tipo de temperatura.
        :return: None
        """
        for i, (eje, tipo) in enumerate(zip(ejes, ['maxima', 'minima', 'media'])):
            self._graficar_dispersion_individual(
                eje, y_real[:, i], y_prediccion[:, i], 
                tipo, metricas[tipo]
            )
    
    def _graficar_dispersion_individual(self, eje: plt.Axes, 
                                        real: np.ndarray, 
                                        prediccion: np.ndarray, 
                                        tipo: str, 
                                        metricas: Dict[str, float]) -> None:
        """
        Grafica la dispersión de un tipo de temperatura (máxima, mínima o media).

        :param eje: Eje de la gráfica donde se dibujará la dispersión.
        :param real: Valores reales de la temperatura.
        :param prediccion: Valores predichos por el modelo.
        :param tipo: Tipo de temperatura (máxima, mínima o media).
        :param metricas: Diccionario con las métricas de rendimiento para el tipo de temperatura.
        :return: None
        """
        sns.scatterplot(ax=eje, x=real, y=prediccion, alpha=0.70)
        sns.regplot(ax=eje, x=real, y=prediccion, scatter=False, 
                   color='red', line_kws={'linewidth': 2})
        eje.plot([real.min(), real.max()], 
                [real.min(), real.max()], 
                color='black', linestyle='--')
        
        eje.set_title(
            f'Temperatura {tipo.capitalize()}:\n'
            f'MAE={metricas["mae"]:.4f}, RMSE={metricas["rmse"]:.4f}, '
            f'R={metricas["r"]:.4f}, R2={metricas["r2"]:.4f}'
        )
        eje.set_xlabel('Real (°C)')
        eje.set_ylabel('Predicción (°C)')
    
    def _graficar_distribucion_errores(self, ejes: np.ndarray, 
                                       errores: Dict[str, np.ndarray]) -> None:
        """
        Grafica la distribución de los errores de predicción para las temperaturas máximas, mínimas y medias.

        :param ejes: Ejes de la gráfica donde se dibujarán los histogramas de errores.
        :param errores: Diccionario con los errores de predicción para cada tipo de temperatura.
        :return: None
        """
        for eje, (tipo, error) in zip(ejes, errores.items()):
            sns.histplot(error, ax=eje, kde=True, color='red', 
                        bins=20, fill=True, alpha=0.15)
            eje.set_title(
                f'Distribución de Error (Temperatura {tipo.capitalize()})\n'
                f'MAX: {error.max():.2f} MIN: {error.min():.2f}'
            )
            eje.set_xlabel('Error (°C)')
            eje.set_ylabel('Frecuencia')
    
    def _graficar_comparacion_temporal(self, ejes: np.ndarray, 
                                       y_real: np.ndarray, 
                                       y_prediccion: np.ndarray) -> None:
        """
        Grafica la comparación temporal de las temperaturas reales versus las predicciones.

        :param ejes: Ejes de la gráfica donde se dibujarán las comparaciones temporales.
        :param y_real: Valores reales de las temperaturas (máxima, mínima y media).
        :param y_prediccion: Valores predichos por el modelo para las temperaturas (máxima, mínima y media).
        :return: None
        """
        for i, (eje, tipo) in enumerate(zip(ejes, ['maxima', 'minima', 'media'])):
            eje.plot(y_real[:, i], label='Real (°C)', 
                    color=self.colores['linea_principal'])
            eje.plot(y_prediccion[:, i], label='Predicción (°C)', 
                    color=self.colores['linea_prediccion'])
            eje.set_title(f'Temperatura {tipo.capitalize()} (Real vs Predicción)')
            eje.set_xlabel('Día')
            eje.set_ylabel('Temperatura (°C)')
            eje.legend()
            eje.grid(True)