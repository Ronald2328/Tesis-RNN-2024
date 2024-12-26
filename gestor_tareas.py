import os
import numpy as np
import pandas as pd
import logging
from gestor_datos_climaticos import GestorDatosClimaticos
from gestor_predicciones import GestorPredicciones

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GestorTareas:
    """
    Clase para gestionar tareas de predicción meteorológica.
    """
    def __init__(self, actualizador_datos: GestorDatosClimaticos, 
                 entrenador_modelo: GestorPredicciones, 
                 ruta_carpeta: str, 
                 nombre_modelo: str, 
                 ciudad: str = 'Piura') -> None:
        """
        Inicializa el gestor de tareas.

        :param actualizador_datos: Instancia del gestor de datos climáticos.
        :return: None
        :nota: Requiere una instancia válida de GestorDatosClimaticos.
        """
        self.actualizador_datos = actualizador_datos
        self.entrenador_modelo = entrenador_modelo
        self.ruta_carpeta = ruta_carpeta
        self.nombre_modelo = nombre_modelo
        self.ciudad = ciudad  

    def ejecutar(self) -> None:
        """
        Ejecuta el ciclo principal del gestor de tareas.

        :return: None
        :nota: Maneja el menú interactivo y las operaciones principales.
        """
        try:
            self.actualizador_datos.actualizar_datos_climaticos(self.ciudad)
            id_ciudad = self.actualizador_datos.obtener_id_ciudad_por_nombre(self.ciudad)
            df_combinado = self.actualizador_datos.obtener_dataframe('clima', id_ciudad)
            modelo_final = self.entrenador_modelo.cargar_modelo(self.ruta_carpeta, self.nombre_modelo)

            ultima_fecha, ultima_fecha_prediccion = self.actualizador_datos.obtener_ultima_fecha_ciudad(self.ciudad), \
                                                    self.actualizador_datos.obtener_ultima_fecha_prediccion(self.ciudad)
            diferencia_dias = (ultima_fecha - ultima_fecha_prediccion).days

            if diferencia_dias >= 0:
                tamano_ventana = 20
                self.procesar_predicciones(df_combinado, modelo_final, diferencia_dias, tamano_ventana)
            else:
                self.imprimir_siguiente_prediccion()

        except Exception as e:
            logger.error(f"Ocurrió un error al ejecutar el GestorTareas: {e}\n")

    def procesar_predicciones(self, df_combinado: pd.DataFrame, modelo_final: object, 
                            diferencia_dias: int, tamano_ventana: int) -> None:
        """
        Procesa las predicciones climáticas para una ciudad.

        :param ciudad: Nombre de la ciudad a procesar.
        :return: None
        :nota: Actualiza la base de datos con nuevas predicciones.
        """
        try:
            columnas = df_combinado.columns
            ultimos_datos = df_combinado.iloc[-(tamano_ventana + diferencia_dias):].copy()
            if diferencia_dias == 0:
                secuencias = [ultimos_datos.iloc[-tamano_ventana:].values]
            else:
                secuencias = [ultimos_datos.iloc[i:i + tamano_ventana].values for i in range(diferencia_dias+1)]
            
            for secuencia in secuencias:
                df_secuencia = pd.DataFrame(secuencia, columns=columnas)
                df_predicciones = self.entrenador_modelo.generar_prediccion_climatica(modelo_final, 
                                                                                      df_secuencia, 
                                                                                      self.ciudad, 
                                                                                      tamano_ventana)
                self.actualizador_datos.insertar_datos('predic', df_predicciones)

        except Exception as e:
            logger.error(f"Error al procesar las predicciones: {e}\n")

    def imprimir_siguiente_prediccion(self) -> None:
        """
        Muestra la predicción climática para el siguiente día.

        :param self: Instancia de la clase.
        :return: None
        :nota: Imprime la predicción en formato tabular usando pandas.
        """
        try:
            ultima_fecha_prediccion = self.actualizador_datos.obtener_ultima_fecha_prediccion(self.ciudad, False)
            prediccion = self.actualizador_datos.siguiente_prediccion(self.ciudad)
            df_prediccion = pd.DataFrame([prediccion])
            logger.info(f"\nPredicción para el próximo día ({ultima_fecha_prediccion}):\n{df_prediccion}\n")
        except Exception as e:
            logger.error(f"No se pudo obtener la siguiente predicción: {e}\n")

    @staticmethod
    def limpiar_terminal() -> None:
        """
        Limpia la pantalla del terminal.

        :return: None
        :nota: Compatible con Windows (cls).
        """
        os.system('cls' if os.name == 'nt' else 'clear')