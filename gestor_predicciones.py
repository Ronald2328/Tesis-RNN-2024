import os
import joblib
import numpy as np
import pandas as pd
import torch
from gestor_datos_climaticos import GestorDatosClimaticos
import logging

class GestorPredicciones:
    """
    Clase para gestionar las predicciones usando modelos pre-entrenados.
    """
    def __init__(self, gestor_datos: GestorDatosClimaticos, 
                 ruta_escalador='escalado.pkl') -> None:
        """
        Inicializa el gestor de entrenamiento de modelos.

        :param gestor_datos: Instancia del gestor de la base de datos.
        :param ruta_escalador: Ruta al archivo del escalador.
        :return: None
        :nota: Configura el logger y carga el escalador.
        """
        self.escalador = joblib.load(ruta_escalador)
        self.gestor_datos = gestor_datos
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def determinar_estacion(self, mes: int, dia: int) -> str:
        """
        Determina la estación del año basada en el mes y el día proporcionados.

        :param mes: Mes del año (1-12).
        :param dia: Día del mes (1-31).
        :return: La estación del año correspondiente.
        """
        if (mes == 12 and dia >= 1) or mes in [1, 2]:
            return 'Verano'
        if mes in [3, 4, 5]:
            return 'Otoño'
        if mes in [6, 7, 8]:
            return 'Invierno'
        if mes in [9, 10, 11]:
            return 'Primavera'
        return None

    def cargar_modelo(self, ruta_carpeta: str, 
                      nombre_modelo: str, modelo=None) -> torch.nn.Module:
        """
        Carga un modelo previamente entrenado.

        :param ruta_carpeta: Ruta de la carpeta del modelo.
        :param nombre_modelo: Nombre del archivo del modelo.
        :param modelo: Modelo pre-inicializado (opcional).
        :return: Modelo cargado.
        :nota: Solo soporta modelos PyTorch (.pth).
        """
        ruta_modelo = os.path.join(ruta_carpeta, nombre_modelo)
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f'El archivo {ruta_modelo} no existe.\n')

        if nombre_modelo.endswith('.pth'):
            modelo = torch.load(ruta_modelo)
            modelo.eval()
            self.logger.info(f"Modelo PyTorch cargado exitosamente desde {ruta_modelo}.\n")
            return modelo
        else:
            raise ValueError("Formato de modelo no soportado. Asegúrese de que el archivo sea .pth (PyTorch).\n")

    def preparar_datos(self, df: pd.DataFrame, 
                       agregar_columnas_mes: bool = True, 
                       agregar_columnas_estacion: bool = True) -> pd.DataFrame:
        """
        Prepara los datos para el entrenamiento o predicción.

        :param df: DataFrame con los datos a preparar.
        :param agregar_columnas_mes: Si se deben agregar columnas dummy para meses.
        :param agregar_columnas_estacion: Si se deben agregar columnas dummy para estaciones.
        :return: DataFrame procesado.
        :nota: Incluye codificación one-hot para meses si se especifica.
        """
        df['time'] = pd.to_datetime(df['time'])
        df['mes'] = df['time'].dt.month
        df['dia'] = df['time'].dt.day
        df['estacion'] = df.apply(lambda row: self.determinar_estacion(row['mes'], row['dia']), axis=1)

        if agregar_columnas_estacion:
            estaciones_dummies = pd.get_dummies(df['estacion'], prefix='', drop_first=False).astype(int)
            todas_estaciones = ['Verano', 'Otoño', 'Invierno', 'Primavera']
            estaciones_dummies = estaciones_dummies.reindex(columns=todas_estaciones, fill_value=0)
            df = pd.concat([df, estaciones_dummies], axis=1)

        if agregar_columnas_mes:
            meses_dummies = pd.get_dummies(df['mes'], prefix='mes').astype(int)
            todos_meses = [f'mes_{i}' for i in range(1, 13)]
            meses_dummies = meses_dummies.reindex(columns=todos_meses, fill_value=0)
            df = pd.concat([df, meses_dummies], axis=1)

        return df

    def generar_prediccion_climatica(self, modelo_final: torch.nn.Module, 
                                     df_combinado: pd.DataFrame, 
                                     nombre_ciudad: str, 
                                     tamano_ventana: int, 
                                     caracteristicas: list=None, 
                                     agregar_columnas_mes: bool=True) -> pd.DataFrame:
        """
        Obtiene predicciones climáticas para una ciudad.

        :param nombre_ciudad: Nombre de la ciudad a predecir.
        :param modelo_final: Modelo pre-entrenado para predicción.
        :return: DataFrame con las predicciones generadas.
        :nota: Utiliza datos históricos para generar predicción del siguiente día.
        """
        df_combinado = df_combinado.dropna(axis=1, how='all')
        df_combinado = self.preparar_datos(df_combinado, agregar_columnas_mes)

        caracteristicas = [col for col in df_combinado.columns if col not in ['id_ciudad', 'time', 'snow', 'wpgt', 'tsun',
                                                                              'estacion','Primavera','mes_7','mes_8',
                                                                              'mes_9','mes_10','mes_11','mes_12']]
        
        datos = df_combinado[caracteristicas].values
        datos = self.escalador.transform(datos)

        ultima_secuencia = datos[-tamano_ventana:]
        ultima_secuencia = np.expand_dims(ultima_secuencia, axis=0)
        ultima_secuencia = torch.tensor(ultima_secuencia, dtype=torch.float32)

        modelo_final.eval()
        with torch.no_grad():
            prediccion = modelo_final(ultima_secuencia)

        prediccion = prediccion.squeeze().cpu().numpy()
        prediccion = np.round(prediccion, decimals=1)

        id_ciudad = self.gestor_datos.obtener_id_ciudad_por_nombre(nombre_ciudad)
        if id_ciudad is None:
            raise ValueError(f"No se pudo obtener el ID de la ciudad '{nombre_ciudad}'.\n")

        df_predicciones = pd.DataFrame({
            'id': [id_ciudad],
            'time': [pd.Timestamp(df_combinado['time'].iloc[-1]) + pd.DateOffset(days=1)],
            'tmax': [prediccion[0]],
            'tmin': [prediccion[1]],
            'tavg': [prediccion[2]]
        })

        print(f"Predicción realizada para la ciudad {nombre_ciudad}:\n{df_predicciones}\n")
        return df_predicciones
    