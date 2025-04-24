import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from meteostat import Daily, Point
from typing import Optional, Tuple
from inicializador_de_bd import InicializadorBaseDatos
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class GestorDatosClimaticos:
    """
    Clase para gestionar la conexión y operaciones con la base de datos.
    """
    def __init__(self, servidor: str = 'localhost', 
                 usuario: str = 'root', 
                 contrasena: str = 'root', 
                 base_datos: str = 'meteorology'):
        """
        Inicializa el gestor de la base de datos con los parámetros de conexión.

        :param servidor: Dirección del servidor de la base de datos.
        :param usuario: Usuario de la base de datos.
        :param contrasena: Contraseña del usuario de la base de datos.
        :param base_datos: Nombre de la base de datos.
        """
        self.url_servidor: str = f"mysql+pymysql://{usuario}:{contrasena}@{servidor}/"
        self.url_base_datos: str = f"mysql+pymysql://{usuario}:{contrasena}@{servidor}/{base_datos}"
        self.base_datos = base_datos
        self.inicializador = InicializadorBaseDatos(self)

    def conectar_base_datos(self) -> None:
        """
        Intenta conectar con la base de datos. Si no existe, la crea.

        :return: Objeto de conexión a la base de datos.
        """
        try:
            return create_engine(self.url_base_datos).connect()
        except Exception as e:
            print("Base de datos no encontrada. Procediendo a crearla...\n")
            self.inicializador.crear_base_datos()
            self.inicializador.crear_tablas()
            print(f"Base de datos {self.base_datos} creada correctamente.\n")
            return create_engine(self.url_base_datos).connect()

    def cerrar_conexion(self, conexion: Optional[object]) -> None:
        """
        Cierra la conexión con la base de datos MySQL.

        :param conexion: Objeto de conexión a la base de datos.
        """
        if conexion:
            conexion.close()

    def obtener_dataframe(self, tabla: str, 
                          filtro_ciudad: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos de una tabla específica y los retorna como DataFrame.

        :param tabla: Nombre de la tabla a consultar.
        :param filtro_ciudad: ID de la ciudad para filtrar los datos.
        :return: DataFrame con los datos de la tabla.
        """
        conexion = self.conectar_base_datos()
        try:
            consulta: str = f"SELECT * FROM {tabla}"
            df: pd.DataFrame = pd.read_sql(consulta, conexion)
            if filtro_ciudad is not None and 'id_ciudad' in df.columns:
                df = df[df['id_ciudad'] == filtro_ciudad]
            return df
        except Exception as e:
            print(f"Error al cargar datos de la tabla {tabla}: {e}\n")
            return pd.DataFrame()
        finally:
            self.cerrar_conexion(conexion)

    def insertar_datos(self, tabla: str, 
                       datos: pd.DataFrame, 
                       columna_ignorar: Optional[str] = None) -> None:
        """
        Inserta datos en la tabla especificada de la base de datos.

        :param tabla: Nombre de la tabla donde insertar los datos.
        :param datos: DataFrame con los datos a insertar.
        :param columna_ignorar: Nombre de la columna a ignorar en la comparación de duplicados.
        """
        conexion = self.conectar_base_datos()
        try:
            datos = datos.fillna(0)
            datos_existentes: pd.DataFrame = self.obtener_dataframe(tabla)
            if columna_ignorar and columna_ignorar in datos.columns:
                datos_sin_ignorar: pd.DataFrame = datos.drop(columns=[columna_ignorar])
            else:
                datos_sin_ignorar: pd.DataFrame = datos
            
            datos_existentes_filtrados: pd.DataFrame = datos_existentes[datos_sin_ignorar.columns]
            nuevos_datos_filtrados: pd.DataFrame = datos_sin_ignorar
            duplicados: pd.DataFrame = nuevos_datos_filtrados.merge(datos_existentes_filtrados, 
                                                                    how='inner', 
                                                                    on=datos_sin_ignorar.columns.tolist())
            
            if not duplicados.empty:
                print(f"Los siguientes datos ya existen en la tabla {tabla}:")
                print(f'{duplicados}\n')
                return 
            
            datos.to_sql(tabla, con=conexion, if_exists='append', index=False)
            print(f"Datos insertados correctamente en la tabla {tabla}.\n")
        except Exception as e:
            print(f"Error al insertar datos en la tabla {tabla}: {e}\n")
        finally:
            self.cerrar_conexion(conexion)

    def obtener_nombre_ciudad_por_id(self, id_ciudad: int) -> Optional[str]:
        """
        Obtiene el nombre de una ciudad según su ID.

        :param id_ciudad: ID de la ciudad a buscar.
        :return: Nombre de la ciudad.
        """
        df: pd.DataFrame = self.obtener_dataframe('ciudades', id_ciudad)
        if not df.empty:
            return df['ciudad'].iloc[0]
        return None

    def obtener_id_ciudad_por_nombre(self, ciudad: str) -> Optional[int]:
        """
        Obtiene el ID de una ciudad según su nombre.

        :param ciudad: Nombre de la ciudad a buscar.
        :return: ID de la ciudad.
        """
        df: pd.DataFrame = self.obtener_dataframe('ciudades')
        fila_ciudad: pd.DataFrame = df[df['ciudad'] == ciudad]
        if not fila_ciudad.empty:
            return fila_ciudad['id'].iloc[0]
        return None

    def obtener_coordenadas_ciudad(self, ciudad: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtiene las coordenadas geográficas de una ciudad.

        :param ciudad: Nombre de la ciudad a buscar.
        :return: Tupla con latitud y longitud de la ciudad.
        """
        id_ciudad: Optional[int] = self.obtener_id_ciudad_por_nombre(ciudad)
        if id_ciudad is None:
            print(f"No se encontró la ciudad {ciudad}. Solicitando coordenadas...\n")
            if self.manejar_ciudad_faltante(ciudad):
                return self.obtener_coordenadas_ciudad(ciudad)
            else:
                return None, None

        df: pd.DataFrame = self.obtener_dataframe('ciudades')
        fila_ciudad: pd.DataFrame = df[df['id'] == id_ciudad]
        return fila_ciudad['latitud'].iloc[0], fila_ciudad['longitud'].iloc[0]

    def manejar_ciudad_faltante(self, ciudad: str) -> bool:
        """
        Maneja el caso de una ciudad no encontrada en la base de datos.

        :param ciudad: Nombre de la ciudad a agregar.
        :return: True si la ciudad se agregó correctamente, False en caso contrario
        """
        lat, lon = self.solicitar_coordenadas_ciudad(ciudad)
        if self.verificar_ciudad_en_api(lat, lon):
            self.agregar_ciudad_a_db(ciudad, lat, lon)
            return True
        else:
            print(f"Las coordenadas proporcionadas para {ciudad} no son válidas. Intente de nuevo.\n")
            return False
    
    def agregar_ciudad_a_db(self, ciudad: str, 
                            lat: float, 
                            lon: float) -> None:
        """
        Agrega una nueva ciudad a la base de datos.

        :param ciudad: Nombre de la ciudad a agregar.
        :param lat: Latitud de la ciudad.
        :param lon: Longitud de la ciudad.
        """
        df: pd.DataFrame = self.obtener_dataframe('ciudades')
        id_maximo: int = df['id'].max() if not df.empty else 0 
        nuevo_id: int = id_maximo + 1

        datos_nueva_ciudad: pd.DataFrame = pd.DataFrame({
            'id': [nuevo_id],
            'ciudad': [ciudad],
            'latitud': [lat],
            'longitud': [lon]
        })

        self.insertar_datos('ciudades', datos_nueva_ciudad)
        print(f"Ciudad {ciudad} agregada con ID {nuevo_id} a la base de datos.\n")    

    def solicitar_coordenadas_ciudad(self, ciudad: str) -> Tuple[float, float]:
        """
        Solicita las coordenadas de una ciudad al usuario.

        :param ciudad: Nombre de la ciudad a buscar.
        :return: Tupla con latitud y longitud de la ciudad.
        """
        while True:
            try:
                lat: float = float(input(f"Ingrese la latitud de {ciudad}: "))
                lon: float = float(input(f"Ingrese la longitud de {ciudad}: "))
                return lat, lon
            except ValueError:
                print("Por favor, ingrese un valor numérico válido para la latitud y longitud.\n")

    def verificar_ciudad_en_api(self, lat: float, lon: float) -> bool:
        """
        Verifica si una ciudad está disponible en la API de clima.

        :param lat: Latitud de la ciudad.
        :param lon: Longitud de la ciudad.
        :return: True si la ciudad es válida, False en caso contrario.
        """
        try:
            punto: Point = Point(lat, lon)
            datos: pd.DataFrame = Daily(punto, datetime(2023, 1, 1), datetime(2023, 1, 2)).fetch()
            if not datos.empty:
                print("La estación es válida y se encontró en la API.\n")
                return True
            print("La estación no es válida o no se encontraron datos en la API.\n")
            return False
        except Exception as e:
            print(f"Error verificando la estación en la API: {e}\n")
            return False

    def obtener_ultima_fecha_ciudad(self, ciudad: str) -> Optional[datetime]:
        """
        Obtiene la última fecha de datos climáticos para una ciudad.

        :param ciudad: Nombre de la ciudad a buscar.
        :return: Última fecha registrada en la base de datos.
        """
        id_ciudad: Optional[int] = self.obtener_id_ciudad_por_nombre(ciudad)
        if id_ciudad is None:
            print(f"No se encontró el ID para la ciudad {ciudad}.\n")
            return None
        
        df: pd.DataFrame = self.obtener_dataframe('clima', id_ciudad)
        if not df.empty:
            ultima_fecha: datetime = df['time'].max()
            print(f"La última fecha registrada en la tabla 'clima' para {ciudad} es: {ultima_fecha}\n")
            return ultima_fecha
        else:
            print(f"No se encontraron datos para la ciudad {ciudad}.\n")
            return None

    def obtener_ultima_fecha_prediccion(self, ciudad: str, 
                                        mostrar: bool = True) -> datetime:
        """
        Obtiene la última fecha de predicción para una ciudad.

        :param ciudad: Nombre de la ciudad a buscar.
        :param mostrar: Indica si se debe mostrar la fecha en consola.
        :return: Última fecha de predicción registrada.
        """
        id_ciudad: Optional[int] = self.obtener_id_ciudad_por_nombre(ciudad)
        if id_ciudad is None:
            print(f"No se encontró el ID para la ciudad {ciudad}.\n")
            return datetime.now()
        
        df: pd.DataFrame = self.obtener_dataframe('predic', id_ciudad)
        if not df.empty:
            ultima_fecha: datetime = df['time'].max()
            if mostrar:
                print(f"La última fecha de predicción en la tabla 'predic' para {ciudad} es: {ultima_fecha}\n")
            return ultima_fecha
        else:
            print(f"No se encontraron datos de predicción para la ciudad {ciudad}. Se devolverá la fecha actual.\n")
            return datetime.now()

    def obtener_datos_de_api(self, ciudad: str, 
                             ultima_fecha: Optional[datetime]) -> pd.DataFrame:
        """
        Obtiene datos climáticos históricos de la API externa.

        :param ciudad: Nombre de la ciudad a buscar.
        :param ultima_fecha: Última fecha registrada en la base de datos.
        :return: DataFrame con los datos climáticos.
        """
        lat, lon = self.obtener_coordenadas_ciudad(ciudad)
        fecha_inicio: datetime = ultima_fecha + pd.DateOffset(days=1) if ultima_fecha else datetime(2018, 1, 1)
        try:
            datos: pd.DataFrame = Daily(Point(lat, lon), fecha_inicio, datetime.now()).fetch()  
            if not datos.empty:
                datos = datos.reset_index()
                id_ciudad: Optional[int] = self.obtener_id_ciudad_por_nombre(ciudad)
                datos['id_ciudad'] = id_ciudad
                return datos
            else:
                print(f"No se encontraron datos para la ciudad {ciudad} en la API.\n")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error al obtener datos de la API para {ciudad}: {e}\n")
            return pd.DataFrame()

    def actualizar_datos_climaticos(self, ciudad: str) -> None:
        """
        Actualiza los datos climáticos de una ciudad.

        :param ciudad: Nombre de la ciudad a actualizar.
        """
        try:
            ultima_fecha: Optional[datetime] = self.obtener_ultima_fecha_ciudad(ciudad)
            
            if ultima_fecha is None:
                print(f"No se pudo obtener la fecha más reciente para la ciudad {ciudad}. Agregando datos desde 01/01/2018.\n")
                datos: pd.DataFrame = self.obtener_datos_de_api(ciudad, None)
            else:
                fecha_actual: datetime = datetime.now()
                if (fecha_actual - ultima_fecha).days == 0:
                    print(f"La base de datos ya está actualizada para la ciudad {ciudad}. No es necesario actualizar los datos.\n")
                    return
                
                print(f"La última fecha registrada para {ciudad} es {ultima_fecha}. Actualizando datos desde {ultima_fecha}.\n")
                datos: pd.DataFrame = self.obtener_datos_de_api(ciudad, ultima_fecha)

            if not datos.empty:
                self.insertar_datos('clima', datos)
                print(f"Datos actualizados correctamente para la ciudad {ciudad}.\n")
            else:
                print("No se encontraron nuevos datos para la actualización.\n")
        
        except Exception as e:
            print(f"Error al actualizar los datos climáticos para la ciudad {ciudad}: {e}\n")

    def siguiente_prediccion(self, ciudad: str) -> Optional[pd.Series]:
        """
        Genera la siguiente predicción climática para una ciudad.

        :param ciudad: Nombre de la ciudad a predecir.
        :return: Serie con la predicción generada.
        """
        try:
            id_ciudad: Optional[int] = self.obtener_id_ciudad_por_nombre(ciudad)
            df: pd.DataFrame = self.obtener_dataframe("predic", id_ciudad)
            
            if df.empty:
                print(f"No se encontraron datos para la ciudad {ciudad} en la tabla 'predic'.\n")
                return None
            
            df['time'] = pd.to_datetime(df['time'], errors='coerce') 
            ultima_prediccion: pd.Series = df.loc[df['time'].idxmax()] 
            return ultima_prediccion 
        
        except Exception as e:
            print(f"Error al obtener la siguiente predicción para {ciudad}: {e}\n")
            return None