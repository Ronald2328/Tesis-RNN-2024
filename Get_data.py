import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from meteostat import Daily, Point
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DatabaseManager:
    def __init__(self, host: str = 'localhost', user: str = 'root', passwd: str = 'root', database: str = 'meteorology'):
        self.database_url: str = f"mysql+pymysql://{user}:{passwd}@{host}/{database}"

    def connect_to_database(self) -> Optional[object]:
        try:
            return create_engine(self.database_url).connect()
        except Exception as e:
            print(f"Error al conectar a la base de datos: {e}\n")
            return None

    def close_connection(self, connection: Optional[object]) -> None:
        if connection:
            connection.close()

    def get_dataframe(self, table: str, city_filter: Optional[int] = None) -> pd.DataFrame:
        connection = self.connect_to_database()
        try:
            query: str = f"SELECT * FROM {table}"
            df: pd.DataFrame = pd.read_sql(query, connection)
            if city_filter is not None and 'id_ciudad' in df.columns:
                df = df[df['id_ciudad'] == city_filter]
            return df
        except Exception as e:
            print(f"Error al cargar datos de la tabla {table}: {e}\n")
            return pd.DataFrame()
        finally:
            self.close_connection(connection)

    def insert_data(self, table: str, data: pd.DataFrame, ignore_column: Optional[str] = None) -> None:
        connection = self.connect_to_database()
        try:
            existing_data: pd.DataFrame = self.get_dataframe(table)
            if ignore_column and ignore_column in data.columns:
                data_without_ignore: pd.DataFrame = data.drop(columns=[ignore_column])
            else:
                data_without_ignore: pd.DataFrame = data
            
            existing_data_filtered: pd.DataFrame = existing_data[data_without_ignore.columns]
            new_data_filtered: pd.DataFrame = data_without_ignore
            duplicates: pd.DataFrame = new_data_filtered.merge(existing_data_filtered, how='inner', on=data_without_ignore.columns.tolist())
            
            if not duplicates.empty:
                print(f"Los siguientes datos ya existen en la tabla {table}:")
                print(f'{duplicates}\n')
                return 
            
            data.to_sql(table, con=connection, if_exists='append', index=False)
            print(f"Datos insertados correctamente en la tabla {table}.\n")
        except Exception as e:
            print(f"Error al insertar datos en la tabla {table}: {e}\n")
        finally:
            self.close_connection(connection)

    def get_city_name_by_id(self, city_id: int) -> Optional[str]:
        df: pd.DataFrame = self.get_dataframe('ciudades', city_id)
        if not df.empty:
            return df['ciudad'].iloc[0]
        return None

    def get_city_id_by_name(self, ciudad: str) -> Optional[int]:
        df: pd.DataFrame = self.get_dataframe('ciudades')
        city_row: pd.DataFrame = df[df['ciudad'] == ciudad]
        if not city_row.empty:
            return city_row['id'].iloc[0]
        return None

    def get_city_coordinates(self, ciudad: str) -> Tuple[Optional[float], Optional[float]]:
        city_id: Optional[int] = self.get_city_id_by_name(ciudad)
        if city_id is None:
            print(f"No se encontró la ciudad {ciudad}. Solicitando coordenadas...\n")
            if self.handle_missing_city(ciudad):
                return self.get_city_coordinates(ciudad)
            else:
                return None, None

        df: pd.DataFrame = self.get_dataframe('ciudades')
        city_row: pd.DataFrame = df[df['id'] == city_id]
        return city_row['latitud'].iloc[0], city_row['longitud'].iloc[0]

    def handle_missing_city(self, ciudad: str) -> bool:
        lat, lon = self.request_city_coordinates(ciudad)
        if self.verify_city_in_api(lat, lon):
            self.add_city_to_db(ciudad, lat, lon)
            return True
        else:
            print(f"Las coordenadas proporcionadas para {ciudad} no son válidas. Intente de nuevo.\n")
            return False
    
    def add_city_to_db(self, ciudad: str, lat: float, lon: float) -> None:
        df: pd.DataFrame = self.get_dataframe('ciudades')
        max_id: int = df['id'].max() if not df.empty else 0 
        new_id: int = max_id + 1

        new_city_data: pd.DataFrame = pd.DataFrame({
            'id': [new_id],
            'ciudad': [ciudad],
            'latitud': [lat],
            'longitud': [lon]
        })

        self.insert_data('ciudades', new_city_data)
        print(f"Ciudad {ciudad} agregada con ID {new_id} a la base de datos.\n")    

    def request_city_coordinates(self, ciudad: str) -> Tuple[float, float]:
        while True:
            try:
                lat: float = float(input(f"Ingrese la latitud de {ciudad}: "))
                lon: float = float(input(f"Ingrese la longitud de {ciudad}: "))
                return lat, lon
            except ValueError:
                print("Por favor, ingrese un valor numérico válido para la latitud y longitud.\n")

    def verify_city_in_api(self, lat: float, lon: float) -> bool:
        try:
            point: Point = Point(lat, lon)
            data: pd.DataFrame = Daily(point, datetime(2023, 1, 1), datetime(2023, 1, 2)).fetch()
            if not data.empty:
                print("La estación es válida y se encontró en la API.\n")
                return True
            print("La estación no es válida o no se encontraron datos en la API.\n")
            return False
        except Exception as e:
            print(f"Error verificando la estación en la API: {e}\n")
            return False

    def get_latest_date_for_city(self, ciudad: str) -> Optional[datetime]:
        city_id: Optional[int] = self.get_city_id_by_name(ciudad)
        if city_id is None:
            print(f"No se encontró el ID para la ciudad {ciudad}.\n")
            return None
        
        df: pd.DataFrame = self.get_dataframe('clima', city_id)
        if not df.empty:
            latest_date: datetime = df['time'].max()
            print(f"La última fecha registrada en la tabla 'clima' para {ciudad} es: {latest_date}\n")
            return latest_date
        else:
            print(f"No se encontraron datos para la ciudad {ciudad}.\n")
            return None

    def get_latest_prediction_date_for_city(self, ciudad: str, bool: bool = True) -> datetime:
        city_id: Optional[int] = self.get_city_id_by_name(ciudad)
        if city_id is None:
            print(f"No se encontró el ID para la ciudad {ciudad}.\n")
            return datetime.now()
        
        df: pd.DataFrame = self.get_dataframe('predic', city_id)
        if not df.empty:
            latest_date: datetime = df['time'].max()
            if bool:
                print(f"La última fecha de predicción en la tabla 'predic' para {ciudad} es: {latest_date}\n")
            return latest_date
        else:
            print(f"No se encontraron datos de predicción para la ciudad {ciudad}. Se devolverá la fecha actual.\n")
            return datetime.now()

    def get_data_from_api(self, ciudad: str, latest_date: Optional[datetime]) -> pd.DataFrame:
        lat, lon = self.get_city_coordinates(ciudad)
        start_date: datetime = latest_date + pd.DateOffset(days=1) if latest_date else datetime(2018, 1, 1)
        try:
            data: pd.DataFrame = Daily(Point(lat, lon), start_date, datetime.now()).fetch()  
            if not data.empty:
                data = data.reset_index()
                city_id: Optional[int] = self.get_city_id_by_name(ciudad)
                data['id_ciudad'] = city_id
                return data
            else:
                print(f"No se encontraron datos para la ciudad {ciudad} en la API.\n")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error al obtener datos de la API para {ciudad}: {e}\n")
            return pd.DataFrame()

    def update_climate_data(self, ciudad: str) -> None:
        try:
            latest_date: Optional[datetime] = self.get_latest_date_for_city(ciudad)
            
            if latest_date is None:
                print(f"No se pudo obtener la fecha más reciente para la ciudad {ciudad}. Agregando datos desde 01/01/2018.\n")
                data: pd.DataFrame = self.get_data_from_api(ciudad, None)  # Obtener desde 01/01/2018
            else:
                current_date: datetime = datetime.now()
                # Verifica si la fecha más reciente está actualizada
                if (current_date - latest_date).days == 0:
                    print(f"La base de datos ya está actualizada para la ciudad {ciudad}. No es necesario actualizar los datos.\n")
                    return
                
                print(f"La última fecha registrada para {ciudad} es {latest_date}. Actualizando datos desde {latest_date}.\n")
                data: pd.DataFrame = self.get_data_from_api(ciudad, latest_date)  # Obtener desde la fecha más reciente

            # Verifica si los datos obtenidos no están vacíos
            if not data.empty:
                self.insert_data('clima', data)
                print(f"Datos actualizados correctamente para la ciudad {ciudad}.\n")
            else:
                print("No se encontraron nuevos datos para la actualización.\n")
        
        except Exception as e:
            print(f"Error al actualizar los datos climáticos para la ciudad {ciudad}: {e}\n")

    def next_prediction(self, ciudad: str) -> Optional[pd.Series]:
        try:
            id_ciudad: Optional[int] = self.get_city_id_by_name(ciudad)
            df: pd.DataFrame = self.get_dataframe("predic", id_ciudad)
            
            if df.empty:
                print(f"No se encontraron datos para la ciudad {ciudad} en la tabla 'predic'.\n")
                return None
            
            # Asegura que la columna 'time' se convierte correctamente a datetime
            df['time'] = pd.to_datetime(df['time'], errors='coerce') 
            
            # Encuentra la última predicción
            latest_prediction: pd.Series = df.loc[df['time'].idxmax()] 
            return latest_prediction 
        
        except Exception as e:
            print(f"Error al obtener la siguiente predicción para {ciudad}: {e}\n")
            return None