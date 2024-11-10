import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
from meteostat import Daily, Point

class DatabaseManager:

    '''La clase DatabaseManager gestiona la conexión y operaciones con una base de datos meteorológica.
    
    Facilita la extracción, actualización y combinación de datos meteorológicos almacenados en varias tablas.
    Incluye métodos para obtener datos específicos, como coordenadas e ID de ciudades, y para actualizar y
    combinar datos históricos y de predicción. Proporciona manejo de conexiones y errores para cada operación,
    permitiendo la interacción estructurada y segura con la base de datos meteorológica.
    '''

    def __init__(self, host='localhost', user='root', passwd='root', database='meteorology'):

        ''' 
        Inicializa la clase DatabaseManager con los parámetros de conexión a la base de datos, 
        configurando la URL de la base de datos con el host, usuario, contraseña y nombre de la base de datos.
        '''

        self.database_url = f"mysql+pymysql://{user}:{passwd}@{host}/{database}"

    def connect_to_database(self):

        '''
        Establece una conexión con la base de datos usando la URL configurada en la inicialización.

        Returns:
            Connection: Objeto de conexión activo si la conexión es exitosa.
        '''

        try:
            return create_engine(self.database_url).connect()
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def close_connection(self, connection):

        '''
        Cierra la conexión a la base de datos si la conexión existe.

        Args:
            connection: Objeto de conexión a cerrar.
        '''

        if connection:
            connection.close()
    
    def get_dataframe(self, table):

        '''
        Obtiene todos los registros de una tabla especificada y los devuelve en un DataFrame de pandas.

        Args:
            table (str): Nombre de la tabla desde la cual se extraen los datos.

        Returns:
            DataFrame: DataFrame con los datos de la tabla o None si ocurre un error.
        '''

        connection = self.connect_to_database()
        try:
            query = f"SELECT * FROM {table}"
            return pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error loading data from {table}: {e}")
        finally:
            self.close_connection(connection)

    def upload_data_to_database(self,df,table):

        '''
        Sube datos de un DataFrame a una tabla de la base de datos, omitiendo valores nulos.

        Args:
            df (DataFrame): DataFrame que contiene los datos a subir.
            table (str): Nombre de la tabla en la base de datos donde se subirán los datos.
        '''

        connection = self.connect_to_database()
        try:
            df = df.where(pd.notnull(df), None)
            df.to_sql(table, con=connection, if_exists='append', index=False, method='multi')
        except Exception as e:
            print(f"Error uploading data: {e}")
        finally:
            self.close_connection(connection)
    
    def get_coordinates(self, city):
        '''
        Recupera las coordenadas de latitud y longitud para una ciudad específica.

        Args:
            city (str): Nombre de la ciudad.

        Returns:
            tuple: Coordenadas (latitud, longitud) o None en caso de error.
        '''
        connection = self.connect_to_database()
        try:
            coordinates = self.fetch_coordinates_from_db(city, connection)
            if coordinates:
                return coordinates
            else:
                print(f"No se encontraron coordenadas para {city}. Solicitando datos...")

                if self.handle_missing_city(city, connection):
                    coordinates = self.fetch_coordinates_from_db(city, connection)

                    if coordinates:
                        return coordinates
                    else:
                        print(f"Error al verificar las coordenadas de {city} después de guardarlas.")
                        return None
                else:
                    return None
        finally:
            self.close_connection(connection)

    def fetch_coordinates_from_db(self, city, connection):
        '''
        Busca las coordenadas de una ciudad en la base de datos.

        Args:
            city (str): Nombre de la ciudad.
            connection: Conexión a la base de datos.

        Returns:
            tuple: Coordenadas (latitud, longitud) o None si no se encuentra la ciudad.
        '''
        try:
            latitud = pd.read_sql(f"SELECT latitud FROM coordenadas WHERE ciudad = '{city}'", connection).iloc[0, 0]
            longitud = pd.read_sql(f"SELECT longitud FROM coordenadas WHERE ciudad = '{city}'", connection).iloc[0, 0]
            return latitud, longitud
        except IndexError:
            return None

    def handle_missing_city(self, city, connection):
        '''
        Solicita al usuario los datos de una nueva ciudad y la verifica en la API.
        Guarda la ciudad en la base de datos si es válida.

        Args:
            city (str): Nombre de la ciudad.
            connection: Conexión a la base de datos.
        '''
        # Solicitar las coordenadas de la ciudad al usuario
        new_lat, new_lon = self.request_city_coordinates(city)
        if self.verify_city_in_api(new_lat, new_lon):
            self.save_city_to_db(city, new_lat, new_lon, connection)
            return True
        else:
            print(f"Las coordenadas proporcionadas para {city} no son válidas o no se pudieron verificar.")
            return False

    def request_city_coordinates(self, city):
        '''
        Solicita al usuario ingresar las coordenadas de latitud y longitud para una ciudad existente.

        Args:
            city (str): Nombre de la ciudad.

        Returns:
            tuple: (latitud, longitud)
        '''
        # Personalizar los mensajes para que incluyan el nombre de la ciudad
        print(f"\n\nIngrese las coordenadas para la ciudad: {city}")
        while True:
            try:
                # Preguntar por la latitud con el nombre de la ciudad
                new_lat = float(input(f"Ingrese la latitud de {city}: "))
                break
            except ValueError:
                print("Por favor, ingrese un valor numérico válido para la latitud.")
        
        while True:
            try:
                # Preguntar por la longitud con el nombre de la ciudad
                new_lon = float(input(f"Ingrese la longitud de {city}: "))
                break
            except ValueError:
                print("Por favor, ingrese un valor numérico válido para la longitud.")

        return new_lat, new_lon

    def verify_city_in_api(self, lat, lon):
        '''
        Verifica si una ciudad con las coordenadas dadas existe en la API de meteostat.

        Args:
            lat (float): Latitud.
            lon (float): Longitud.

        Returns:
            bool: True si la estación es válida, False de lo contrario.
        '''
        try:
            point = Point(lat, lon)
            data = Daily(point, datetime(2023, 1, 1), datetime(2023, 1, 2)).fetch()
            if not data.empty:
                print("La estación es válida y se encontró en la API.")
                return True
            else:
                print("La estación no es válida o no se encontraron datos en la API.")
                return False
        except Exception as e:
            print(f"Error verificando la estación en la API: {e}")
            return False
    
    def save_city_to_db(self, city, lat, lon, connection):
        '''
        Guarda una nueva ciudad y sus coordenadas en la base de datos.

        Args:
            city (str): Nombre de la ciudad.
            lat (float): Latitud.
            lon (float): Longitud.
        '''
        try:
            # Obtener el nuevo id para la ciudad
            new_id = pd.read_sql("SELECT MAX(id) FROM ciudad", connection).iloc[0, 0] + 1

            # Imprimir valores para depuración
            print(f"\nInsertando ciudad: {city}, ID: {new_id}")
            print(f"Insertando coordenadas: {city}, Latitud: {lat}, Longitud: {lon}\n")
            
            # Preparar las consultas con `text()`
            query_city = text("INSERT INTO ciudad (id, ciudad) VALUES (:id, :ciudad)")
            query_coords = text("INSERT INTO coordenadas (ciudad, latitud, longitud) VALUES (:ciudad, :lat, :lon)")
            
            # Ejecutar las consultas con parámetros
            connection.execute(query_city, {"id": new_id, "ciudad": city})
            connection.execute(query_coords, {"ciudad": city, "lat": lat, "lon": lon})
            
            # Commit de las transacciones
            connection.commit()
            
            print(f"Datos de {city} guardados en la base de datos.")
            
        except Exception as e:
            # Rollback en caso de error
            connection.rollback()
            print(f"Error al guardar la ciudad en la base de datos: {e}")
    
    def get_city_id(self, city):

        '''
        Obtiene el ID de una ciudad específica desde la base de datos.

        Args:
            city (str): Nombre de la ciudad.

        Returns:
            int or None: ID de la ciudad si existe, o None si ocurre un error.
        '''

        connection = self.connect_to_database()
        try:
            return pd.read_sql(f"SELECT id FROM ciudad WHERE ciudad = '{city}'", connection).iloc[0].iloc[0]
        except Exception as e:
            print(f"Error fetching city ID for {city}: {e}")
        finally:
            self.close_connection(connection)

    def get_latest_date(self, city=None):
        '''
        Recupera la fecha más reciente en la que se registraron datos en las tablas de temperatura y viento_nieve.
        Si se especifica una ciudad, se recupera la fecha más reciente de esa ciudad. Si no se especifica, se
        recupera la fecha más reciente de toda la base de datos e imprime la ciudad de origen.

        Args:
            city (str, optional): Nombre de la ciudad.

        Returns:
            datetime or None: La fecha más reciente o None si las fechas no coinciden o hay un error.
        '''
        connection = self.connect_to_database()
        try:
            if city:
                # Obtener el id_ciudad correspondiente a la ciudad
                city_id_query = f"SELECT id FROM ciudad WHERE ciudad = '{city}'"
                city_id = pd.read_sql(city_id_query, connection).iloc[0, 0]

                # Obtener la fecha más reciente para la ciudad específica
                fecha1 = pd.read_sql(f"SELECT MAX(dia) FROM temperatura WHERE id_ciudad = {city_id}", connection).iloc[0, 0]
                fecha2 = pd.read_sql(f"SELECT MAX(dia) FROM viento_nieve WHERE id_ciudad = {city_id}", connection).iloc[0, 0]
            else:
                # Obtener la fecha más reciente de todas las ciudades
                fecha1_df = pd.read_sql("SELECT id_ciudad, MAX(dia) as max_fecha FROM temperatura GROUP BY id_ciudad ORDER BY max_fecha DESC LIMIT 1", connection)
                fecha2_df = pd.read_sql("SELECT id_ciudad, MAX(dia) as max_fecha FROM viento_nieve GROUP BY id_ciudad ORDER BY max_fecha DESC LIMIT 1", connection)
                
                fecha1 = fecha1_df.iloc[0]['max_fecha']
                fecha2 = fecha2_df.iloc[0]['max_fecha']
                
                if fecha1 and fecha1 == fecha2:
                    # Obtener el nombre de la ciudad de la que proviene la fecha más reciente
                    city_id = fecha1_df.iloc[0]['id_ciudad']
                    city_name_query = f"SELECT ciudad FROM ciudad WHERE id = {city_id}"
                    city_name = pd.read_sql(city_name_query, connection).iloc[0, 0]
                    print(f"La fecha más reciente proviene de la ciudad: {city_name}")
                
            return fecha1 if fecha1 == fecha2 else None

        except Exception as e:
            print(f"Error fetching latest date: {e}")
            return None
        finally:
            self.close_connection(connection)
    
    def get_combined_dataframe(self, city, table1='temperatura', table2='viento_nieve'):
        '''
        Combina datos de dos tablas especificadas en un DataFrame único filtrado por una ciudad específica y lo retorna.

        Args:
            city (str): Nombre de la ciudad.
            table1 (str): Nombre de la primera tabla (por defecto, 'temperatura').
            table2 (str): Nombre de la segunda tabla (por defecto, 'viento_nieve').

        Returns:
            DataFrame: DataFrame combinado de las dos tablas filtrado por la ciudad especificada.
        '''
        connection = self.connect_to_database()
        try:
            city_id_query = f"SELECT id FROM ciudad WHERE ciudad = '{city}'"
            city_id = pd.read_sql(city_id_query, connection).iloc[0, 0]

            if city_id:
                df_temperatura = pd.read_sql(f'SELECT * FROM {table1} WHERE id_ciudad = {city_id}', connection)
                df_viento_nieve = pd.read_sql(f'SELECT * FROM {table2} WHERE id_ciudad = {city_id}', connection)
                
                df_combined = pd.merge(df_temperatura, df_viento_nieve, on=['id_ciudad', 'dia'])
                return df_combined
            else:
                print(f"No se encontró la ciudad {city} en la base de datos.")
                return None

        except Exception as e:
            print(f"Error combining data: {e}")
            return None
        finally:
            self.close_connection(connection)

    def update_database(self, city):
        '''
        Actualiza la base de datos con los datos meteorológicos de una ciudad específica si hay nuevos registros.

        Args:
            city (str): Nombre de la ciudad para la cual se actualizarán los datos.
        '''
        try:
            # Obtener la fecha más reciente en la base de datos
            latest_date = self.get_latest_date(city)
            current_date = datetime.now() - pd.DateOffset(days=1)

            # Asegúrate de que latest_date no sea None
            if latest_date is None:
                print(f"No se pudo obtener la fecha más reciente para la ciudad {city}.")
                return
            
            # Comprobar si hay datos nuevos
            if (current_date - latest_date).days > 0:
                latitud, longitud = self.get_coordinates(city)
                start = latest_date + pd.DateOffset(days=1)
                data = Daily(Point(latitud, longitud), start, current_date).fetch()

                if not data.empty:
                    df_temperature, df_wind_snow = self.prepare_dataframes(data, city)
                    self.upload_data_to_database(df_temperature, 'temperatura')
                    self.upload_data_to_database(df_wind_snow, 'viento_nieve')
                    print(f"Data for {city} from {start} to {current_date} has been added.")
                else:
                    print("No new data available.")
            else:
                print("Database is already up to date.")
        except Exception as e:
            print(f"Error updating database for {city}: {e}")

    def load_or_update_weather_data(self, city):
        '''
        Carga o actualiza los datos meteorológicos de una ciudad.
        Si la ciudad es nueva, se cargan todos sus datos. Si ya está en la base de datos, se actualizan los datos faltantes.

        Args:
            city (str): Nombre de la ciudad.
        '''
        connection = None
        try:
            connection = self.connect_to_database()
            
            # Verifica si la ciudad ya tiene datos en la tabla 'temperatura'
            city_exists = self.city_exists_in_db(city, connection)
            
            if not city_exists:
                print(f"La ciudad {city} es nueva. Cargando datos completos...")
                self.load_full_weather_data(city)
            else:
                print(f"La ciudad {city} ya está en la base de datos. Actualizando datos...")
                self.update_database(city)  
        except Exception as e:
            print(f"Error loading or updating weather data for {city}: {e}")
        finally:
            self.close_connection(connection)
    
    def city_exists_in_db(self, city, connection):
        '''
        Verifica si una ciudad ya tiene datos en la tabla 'temperatura' usando SQLAlchemy.
        Si tiene registros en la tabla 'temperatura', se considera que la ciudad existe.

        Args:
            city (str): Nombre de la ciudad.
            connection: Conexión a la base de datos.

        Returns:
            bool: True si la ciudad tiene datos en la tabla 'temperatura', False si no.
        '''
        try: 
            # Consulta SQL para verificar si la ciudad tiene registros en la tabla 'temperatura'
            query = text("SELECT COUNT(*) FROM temperatura WHERE id_ciudad IN (SELECT id FROM ciudad WHERE ciudad = :city)")
            
            # Ejecutar la consulta con el parámetro 'city'
            result = connection.execute(query, {"city": city}).fetchone()

            # Si el resultado es mayor que 0, la ciudad tiene datos
            if result[0] > 0:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error al verificar si la ciudad tiene datos: {e}")
            return False
    
    def load_full_weather_data(self, city):
        '''
        Carga los datos meteorológicos de una nueva ciudad desde la API de Meteostat y los inserta en la base de datos,
        omitiendo las fechas con valores nulos en ciertas columnas hasta que no haya más valores nulos.

        Args:
            city (str): Nombre de la ciudad.
        '''
        connection = None
        try:
            connection = self.connect_to_database()
            lat, lon = self.get_coordinates(city)
            if lat is None or lon is None:
                print("Error: No se encontraron las coordenadas de la ciudad.")
                return

            train_start_date = datetime.strptime("01/01/2018", "%d/%m/%Y")
            end_date = datetime.now()
            point = Point(lat, lon)

            data = Daily(point, train_start_date, end_date).fetch()

            # Verificar y avanzar hasta la fecha en la que ya no haya valores nulos en las columnas de interés
            if data[['tmin', 'tmax', 'tavg', 'pres']].isnull().any().any():
                first_valid_index = data[['tmin', 'tmax', 'tavg', 'pres']].dropna().index[0]
                data = data.loc[first_valid_index:]
                print(f"Datos filtrados desde la fecha {first_valid_index} en adelante para evitar valores nulos.")

            # Preparar DataFrames
            df_temperature, df_wind_snow = self.prepare_dataframes(data, city)

            # Subir los datos a la base de datos
            self.upload_data_to_database(df_temperature, 'temperatura')
            self.upload_data_to_database(df_wind_snow, 'viento_nieve')

            print(f"Datos meteorológicos de {city} cargados exitosamente.")
        except Exception as e:
            print(f"Error al cargar datos meteorológicos: {e}")
        finally:
            self.close_connection(connection)
    
    def prepare_dataframes(self, data, city):
        '''
        Prepara dos DataFrames con los datos de temperatura y viento/nieve para su subida a la base de datos.

        Args:
            data (DataFrame): DataFrame con los datos meteorológicos recientes.
            city (str): Nombre de la ciudad.

        Returns:
            tuple: DataFrames listos para subir (temperatura, viento_nieve).
        '''
        df = pd.DataFrame(data)
        df['dia'], df['id_ciudad'] = df.index, self.get_city_id(city)
        
        df_temperature = df[['id_ciudad', 'dia', 'tmax', 'tmin', 'tavg', 'prcp']].rename(columns={
            'tmax': 'temp_max', 'tmin': 'temp_min', 'tavg': 'avg_temp', 'prcp': 'precp'
        })
        
        df_wind_snow = df[['id_ciudad', 'dia', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']].rename(columns={
            'snow': 'nieve', 'wdir': 'direc_viento', 'wspd': 'vel_viento',
            'wpgt': 'rafaga', 'pres': 'presion', 'tsun': 'tiempo_sol'
        })

        return df_temperature, df_wind_snow

    def get_latest_date_prediction(self):
        '''
        Obtiene la fecha más reciente en la que se registraron predicciones en la base de datos.

        Returns:
            datetime: Fecha de la predicción más reciente o None si ocurre un error.
        '''
        connection = None
        try:
            connection = self.connect_to_database()
            query = "SELECT MAX(dia) FROM prediccion"
            return pd.read_sql(query, connection).iloc[0].iloc[0]
        except Exception as e:
            print(f"Error retrieving latest prediction date: {e}")
            return None
        finally:
            self.close_connection(connection)
    
    def next_prediction(self):
        '''
        Recupera la predicción más reciente registrada en la base de datos.

        Returns:
            DataFrame: DataFrame con los datos de la predicción más reciente.
        '''
        connection = None
        try:
            connection = self.connect_to_database()
            query = "SELECT * FROM prediccion WHERE dia = (SELECT MAX(dia) FROM prediccion)"
            df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            print(f"Error retrieving next prediction: {e}")
            return pd.DataFrame() 
        finally:
            self.close_connection(connection)