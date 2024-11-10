# Importar las bibliotecas necesarias
from Get_data import DatabaseManager  

def print_section_header(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}\n")

def test_database_manager():
    # Crear una instancia de la clase DatabaseManager
    db_manager = DatabaseManager(host='localhost', user='root', passwd='root', database='meteorology')

    # Conectar a la base de datos
    connection = db_manager.connect_to_database()
    if connection:
        print("\n")
        print("Conexión exitosa a la base de datos")
    else:
        print("Error al conectar a la base de datos")
    
    print("\n")

    # Probar obtener datos de la tabla 'temperatura' (Piura)
    city = 'Piura'
    df_temperatura = db_manager.get_dataframe('temperatura')
    if df_temperatura is not None:
        print(f"Datos obtenidos de la tabla 'temperatura' para {city}:")
        print(df_temperatura.head())  # Mostrar las primeras filas
    else:
        print(f"No se pudieron obtener los datos de la tabla 'temperatura' para {city}")

    print("\n")

    # Probar obtener las coordenadas de Piura
    coordinates = db_manager.get_coordinates(city)
    if coordinates:
        print(f"Coordenadas de {city}: {coordinates}")
    else:
        print(f"No se pudieron obtener las coordenadas de {city}")

    print("\n")

    # Probar obtener el ID de Piura
    city_id = db_manager.get_city_id(city)
    if city_id:
        print(f"ID de la ciudad {city}: {city_id}")
    else:
        print(f"No se pudo obtener el ID de la ciudad {city}")

    print("\n")

    # Probar obtener la fecha más reciente
    latest_date = db_manager.get_latest_date(city)
    if latest_date:
        print(f"Fecha más reciente en la base de datos: {latest_date}")
    else:
        print("No se pudo obtener la fecha más reciente de la base de datos")

    print("\n")

    # Probar combinar datos de dos tablas
    combined_df = db_manager.get_combined_dataframe(city)
    if combined_df is not None:
        print(f"Datos combinados de las tablas 'temperatura' y 'viento_nieve' para {city}:")
        print(combined_df.head())  # Mostrar las primeras filas combinadas
    else:
        print(f"No se pudieron combinar los datos de las tablas para {city}")

    print("\n")

    # Cerrar la conexión
    db_manager.close_connection(connection)
    print("Conexión cerrada")

    # Prueba con Lima
    print_section_header("Prueba con Lima")

    # Crear una instancia de la clase DatabaseManager nuevamente para Lima
    db_manager = DatabaseManager(host='localhost', user='root', passwd='root', database='meteorology')
    
    # Conectar a la base de datos
    connection = db_manager.connect_to_database()
    if connection:
        print("Conexión exitosa a la base de datos")
    else:
        print("Error al conectar a la base de datos")
    
    print("\n")

    # Probar obtener datos de la tabla 'temperatura' (Lima)
    city = 'Lima'
    df_temperatura = db_manager.get_dataframe('temperatura')
    if df_temperatura is not None:
        print(f"Datos obtenidos de la tabla 'temperatura' para {city}:")
        print(df_temperatura.head())  # Mostrar las primeras filas
    else:
        print(f"No se pudieron obtener los datos de la tabla 'temperatura' para {city}")

    print("\n")

    # Probar obtener las coordenadas de Lima
    coordinates = db_manager.get_coordinates(city)
    if coordinates:
        print(f"Coordenadas de {city}: {coordinates}")
    else:
        print(f"No se pudieron obtener las coordenadas de {city}")

    print("\n")

    # Probar obtener el ID de Lima
    city_id = db_manager.get_city_id(city)
    if city_id:
        print(f"ID de la ciudad {city}: {city_id}")
    else:
        print(f"No se pudo obtener el ID de la ciudad {city}")

    print("\n")

    # Probar obtener la fecha más reciente
    latest_date = db_manager.get_latest_date(city)
    if latest_date:
        print(f"Fecha más reciente en la base de datos: {latest_date}")
    else:
        print("No se pudo obtener la fecha más reciente de la base de datos")

    print("\n")

    # Probar combinar datos de dos tablas
    combined_df = db_manager.get_combined_dataframe(city)
    if combined_df is not None:
        print(f"Datos combinados de las tablas 'temperatura' y 'viento_nieve' para {city}:")
        print(combined_df.head())
    else:
        print(f"No se pudieron combinar los datos de las tablas para {city}")

    print("\n")

    # Cerrar la conexión
    db_manager.close_connection(connection)
    print("Conexión cerrada")

    # Prueba sin especificar ciudades, funciones específicas
    print_section_header("Prueba sin especificar ciudades, funciones específicas")

    # Crear una instancia de la clase DatabaseManager nuevamente
    db_manager = DatabaseManager(host='localhost', user='root', passwd='root', database='meteorology')
    
    # Conectar a la base de datos
    connection = db_manager.connect_to_database()
    if connection:
        print("Conexión exitosa a la base de datos")
    else:
        print("Error al conectar a la base de datos")
    
    print("\n")

    # Probar obtener la fecha más reciente sin especificar ciudad
    latest_date = db_manager.get_latest_date()
    if latest_date:
        print(f"Fecha más reciente en la base de datos: {latest_date}")
    else:
        print("No se pudo obtener la fecha más reciente de la base de datos")

    print("\n")

    # Prueba sin especificar ciudades, funciones específicas
    print_section_header("Prueba de actualización de datos")

    # Test de actualización de base de datos para una ciudad específica
    city_to_test = "Piura"  

    print(f"Probando la función update_database para la ciudad: {city_to_test}")
    db_manager.update_database(city_to_test) 

    # Test de carga o actualización de datos meteorológicos
    print(f"\nProbando la función load_or_update_weather_data para la ciudad: {city_to_test}")
    db_manager.load_or_update_weather_data(city_to_test)
    print("\n")

    # Test de actualización de base de datos para una ciudad diferente
    city_to_test = "Lima"  

    print(f"Probando la función update_database para la ciudad: {city_to_test}")
    db_manager.update_database(city_to_test) 

    # Test de carga o actualización de datos meteorológicos
    print(f"\nProbando la función load_or_update_weather_data para la ciudad: {city_to_test}")
    db_manager.load_or_update_weather_data(city_to_test)  

# Ejecutar las pruebas
if __name__ == "__main__":
    test_database_manager()