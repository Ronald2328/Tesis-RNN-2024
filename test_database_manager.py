from Get_data import DatabaseManager
import pandas as pd

# Creación de objeto de prueba
db_manager = DatabaseManager()

# Prueba de conexión a la base de datos
print("=== Prueba de Conexión a la Base de Datos ===")
connection = db_manager.connect_to_database()
if connection:
    print("Conexión exitosa")
else:
    print("Conexión fallida")
print()

# Prueba de obtención de DataFrame
print("=== Prueba de Obtener DataFrame ===")
ciudades_df = db_manager.get_dataframe('ciudades')
print("Muestra de la tabla 'ciudades':")
print(ciudades_df.head())
print()

# Prueba de inserción de datos
print("=== Prueba de Inserción de Datos ===")
data = pd.DataFrame({
    'id': [2],
    'ciudad': ['Lima'],
    'latitud': [-12.0432],
    'longitud': [-77.0282]
})
db_manager.insert_data('ciudades', data)
print()

# Prueba de obtener nombre de ciudad por ID
print("=== Prueba de Obtener Nombre de Ciudad por ID ===")
city_name = db_manager.get_city_name_by_id(1)
print(f"Nombre de la ciudad con ID 1: {city_name}")
print()

# Prueba de obtener coordenadas de ciudad
print("=== Prueba de Obtener Coordenadas de Ciudad ===")
lat, lon = db_manager.get_city_coordinates('Piura')
print(f"Coordenadas de Piura: Latitud: {lat}, Longitud: {lon}")
print()

# Prueba de actualización de datos climáticos
print("=== Prueba de Actualización de Datos Climáticos ===")
db_manager.update_climate_data('Piura')
print()

# Prueba de subida de datos climáticos
print("=== Prueba de Subida de Datos Climáticos ===")
db_manager.update_climate_data('Lima')
print()

# Prueba para obtener el id de una ciudad
print("=== Prueba de Obtención de ID ===")
id_city = db_manager.get_city_id_by_name('Lima')
print(f"ID de la ciudad con Lima: {id_city}")
print()

# Prueba para visualizar los datos nuevos
print("=== Prueba para ver los Datos Nuevos ===")
ciudades_df = db_manager.get_dataframe('clima',id_city)
print("Muestra de los datos de Lima:")
print(ciudades_df.head())
print()

# Prueba de obtener la siguiente predicción
print("=== Prueba de Obtener Siguiente Predicción ===")
next_pred = db_manager.next_prediction('Piura')
print(f"Siguiente predicción para Piura: {next_pred}")
print()
