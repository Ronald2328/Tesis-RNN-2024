# Sistema de Gestión de Datos Meteorológicos

Este proyecto forma parte de una tesis que implementa un sistema para gestionar y analizar datos meteorológicos utilizando Python.

## Descripción

El sistema permite:
- Conectar con una base de datos MySQL para almacenar datos meteorológicos
- Obtener datos climáticos históricos usando la API de Meteostat
- Gestionar información de ciudades y sus coordenadas geográficas
- Actualizar automáticamente los registros climáticos
- Manejar predicciones meteorológicas

## Requisitos

- Python 3.x
- MySQL
- Entorno virtual de Python (virtualenv)
- Archivo requirements.txt con las dependencias:
  - pandas
  - sqlalchemy
  - meteostat
  - pymysql

## Instalación

1. Crear y activar el entorno virtual:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Asegúrese de tener una base de datos MySQL llamada 'meteorology'
2. Configure los parámetros de conexión:
```python
db_manager = DatabaseManager(
    host='localhost',
    user='root',
    passwd='root',
    database='meteorology'
)
```

## Uso Principal

```python
# Inicializar el gestor de base de datos
db_manager = DatabaseManager()

# Obtener datos de una ciudad
ciudad = "Piura"
db_manager.update_climate_data(ciudad)

# Obtener última predicción
prediccion = db_manager.next_prediction(ciudad)
```

## Funcionalidades Principales

- `update_climate_data(ciudad)`: Actualiza los datos climáticos de una ciudad específica
- `get_city_coordinates(ciudad)`: Obtiene las coordenadas geográficas de una ciudad
- `get_latest_date_for_city(ciudad)`: Obtiene la última fecha registrada para una ciudad
- `next_prediction(ciudad)`: Obtiene la siguiente predicción climática

## Manejo de Errores

El sistema incluye manejo de errores para:
- Conexiones fallidas a la base de datos
- Ciudades no encontradas
- Datos duplicados
- Errores en la API de meteorología

## Autor

Ronaldo Ofael Olivares Palacios

## Licencia

MIT License
