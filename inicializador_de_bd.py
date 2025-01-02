from sqlalchemy import create_engine, text

class InicializadorBaseDatos:
    """
    Clase para inicializar la base de datos y sus tablas.
    """
    def __init__(self, gestor_datos):
        """
        Inicializa el inicializador con una referencia al gestor de datos.
        
        :param gestor_datos: Instancia de GestorDatosClimaticos
        """
        self.gestor_datos = gestor_datos

    def crear_base_datos(self) -> None:
        """
        Crea la base de datos si no existe.
        """
        engine = create_engine(self.gestor_datos.url_servidor)
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {self.gestor_datos.base_datos}"))
            conn.commit()

    def crear_tablas(self) -> None:
        """
        Crea las tablas necesarias en la base de datos.
        """
        engine = create_engine(self.gestor_datos.url_base_datos)
        with engine.connect() as conn:
            # Crear tabla ciudades
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS ciudades (
                    id INT NOT NULL AUTO_INCREMENT,
                    ciudad VARCHAR(40) NOT NULL,
                    latitud FLOAT NOT NULL,
                    longitud FLOAT NOT NULL,
                    PRIMARY KEY (id),
                    UNIQUE KEY idx_ciudad (ciudad)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """))

            # Crear tabla clima
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS clima (
                    id_ciudad INT NOT NULL,
                    time DATETIME NOT NULL,
                    tmax FLOAT DEFAULT 0,
                    tmin FLOAT DEFAULT 0,
                    tavg FLOAT DEFAULT 0,
                    prcp FLOAT DEFAULT 0,
                    snow FLOAT DEFAULT 0,
                    wdir INT DEFAULT 0,
                    wspd FLOAT DEFAULT 0,
                    wpgt FLOAT DEFAULT 0,
                    pres FLOAT DEFAULT 0,
                    tsun FLOAT DEFAULT 0,
                    CONSTRAINT fk_clima_ciudades FOREIGN KEY (id_ciudad) REFERENCES ciudades (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """))

            # Crear tabla predic
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predic (
                    id INT NOT NULL,
                    time DATETIME NOT NULL,
                    tmax FLOAT UNSIGNED NOT NULL,
                    tmin FLOAT UNSIGNED NOT NULL,
                    tavg FLOAT UNSIGNED NOT NULL,
                    PRIMARY KEY (time),
                    CONSTRAINT fk_predic_ciudades FOREIGN KEY (id) REFERENCES ciudades (id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
            """))

            # Insertar datos iniciales de Piura
            conn.execute(text("""
                INSERT IGNORE INTO ciudades (id, ciudad, latitud, longitud)
                VALUES (1, 'Piura', -5.1833, -80.6)
            """))
            
            conn.commit()