import json

class Configuracion:
    """
    Maneja las configuraciones del sistema.
    """
    def __init__(self, archivo_configuracion='configuracion.json'):
        self.archivo_configuracion = archivo_configuracion
        self.configuracion = self.cargar_configuracion()

    def cargar_configuracion(self):
        """
        Carga las configuraciones desde el archivo JSON.
        """
        try:
            with open(self.archivo_configuracion, 'r') as archivo:
                return json.load(archivo)
        except FileNotFoundError:
            print(f"Archivo {self.archivo_configuracion} no encontrado.")
            return {}
        except json.JSONDecodeError:
            print(f"El archivo {self.archivo_configuracion} contiene errores de formato.")
            return {}

    def obtener_usuario(self):
        """
        Obtiene el usuario de la configuración.
        """
        return self.configuracion.get('usuario', None)

    def obtener_contrasena(self):
        """
        Obtiene la contraseña de la configuración.
        """
        return self.configuracion.get('contrasena', None)

    def obtener_base_datos(self):
        """
        Obtiene el nombre de la base de datos de la configuración.
        """
        return self.configuracion.get('base_datos', None)

    def obtener_servidor(self):
        """
        Obtiene el servidor de la configuración.
        """
        return self.configuracion.get('servidor', None)
    
    def obtener_ciudad(self):
        """
        Obtiene la ciudad de la configuración.
        """
        return self.configuracion.get('ciudad', None)
