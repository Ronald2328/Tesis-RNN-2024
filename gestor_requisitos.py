import logging
import os
import subprocess
import sys
from typing import List, Dict, Optional

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorDependencias(Exception):
    """Clase personalizada para errores relacionados con dependencias."""
    pass

def verificar_pip() -> None:
    """
    Verifica si pip está instalado y accesible en el sistema.  
    """
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.SubprocessError:
        raise ErrorDependencias("Pip no está instalado o no es accesible en el sistema.")

def leer_archivo_requisitos(ruta_archivo: str) -> List[str]:
    """
    Lee y procesa el archivo de requisitos.
    
    :param ruta_archivo: Ruta al archivo de requisitos.
    :return: Lista de dependencias.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
    
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            return [
                linea.strip() 
                for linea in archivo 
                if linea.strip() and not linea.startswith("#")
            ]
    except Exception as error:
        raise ErrorDependencias(f"Error al leer el archivo {ruta_archivo}: {error}")

def obtener_paquetes_instalados() -> Dict[str, str]:
    """
    Obtiene un diccionario de paquetes instalados y sus versiones.
    
    :return: Diccionario de paquetes instalados.
    """
    try:
        salida = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            universal_newlines=True
        )
        paquetes_instalados = {}
        for paquete in salida.split('\n'):
            if "==" in paquete:
                nombre, version = paquete.split("==")
                paquetes_instalados[nombre.lower().strip()] = version.strip()
        return paquetes_instalados
    except subprocess.SubprocessError as error:
        raise ErrorDependencias(f"Error al obtener paquetes instalados: {error}")

def encontrar_paquetes_faltantes(
    dependencias: List[str], 
    paquetes_instalados: Dict[str, str]) -> List[str]:
    """
    Identifica los paquetes que necesitan ser instalados o actualizados.
    
    :param dependencias: Lista de dependencias requeridas.
    :param paquetes_instalados: Diccionario de paquetes instalados.
    :return: Lista de paquetes faltantes.
    """
    paquetes_faltantes = []
    
    for dependencia in dependencias:
        if "==" in dependencia:
            nombre, version = dependencia.split("==")
            nombre = nombre.lower().strip()
            version = version.strip()
            if (nombre not in paquetes_instalados or 
                paquetes_instalados[nombre] != version):
                paquetes_faltantes.append(dependencia)
        else:
            nombre = dependencia.lower().strip()
            if nombre not in paquetes_instalados:
                paquetes_faltantes.append(dependencia)
    
    return paquetes_faltantes

def instalar_paquetes(paquetes: List[str]) -> None:
    """
    Instala los paquetes especificados usando pip.
    
    :param paquetes: Lista de paquetes a instalar.
    """
    if not paquetes:
        return
    
    try:
        logger.info(f"Instalando paquetes: {', '.join(paquetes)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + paquetes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Instalación completada exitosamente")
    except subprocess.SubprocessError as error:
        raise ErrorDependencias(f"Error durante la instalación de paquetes: {error}")

def verificar_e_instalar_requisitos(archivo_requisitos: str = "requirements.txt") -> None:
    """
    Función principal que verifica e instala las dependencias necesarias.
    
    :param archivo_requisitos: Ruta al archivo de requisitos.
    """
    try:
        verificar_pip()
        dependencias = leer_archivo_requisitos(archivo_requisitos)
        paquetes_instalados = obtener_paquetes_instalados()
        paquetes_faltantes = encontrar_paquetes_faltantes(dependencias, paquetes_instalados)
        
        if paquetes_faltantes:
            instalar_paquetes(paquetes_faltantes)
        else:
            logger.info("Todas las dependencias están instaladas correctamente")
            
    except (FileNotFoundError, ErrorDependencias) as error:
        logger.error(str(error))
        sys.exit(1)
    except Exception as error:
        logger.error(f"Error inesperado: {error}")
        sys.exit(1)

if __name__ == "__main__":
    verificar_e_instalar_requisitos()