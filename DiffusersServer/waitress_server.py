"""
Módulo para ejecutar servidores usando Waitress con monitoreo de memoria
waitress_server.py
"""

from waitress import serve
import logging
import gc
import psutil
import os
import threading
import time

# Configuración de logging
def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger('waitress')

logger = setup_logging()

def memory_cleanup(interval=30):
    """
    Función para monitorear y limpiar la memoria periódicamente
    
    Args:
        interval (int): Intervalo en segundos entre limpiezas
    """
    while True:
        try:
            # Forzar recolección de basura
            gc.collect()
            
            # Obtener información de memoria actual
            process = psutil.Process(os.getpid())
            mem = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memoria en uso: {mem:.2f} MB")
            
            time.sleep(interval)
        except Exception as e:
            logger.error(f"Error en limpieza de memoria: {str(e)}")
            time.sleep(interval)

def run_waitress_server(
    app,
    threads=5, 
    cleanup_interval=30, 
    channel_timeout=900, 
    ident='DiffusersServer',
    enable_memory_monitor=True
):
    """
    Ejecuta un servidor Flask utilizando Waitress con monitoreo de memoria opcional
    
    Args:
        app: Aplicación Flask
        threads (int): Número de hilos para Waitress
        cleanup_interval (int): Intervalo de limpieza para Waitress
        channel_timeout (int): Tiempo de espera máximo para canales
        ident (str): Identificador del servidor
        enable_memory_monitor (bool): Si se debe activar el monitoreo de memoria
        
    Returns:
        El resultado de serve() (aunque normalmente no retorna)
    """
    # Configurar el recolector de basura
    gc.enable()
    gc.set_threshold(700, 10, 5)
    
    # Iniciar monitoreo de memoria si está habilitado
    if enable_memory_monitor:
        cleanup_thread = threading.Thread(
            target=memory_cleanup, 
            args=(cleanup_interval,), 
            daemon=True
        )
        cleanup_thread.start()
        logger.info("Monitor de memoria activado")

    host = '127.0.0.1'
    port = 8500
    
    logger.info(f"Iniciando servidor Waitress en {host}:{port}...")
    return serve(
        app,
        host='127.0.0.1',
        port=8500,
        threads=threads,
        cleanup_interval=cleanup_interval,
        channel_timeout=channel_timeout,
        ident=ident
    )

