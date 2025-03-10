# create_server.py

from .Pipelines import *
from .server import *
from .waitress_server import *

# Función principal para iniciar un servidor completo
def create_inference_server(
    model:str,
    type_model: str = 't2im',
    threads=5,
    enable_memory_monitor=True,
    custom_model: bool = False,
    custom_pipeline: Optional[type] | None = None,
    constructor_pipeline: Optional[type] | None = None,
    components: Optional[Dict[str, Any]] = None
):
    """
    Crea y ejecuta un servidor de inferencia de IA completo.
    
    Args:
        model (str): Modelo por defecto a utilizar
        type_model ('t2im' o 't2v'): Tipo de modelo a usar (Solo disponibles modelos t2im)
        threads (int): Número de hilos para Waitress
        enable_memory_monitor (bool): Activar monitoreo de memoria
        
    Returns:
        flask.Flask: La aplicación Flask creada
    """
    # Configurar valores por defecto
    config = ServerConfigModels(
        model=model,
        type_models=type_model,
        custom_model=custom_model,
        custom_pipeline=custom_pipeline,
        constructor_pipeline=constructor_pipeline,
        components=components
    )
    
    # Crear la aplicación
    app = create_app(config)
    
    # Ejecutar con Waitress en un hilo separado
    run_waitress_server(
        app,
        threads=threads,
        enable_memory_monitor=enable_memory_monitor
    )
    
    return app