# create_server.py

from .Pipelines import *
from .server import *
from .waitress_server import *
from .serverasync import *
from .uvicorn_diffu import *
import asyncio

# Función principal para iniciar un servidor completo
def create_inference_server(
    model:str,
    type_model: str = 't2im',
    threads=5,
    enable_memory_monitor=True,
    custom_model: bool = False,
    custom_pipeline: Optional[Type] | None = None,
    constructor_pipeline: Optional[Type] | None = None,
    components: Optional[Dict[str, Any]] = None,
    api_name: Optional[str] = 'custom_api',
    torch_dtype = torch.bfloat16
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
        components=components,
        api_name=api_name,
        torch_dtype=torch_dtype
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

def create_inference_server_Async(
    model:str,
    type_model: str = 't2im',
    host: str = '0.0.0.0',
    port: int = 8500,
    threads=5,
    enable_memory_monitor=True,
    custom_model: bool = False,
    custom_pipeline: Optional[Type] | None = None,
    constructor_pipeline: Optional[Type] | None = None,
    components: Optional[Dict[str, Any]] = None,
    api_name: Optional[str] = 'custom_api',
    torch_dtype = torch.bfloat16
):
    config = ServerConfigModels(
        model=model,
        type_models=type_model,
        custom_model=custom_model,
        custom_pipeline=custom_pipeline,
        constructor_pipeline=constructor_pipeline,
        components=components,
        api_name=api_name,
        torch_dtype=torch_dtype,
        host=host,
        port=port
    )

    app = create_app_fastapi(config)

    asyncio.run(run_uvicorn_server(
        app, 
        host=host, 
        port=port, 
        workers=threads,
        enable_memory_monitor=enable_memory_monitor
    ))

    return app