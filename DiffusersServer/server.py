# server.py

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from flask import Flask, request, jsonify, send_from_directory, make_response, Response
from flask_cors import CORS
from .Pipelines import TextToImagePipelineSD3, TextToImagePipelineFlux, TextToImagePipelineSD
import logging
from diffusers.utils.export_utils import export_to_video
import random
import uuid
import tempfile
from dataclasses import dataclass
import os
import torch
import gc
from typing import Union, Tuple, Optional, Dict, Any, Type
from dataclasses import dataclass, field
from typing import List

@dataclass
class PresetModels:
    SD3: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3-medium'])
    SD3_5: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-large-turbo', 'stabilityai/stable-diffusion-3.5-medium'])
    Flux: List[str] = field(default_factory=lambda: ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell'])
    WanT2V: List[str] = field(default_factory=lambda: ['Wan-AI/Wan2.1-T2V-14B-Diffusers', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'])
    LTXVideo: List[str] = field(default_factory=lambda: ['Lightricks/LTX-Video'])

class ModelPipelineInitializer:
    def __init__(self, model: str = '', type_models: str = 't2im'):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.model_type = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Check if model exists in PresetModels
        preset_models = PresetModels()

        # Determine which model type we're dealing with
        if self.model in preset_models.SD3:
            self.model_type = "SD3"
        elif self.model in preset_models.SD3_5:
            self.model_type = "SD3_5"
        elif self.model in preset_models.Flux:
            self.model_type = "Flux"
        elif self.model in preset_models.WanT2V:
            self.model_type = "WanT2V"
        elif self.model in preset_models.LTXVideo:
            self.model_type = "LTXVideo"
        else:
            self.model_type = "SD"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == 't2im':
            if self.model_type in ["SD3", "SD3_5"]:
                self.pipeline = TextToImagePipelineSD3(self.model)
            elif self.model_type == "Flux":
                self.pipeline = TextToImagePipelineFlux(self.model)
            elif self.model_type == "SD":
                self.pipeline = TextToImagePipelineSD(self.model)
            else:
                raise ValueError(f"Model type {self.model_type} not supported for text-to-image")
        elif self.type_models == 't2v':
            if self.model_type == "WanT2V":
                try: 
                    from .VideoPipelines import WanT2VPipelines
                    self.pipeline = WanT2VPipelines(self.model)
                except ImportError as e:
                    print('No se pudo importar correctamente, verifica tu versión de diffusers')
                    pass
            if self.model_type == "LTXVideo":
                try:
                    from .VideoPipelines import LTXT2VPipelines
                    self.pipeline = LTXT2VPipelines(self.model)
                except ImportError as e:
                    print('No se pudo importar correctamente, verifica tu versión de diffusers')
                    pass
            else:
                pass
        else:
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline


# Configuraciones del servidor
service_url = 'http://localhost:8500'
logger = logging.getLogger(__name__)

image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

video_dir = os.path.join(tempfile.gettempdir(), "videos")
if not os.path.exists(video_dir):
    os.makedirs(video_dir)

def save_image(image):
    filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
    image_path = os.path.join(image_dir, filename)
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return os.path.join(service_url, "images", filename)

def save_video(video, fps):
    filename = "video" + str(uuid.uuid4()).split("-")[0] + ".mp4"
    video_path = os.path.join(video_dir, filename)
    export = export_to_video(video, video_path, fps=fps)
    logger.info(f"Saving video to {video_path}")
    return os.path.join(service_url, "video", filename)


@dataclass
class ServerConfigModels:
    model: str = 'stabilityai/stable-diffusion-3-medium'  # Valor predeterminado
    type_models: str = 't2im'  # Solo hay t2im y t2v
    custom_model : bool = False
    custom_pipeline : Optional[Type]
    constructor_pipeline: Optional[Type]
    components: Optional[Dict[str, Any]] = None 

def create_app(config=None):
    app = Flask(__name__)
    CORS(app)
    app.config['SERVER_CONFIG'] = config or ServerConfigModels()

    configs = config or ServerConfigModels()
    
    # Inicialización del pipeline de modelo único
    logger.info(f"Inicializando pipeline para el modelo: {app.config['SERVER_CONFIG'].model}")
    model_initializer = ModelPipelineInitializer(
        model=app.config['SERVER_CONFIG'].model,
        type_models=app.config['SERVER_CONFIG'].type_models
    )
    model_pipeline = model_initializer.initialize_pipeline()
    model_pipeline.start()  # Iniciamos el pipeline
    app.config["MODEL_PIPELINE"] = model_pipeline
    app.config["MODEL_INITIALIZER"] = model_initializer
    logger.info("Pipeline inicializado y listo para recibir solicitudes")

    @app.route('/api/diffusers/inference', methods=['POST'])
    def api():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Recuperamos el pipeline cargado
        model_pipeline = app.config["MODEL_PIPELINE"]
        model_initializer = app.config["MODEL_INITIALIZER"]
        
        if not model_pipeline or not model_pipeline.pipeline:
            return jsonify({'error': 'Modelo no inicializado correctamente'}), 500

        # Si se solicita un modelo diferente, ignoramos esa solicitud y usamos el modelo actual
        requested_model = data.get("model")
        if requested_model and requested_model != app.config['SERVER_CONFIG'].model:
            logger.warning(f"Se solicitó el modelo '{requested_model}' pero se utilizará el modelo cargado inicialmente: '{app.config['SERVER_CONFIG'].model}'")

        # Extraemos los parámetros de la solicitud
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({'error': 'No se proporcionó prompt'}), 400
        
        negative_prompt = data.get("negative_prompt", "")
        num_inference_steps = data.get("num_inference_steps", 28)
        num_images = data.get("num_images", 1)
            
        try:
            # Usamos directamente el pipeline cargado, pero clonamos el scheduler para thread-safety
            scheduler = model_pipeline.pipeline.scheduler.from_config(model_pipeline.pipeline.scheduler.config)
            
            # Determinar el tipo de pipeline para clonar correctamente
            model_type = model_initializer.model_type
            
            if model_type in ["SD3", "SD3_5"]:
                pipeline = StableDiffusion3Pipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
            elif model_type == "Flux":
                pipeline = FluxPipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
            else:
                pipeline = StableDiffusionPipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
            
            # Configuramos el generador
            generator = torch.Generator(device=model_initializer.device)
            generator.manual_seed(random.randint(0, 10000000))
            
            # Procesamos la inferencia
            logger.info(f"Procesando prompt: {prompt[:50]}...")
            output = pipeline(
                prompt, 
                negative_prompt=negative_prompt, 
                generator=generator, 
                num_inference_steps=num_inference_steps, 
                num_images_per_prompt=num_images
            )
            
            # Guardamos la imagen y devolvemos la respuesta
            image_urls = []
            for i in range(len(output.images)):
                image_url = save_image(output.images[i])
                image_urls.append(image_url)
                
            # Limpieza después de inferencia (solo el pipeline clonado, mantenemos el original)
            del pipeline
            del output
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return jsonify({'response': image_urls})
            
        except Exception as e:
            logger.error(f"Error en inferencia: {str(e)}")
            return jsonify({'error': f'Error en procesamiento: {str(e)}'}), 500

    @app.route('/images/<filename>')
    def serve_image(filename):
        return send_from_directory(image_dir, filename)
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        # Devolvemos solo el modelo actual como disponible
        return jsonify({
            "current_model": app.config['SERVER_CONFIG'].model,
            "type": app.config['SERVER_CONFIG'].type_models,
            "all_models": {
                "type": "T2Img",
                "SD3": PresetModels().SD3,
                "SD3_5": PresetModels().SD3_5,
                "Flux": PresetModels().Flux,
                "type": "T2V",
                "WanT2V": PresetModels().WanT2V,
                "LTX-Video": PresetModels().LTXVideo,
            }
        })
    
    @app.route('/api/status', methods=['GET'])
    def get_status():
        memory_info = {}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_info = {
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "device": torch.cuda.get_device_name(0)
            }
        
        return jsonify({
            "current_model": app.config['SERVER_CONFIG'].model,
            "type_models": app.config['SERVER_CONFIG'].type_models,
            "memory": memory_info
        })
    
    @app.route('/api/diffusers/video/inference', methods=['POST'])
    def api_video() -> Union[Response, Tuple[Response, int]]:
        if configs.type_models == 't2v':
            try:
                from diffusers.pipelines.wan.pipeline_wan import WanPipeline
                from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
            
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400

                model_pipeline = app.config["MODEL_PIPELINE"]
                model_initializer = app.config["MODEL_INITIALIZER"]

                if not model_pipeline or not model_pipeline.pipeline:
                    return jsonify({'error': 'Modelo no inicializado correctamente'}), 500

                # Si se solicita un modelo diferente, ignoramos esa solicitud y usamos el modelo actual
                requested_model = data.get("model")
                if requested_model and requested_model != app.config['SERVER_CONFIG'].model:
                    logger.warning(f"Se solicitó el modelo '{requested_model}' pero se utilizará el modelo cargado inicialmente: '{app.config['SERVER_CONFIG'].model}'")

                # Extraemos los parámetros de la solicitud
                prompt = data.get("prompt")
                if not prompt:
                    return jsonify({'error': 'No se proporcionó prompt'}), 400
            
                height = data.get("height", 480)
                width = data.get("width", 832)
                num_frames = data.get("num_frames", 81)
                num_inference_steps = data.get("num_inference_steps", 50)
                fps = data.get("fps", 15)

                try:
                    # Solo clonamos el scheduler para thread-safety
                    scheduler = model_pipeline.pipeline.scheduler.from_config(model_pipeline.pipeline.scheduler.config)
        
                    # Determinar el tipo de pipeline para clonar correctamente
                    model_type = model_initializer.model_type

                    # Configurar generador con semilla aleatoria para reproducibilidad
                    generator = torch.Generator(device=model_initializer.device)
                    generator.manual_seed(random.randint(0, 10000000))

                    if model_type == "WanT2V":
                        pipeline = WanPipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
                    elif model_type == "LTXVideo":
                        pipeline = LTXPipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
                    else:
                        raise RuntimeError(f"Modelo {model_type} incompatible con T2V")
                
                    logger.info(f"Procesando prompt para video: {prompt[:50]}...")
                    output = pipeline(
                        prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        num_inference_steps=num_inference_steps,
                        generator=generator
                    ).frames[0]
                
                    video_url = save_video(output, fps=fps)
                
                    # Limpieza exhaustiva de recursos
                    del scheduler
                    del pipeline
                    del output
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Asegurar que todas las operaciones CUDA terminen

                    return make_response(jsonify({'response': video_url}), 200)
            
                except Exception as e:
                    logger.error(f"Error en inferencia de video: {str(e)}")
                    # Intentar limpieza incluso en caso de error
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return jsonify({'error': f'Error en procesamiento de video: {str(e)}'}), 500

            except ImportError as e:
                logger.error(f"Error de importación para T2V: {str(e)}")
                return jsonify({'error': 'Dependencias de T2V no disponibles'}), 500
        else:
            return jsonify({
                "response": "El servidor está operando en modo T2Img. Para utilizar esta API, por favor cambie al modo de generación de video."
            })
        
    @app.route('/video/<filename>')
    def serve_video(filename):
        if configs.type_models == 't2v':
            return send_from_directory(video_dir, filename)
        else:
            return jsonify({
                "response" : "Opción no disponible porque el servidor esta ejecutando modelos T2Img"
            })
    
    return app