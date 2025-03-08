# server.py

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from .Pipelines import TextToImagePipelineSD3, TextToImagePipelineFlux, TextToImagePipelineSD
#from .VideoPipelines import WanT2VPipelines
import logging
from diffusers.utils import export_to_video
import random
import uuid
import tempfile
from dataclasses import dataclass
import os
import torch
import gc

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
                # Uncomment when VideoPipelines is implemented
                # self.pipeline = WanT2VPipelines(self.model)
                raise NotImplementedError("Text-to-video pipeline not yet implemented")
            else:
                raise ValueError(f"Model type {self.model_type} not supported for text-to-video")
        else:
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline


# Configuraciones del servidor
service_url = 'http://localhost:8500'
logger = logging.getLogger(__name__)

image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def save_image(image):
    filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
    image_path = os.path.join(image_dir, filename)
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return os.path.join(service_url, "images", filename)

@dataclass
class ServerConfigModels:
    model: str = 'stabilityai/stable-diffusion-3-medium'  # Valor predeterminado
    type_models: str = 't2im'  # Solo hay t2im y t2v

def create_app(config=None):
    app = Flask(__name__)
    CORS(app)
    app.config['SERVER_CONFIG'] = config or ServerConfigModels()
    
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
    def api_video():
        return jsonify({
            "status": "En desarrollo",
            "description": "Los modelos de Text-to-Video (T2V) se encuentran actualmente en fase de implementación. Agradecemos su paciencia mientras optimizamos esta funcionalidad."
        })
    
    return app