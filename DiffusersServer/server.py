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

from dataclasses import dataclass, field
from typing import List

@dataclass
class PresetModels:
    SD3: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3-medium'])
    SD3_5: List[str] = field(default_factory=lambda: ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-large-turbo', 'stabilityai/stable-diffusion-3.5-medium'])
    Flux: List[str] = field(default_factory=lambda: ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell'])
    WanT2V: List[str] = field(default_factory=lambda: ['Wan-AI/Wan2.1-T2V-14B-Diffusers', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'])

class RouteModels:
    def __init__(self, model: str = '', type_models: str = 't2im'):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"

    def create_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Check if model exists in PresetModels
        preset_models = PresetModels()
        model_type = None

        # Determine which model type we're dealing with
        if self.model in preset_models.SD3:
            model_type = "SD3"
        elif self.model in preset_models.SD3_5:
            model_type = "SD3_5"
        elif self.model in preset_models.Flux:
            model_type = "Flux"
        elif self.model in preset_models.WanT2V:
            model_type = "WanT2V"
        else:
            model_type = "SD"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == 't2im':
            if model_type in ["SD3", "SD3_5"]:
                self.pipeline = TextToImagePipelineSD3(self.model)
            elif model_type == "Flux":
                self.pipeline = TextToImagePipelineFlux(self.model)
            elif model_type == "SD":
                self.pipeline = TextToImagePipelineSD(self.model)
            else:
                raise ValueError(f"Model type {model_type} not supported for text-to-image")
        elif self.type_models == 't2v':
            if model_type == "WanT2V":
                # Uncomment when VideoPipelines is implemented
                # self.pipeline = WanT2VPipelines(self.model)
                raise NotImplementedError("Text-to-video pipeline not yet implemented")
            else:
                raise ValueError(f"Model type {model_type} not supported for text-to-video")
        else:
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline

    def get_pipeline(self):
        if self.pipeline is None:
            self.create_pipeline()
        return self.pipeline


class ReturnPipelines:
    def __init__(self, model: str = '', type_models: str = 't2im', scheduler = None):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.scheduler = scheduler

    def create_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Check if model exists in PresetModels
        preset_models = PresetModels()
        model_type = None

        # Determine which model type we're dealing with
        if self.model in preset_models.SD3:
            model_type = "SD3"
        elif self.model in preset_models.SD3_5:
            model_type = "SD3_5"
        elif self.model in preset_models.Flux:
            model_type = "Flux"
        elif self.model in preset_models.WanT2V:
            model_type = "WanT2V"
        else:
            model_type = "SD"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == 't2im':
            if model_type in ["SD3", "SD3_5"]:
                self.pipeline = StableDiffusion3Pipeline.from_pretrained(self.model, scheduler=self.scheduler, torch_dtype=torch.float16)
                self.pipeline = self.pipeline.to(self.device)
            elif model_type == "Flux":
                self.pipeline = FluxPipeline.from_pretrained(self.model, scheduler=self.scheduler, torch_dtype=torch.float16)
                self.pipeline = self.pipeline.to(self.device)
            else:
                self.pipeline = StableDiffusionPipeline.from_pretrained(self.model, scheduler=self.scheduler, torch_dtype=torch.float16)
                self.pipeline = self.pipeline.to(self.device)
        elif self.type_models == 't2v':
            if model_type == "WanT2V":
                # Uncomment when VideoPipelines is implemented
                # self.pipeline = WanT2VPipelines(self.model)
                raise NotImplementedError("Text-to-video pipeline not yet implemented")
            else:
                raise ValueError(f"Model type {model_type} not supported for text-to-video")
        else:
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline

    def return_pipeline(self):
        if self.pipeline is None:
            self.create_pipeline()
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
    
    # Inicialización del router de modelos
    logger.info("Inicializando router de modelos...")
    model_router = RouteModels(
        model=app.config['SERVER_CONFIG'].model,
        type_models=app.config['SERVER_CONFIG'].type_models
    )
    model_pipeline = model_router.get_pipeline()
    model_pipeline.start()  # Iniciamos el pipeline
    app.config["MODEL_ROUTER"] = model_router
    app.config["MODEL_PIPELINE"] = model_pipeline
    logger.info(f"Pipeline inicializado para el modelo: {app.config['SERVER_CONFIG'].model}")

    @app.route('/api/diffusers/inference', methods=['POST'])
    def api():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Recuperamos el router y pipeline cargados previamente
        model_router = app.config["MODEL_ROUTER"]
        model_pipeline = app.config["MODEL_PIPELINE"]
        
        if not model_pipeline:
            return jsonify({'error': 'Modelo no inicializado correctamente'}), 500

        # Verificar si se solicita un modelo diferente
        requested_model = data.get("model")
        requested_type = data.get("type_models", "t2im")
        
        # Si se solicita un modelo diferente al actual, cambiamos
        current_model = app.config['SERVER_CONFIG'].model
        current_type = app.config['SERVER_CONFIG'].type_models
        
        if (requested_model and requested_model != current_model) or (requested_type != current_type):
            logger.info(f"Cambiando modelo de {current_model} a {requested_model or current_model}")
            
            # Actualizamos la configuración
            if requested_model:
                app.config['SERVER_CONFIG'].model = requested_model
            app.config['SERVER_CONFIG'].type_models = requested_type
            
            # Creamos nuevo router y pipeline
            model_router = RouteModels(
                model=app.config['SERVER_CONFIG'].model,
                type_models=app.config['SERVER_CONFIG'].type_models
            )
            model_pipeline = model_router.get_pipeline()
            model_pipeline.start()
            
            # Actualizamos en app.config
            app.config["MODEL_ROUTER"] = model_router
            app.config["MODEL_PIPELINE"] = model_pipeline

        # Extraemos los parámetros de la solicitud
        prompt = data.get("prompt")
        if not prompt:
            return jsonify({'error': 'No se proporcionó prompt'}), 400
        
        negative_prompt = data.get("negative_prompt", "")
        num_inference_steps = data.get("num_inference_steps", 28)
        num_images = data.get("num_images", 1)
            
        try:
            # Obtenemos el tipo de modelo actual
            model_type = None
            preset_models = PresetModels()
            current_model = app.config['SERVER_CONFIG'].model
            
            if current_model in preset_models.SD3 or current_model in preset_models.SD3_5:
                pipeline_class = StableDiffusion3Pipeline
            elif current_model in preset_models.Flux:
                pipeline_class = FluxPipeline
            else:
                pipeline_class = StableDiffusionPipeline
                
            # Clonamos el scheduler para thread-safety
            scheduler = model_pipeline.pipeline.scheduler.from_config(model_pipeline.pipeline.scheduler.config)
            
            # Creamos una instancia específica para esta solicitud
            return_pipeline = ReturnPipelines(
                model=current_model,
                type_models=app.config['SERVER_CONFIG'].type_models,
                scheduler=scheduler
            )
            
            # Obtenemos el pipeline optimizado para esta solicitud
            pipeline = return_pipeline.return_pipeline()
            
            # Configuramos el generador
            generator = torch.Generator(device=model_router.device)
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
                
            return jsonify({'response': image_urls})
            
        except Exception as e:
            logger.error(f"Error en inferencia: {str(e)}")
            return jsonify({'error': f'Error en procesamiento: {str(e)}'}), 500

    @app.route('/images/<filename>')
    def serve_image(filename):
        return send_from_directory(image_dir, filename)
    
    @app.route('/api/models', methods=['GET'])
    def list_models():
        preset_models = PresetModels()
        available_models = {
            "SD3": preset_models.SD3,
            "SD3_5": preset_models.SD3_5,
            "Flux": preset_models.Flux,
            "WanT2V": preset_models.WanT2V
        }
        return jsonify(available_models)
    
    return app