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
    model: str = ''
    type_models: str = 't2im' # Solo hay t2im y t2v (Por ahorita aún en desarrollo y solo compatible con WanT2V)

@dataclass
class PresetModels:
    SD3 : list = ['stabilityai/stable-diffusion-3-medium']
    SD3_5: list = ['stabilityai/stable-diffusion-3.5-large', 'stabilityai/stable-diffusion-3.5-large-turbo', 'stabilityai/stable-diffusion-3.5-medium']
    Flux: list = ['black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell']
    WanT2V: list = ['Wan-AI/Wan2.1-T2V-14B-Diffusers', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers']


class RouteModels:
    def __init__(self, model: str = '', type_models: str = 't2im'):
        pass

def create_app(config=None):
    app = Flask(__name__)
    CORS(app)
    app.config['SERVER_CONFIG'] = config or ServerConfigModels()
    
    # Carga del modelo en memoria al iniciar la aplicación, una sola vez
    logger.info("Inicializando pipeline de modelo...")
    model_pipeline = TextToImagePipelineSD3(config.model if config else None)
    model_pipeline.start()  # Solo una llamada a start()
    app.config["MODEL_PIPELINE"] = model_pipeline
    logger.info("Pipeline inicializado y guardado en app.config")

    @app.route('/api/inference', methods=['POST'])
    def api():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Recuperamos el pipeline cargado previamente
        model_pipeline = app.config["MODEL_PIPELINE"]
        if not model_pipeline or not model_pipeline.pipeline:
            return jsonify({'error': 'Modelo no inicializado correctamente'}), 500

        prompt = data.get("prompt")
        if not prompt:
            return jsonify({'error': 'No se proporcionó prompt'}), 400
            
        try:
            # Si tu modelo o scheduler no son thread-safe, es recomendable clonar el scheduler
            scheduler = model_pipeline.pipeline.scheduler.from_config(model_pipeline.pipeline.scheduler.config)
            pipeline = StableDiffusion3Pipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
            
            generator = torch.Generator(device=model_pipeline.device)
            generator.manual_seed(random.randint(0, 10000000))
            
            logger.info(f"Procesando prompt: {prompt[:50]}...")
            output = pipeline(prompt, generator=generator)
            
            # Aquí guardarías la imagen y devolverías la respuesta
            image_url = save_image(output.images[0])
            return jsonify({'response': image_url})
        except Exception as e:
            logger.error(f"Error en inferencia: {str(e)}")
            return jsonify({'error': f'Error en procesamiento: {str(e)}'}), 500

    @app.route('/images/<filename>')
    def serve_image(filename):
        import tempfile
        image_dir = os.path.join(tempfile.gettempdir(), "images")
        return send_from_directory(image_dir, filename)
    
    return app
