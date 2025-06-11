# Voy a mudar todo el servidor a un servidor asincrono con FastAPI y Uvicorn
# Mientras complete esto, el servidor actual sigue funcionando
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .Pipelines import TextToImagePipelineSD3, TextToImagePipelineFlux, TextToImagePipelineSD
import logging
from diffusers.utils.export_utils import export_to_video
from diffusers import *
from .superpipeline import *
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

class Utils:
    def __init__(self, host: str = '0.0.0.0', port: int = 8500):
        self.service_url = f"http://{host}:{port}"
        self.image_dir = os.path.join(tempfile.gettempdir(), "images")
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.video_dir = os.path.join(tempfile.gettempdir(), "videos")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir)

    def save_image(self, image):
        filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
        image_path = os.path.join(self.image_dir, filename)
        logger.info(f"Saving image to {image_path}")
        image.save(image_path)
        return os.path.join(self.service_url, "images", filename)

    def save_video(self, video, fps):
        filename = "video" + str(uuid.uuid4()).split("-")[0] + ".mp4"
        video_path = os.path.join(self.video_dir, filename)
        export = export_to_video(video, video_path, fps=fps)
        logger.info(f"Saving video to {video_path}")
        return os.path.join(self.service_url, "video", filename)

@dataclass
class ServerConfigModels:
    model: str = 'stabilityai/stable-diffusion-3-medium'  # Valor predeterminado
    type_models: str = 't2im'  # Solo hay t2im y t2v
    custom_model : bool = False
    constructor_pipeline: Optional[Type] = None
    custom_pipeline: Optional[Type] = None  # Añadimos valor por defecto
    components: Optional[Dict[str, Any]] = None
    api_name: Optional[str] = 'custom_api'
    torch_dtype: Optional[torch.dtype] = None

def create_app(config=None):
    app = FastAPI()

    # Configuración del logger
    logging.basicConfig(level=logging.INFO)
    global logger
    logger = logging.getLogger(__name__)

    if config is None:
        config = ServerConfigModels()

    @app.get("/")
    async def root():
        return {"message": "Welcome to the Diffusers Server"}

    # Configuración de CORS. Lo hago aqui porque en Apps anteriores tuve errores al iniciar el servidor
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app