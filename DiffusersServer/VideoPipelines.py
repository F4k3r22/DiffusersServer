# VideoPipelines.py

import torch
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
import os
import logging
from pydantic import BaseModel
from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline

logger = logging.getLogger(__name__)

class TextToVideoInput(BaseModel):
    model: str
    prompt: str

class WanT2VPipelines:
    def __init__(self, model_path: str | None = None):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: WanPipeline = None
        self.device: str = None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
            self.pipeline = WanPipeline.from_pretrained(model_path,
            torch_dtype=torch.bfloat16,
            vae=self.vae,
            ).to(device=self.device)
        elif  torch.backends.mps.is_available():
            model_path = self.model_path or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
            self.pipeline = WanPipeline.from_pretrained(model_path, 
                vae=self.vae,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class LTXT2VPipelines:
    def __init__(self, model_path: str | None = None):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: LTXPipeline = None
        self.device: str = None

    def start(self):
        if torch.cuda.is_available():
            model_path = self.model_path or "Lightricks/LTX-Video"
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = LTXPipeline.from_pretrained(model_path, 
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        elif  torch.backends.mps.is_available():
            model_path = self.model_path or "Lightricks/LTX-Video"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = LTXPipeline.from_pretrained(model_path, 
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")
