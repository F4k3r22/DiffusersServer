from diffusers import StableDiffusion3Pipeline
import asyncio
from pydantic import BaseModel
import torch
import logging
import os

logger = logging.getLogger(__name__)

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None

class TextToImagePipelineSD3:
    pipeline: StableDiffusion3Pipeline = None
    device: str = None

    def start(self):
        if torch.cuda.is_available():
            model_path = os.getenv("MODEL_PATH", "stabilityai/stable-diffusion-3.5-large")
            logger.info("Loading CUDA")
            self.device = "cuda"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            model_path = os.getenv("MODEL_PATH", "stabilityai/stable-diffusion-3.5-medium")
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")
