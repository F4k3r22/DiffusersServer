from diffusers.pipelines import *
from diffusers  import *
import torch
from typing import Optional, Dict, Any, Type
import logging

logger = logging.getLogger(__name__)

class SuperPipelinesT2Img:
    def __init__(self, model_path: str, 
                pipeline: Type, 
                torch_dtype = torch.bfloat16, 
                components: Optional[Dict[str, Any]] = None,):
        """
        Clase para crear tus Pipelines personalizados para tu API custom
        Args:
            model_path: Ruta o nombre del modelo
            pipeline: Clase del pipeline a utilizar
            torch_dtype: Tipo de datos de PyTorch a utilizar
            components: Diccionario de componentes personalizados
        """
        self.model_path = model_path
        self.pipeline = pipeline
        self.torch_dtype = torch_dtype
        self.components = components or {}
        self.device: str = None
    
    def start(self):
        if torch.cuda.is_available():
            logger.info("Loading CUDA")
            model_path = self.model_path
            self.device = 'cuda'
            self.pipeline = self.pipeline.from_pretrained(
                    model_path,
                    torch_dtype = self.torch_dtype,
                    ** self.components
                ).to(device=self.device)
        elif torch.backends.mps.is_available():
            logger.info("Loading MPS for Mac M Series")
            model_path = self.model_path
            self.device = 'mps'
            self.pipeline = self.pipeline.from_pretrained(
                    model_path,
                    torch_dtype = self.torch_dtype,
                    **self.components
                ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")
        
        return self