# Pipelines.py

from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch
import os
import logging
from pydantic import BaseModel
import gc
import asyncio
from contextlib import asynccontextmanager
import time
from typing import Optional

logger = logging.getLogger(__name__)

class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: str | None = None
    n: int | None = None

class TextToImagePipelineSD3:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusion3Pipeline | None = None
        self.device: str | None = None
        
    def start(self):
        torch.set_float32_matmul_precision("high")
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
        
        if torch.cuda.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info(f"Loading CUDA with model: {model_path}")
            self.device = "cuda"
            
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
            gc.collect()
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16" if "fp16" in model_path else None,
                low_cpu_mem_usage=True,
            )
            
            self.pipeline = self.pipeline.to(device=self.device)
            
            
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
            except Exception as e:
                logger.info(f"XFormers not available: {e}")
            
            logger.info("Skipping torch.compile to avoid memory leaks")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            
            logger.info("CUDA pipeline fully optimized and ready")
            
        elif torch.backends.mps.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-medium"
            logger.info(f"Loading MPS for Mac M Series with model: {model_path}")
            self.device = "mps"
            
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            ).to(device=self.device)
            
                
            logger.info("MPS pipeline optimized and ready")
            
        else:
            raise Exception("No CUDA or MPS device available")
        

        self._warmup()
        
        logger.info("Pipeline initialization completed successfully")
    
    def _warmup(self):
        if self.pipeline:
            logger.info("Running warmup inference...")
            with torch.no_grad():
                _ = self.pipeline(
                    prompt="warmup",
                    num_inference_steps=1,
                    height=512,
                    width=512,
                    guidance_scale=1.0,
                )
            
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.ipc_collect()
            
            gc.collect()
            logger.info("Warmup completed with memory cleanup")

class TextToImagePipelineFlux:
    def __init__(self, model_path: str | None = None, low_vram: bool = False):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: FluxPipeline = None
        self.device: str = None
        self.low_vram = low_vram

    def start(self):
        if torch.cuda.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para CUDA.
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
            if self.low_vram:
                self.pipeline.enable_model_cpu_offload()
            else:
                pass
        elif torch.backends.mps.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para MPS.
            model_path = self.model_path or "black-forest-labs/FLUX.1-schnell"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class TextToImagePipelineSD:
    def __init__(self, model_path: str | None = None):
        """
        Inicialización de la clase con la ruta del modelo.
        Si no se proporciona, se obtiene de la variable de entorno.
        """
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: StableDiffusionPipeline = None
        self.device: str = None

    def start(self):
        if torch.cuda.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para CUDA.
            model_path = self.model_path or "sd-legacy/stable-diffusion-v1-5"
            logger.info("Loading CUDA")
            self.device = "cuda" 
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        elif torch.backends.mps.is_available():
            # Si no se definió model_path, se asigna el valor por defecto para MPS.
            model_path = self.model_path or "sd-legacy/stable-diffusion-v1-5"
            logger.info("Loading MPS for Mac M Series")
            self.device = "mps"
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            ).to(device=self.device)
        else:
            raise Exception("No CUDA or MPS device available")

class VAELock:    
    def __init__(self, 
                 cleanup_delay: float = 0.1,
                 enable_memory_tracking: bool = True,
                 max_wait_time: Optional[float] = 30.0):
        self.lock = asyncio.Lock()
        self.active_decodes = 0
        self.waiting_count = 0
        self.total_processed = 0
        self.cleanup_delay = cleanup_delay
        self.enable_memory_tracking = enable_memory_tracking
        self.max_wait_time = max_wait_time
        
        self.stats = {
            'total_decodes': 0,
            'total_wait_time': 0.0,
            'max_wait_time': 0.0,
            'total_decode_time': 0.0,
            'max_decode_time': 0.0,
            'memory_peaks': []
        }
        
        self._acquire_time = None
        self._decode_start_time = None
    
    def get_gpu_memory_info(self) -> dict:
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                           torch.cuda.memory_allocated()) / 1024**3
            }
        return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}
    
    async def acquire_with_timeout(self):
        if self.max_wait_time:
            try:
                await asyncio.wait_for(
                    self.lock.acquire(), 
                    timeout=self.max_wait_time
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"VAE decode timeout: waited {self.max_wait_time}s. "
                    f"There are {self.waiting_count} requests queued."
                )
        else:
            await self.lock.acquire()
    
    async def __aenter__(self):
        self.waiting_count += 1
        wait_start = time.time()
        
        if self.waiting_count > 3:
            logger.warning(
                f"VAE Queue congested: {self.waiting_count} requests waiting"
            )
        elif self.waiting_count > 1:
            logger.info(f"VAE Queue: {self.waiting_count} requests waiting")
        
        try:
            await self.acquire_with_timeout()
            
            wait_time = time.time() - wait_start
            self.stats['total_wait_time'] += wait_time
            self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)
            
            if wait_time > 2.0:
                logger.warning(f"VAE decode waited {wait_time:.2f}s to start")
            
            self.waiting_count -= 1
            self.active_decodes = 1  
            self._decode_start_time = time.time()
            
            if self.enable_memory_tracking:
                mem_info = self.get_gpu_memory_info()
                logger.info(
                    f"VAE decode started | "
                    f"GPU: {mem_info['allocated_gb']:.2f}GB used, "
                    f"{mem_info['free_gb']:.2f}GB free | "
                    f"Cola: {self.waiting_count}"
                )
            else:
                logger.info(f"VAE decode started | Queue: {self.waiting_count}")
            
            return self
            
        except Exception as e:
            self.waiting_count -= 1
            raise e
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._decode_start_time:
                decode_time = time.time() - self._decode_start_time
                self.stats['total_decode_time'] += decode_time
                self.stats['max_decode_time'] = max(
                    self.stats['max_decode_time'], 
                    decode_time
                )
                
                if decode_time > 5.0:
                    logger.warning(f"VAE decode took a while {decode_time:.2f}s")
            
            self.active_decodes = 0
            self.total_processed += 1
            self.stats['total_decodes'] += 1
            
            if self.enable_memory_tracking:
                mem_before = self.get_gpu_memory_info()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                
                if self.enable_memory_tracking:
                    if mem_before['reserved_gb'] - mem_before['allocated_gb'] > 1.0:
                        logger.info(
                            f"Fragmented memory detected: "
                            f"{mem_before['reserved_gb'] - mem_before['allocated_gb']:.2f}GB"
                        )
                        torch.cuda.empty_cache()
            
            gc.collect()
            
            if self.enable_memory_tracking:
                mem_after = self.get_gpu_memory_info()
                mem_freed = mem_before['allocated_gb'] - mem_after['allocated_gb']
                
                self.stats['memory_peaks'].append(mem_before['allocated_gb'])
                if len(self.stats['memory_peaks']) > 100:
                    self.stats['memory_peaks'].pop(0)
                
                logger.info(
                    f"VAE decode completed | " 
                    f"Memory freed: {mem_freed:.2f}GB | " 
                    f"GPU now: {mem_after['allocated_gb']:.2f}GB used | " 
                    f"Total processed: {self.total_processed}"
                )
            else:
                logger.info(
                    f"VAE decode completed | Total processed: {self.total_processed}"
                )
            
            if self.waiting_count > 0:
                await asyncio.sleep(min(self.cleanup_delay, 0.05))
            else:
                await asyncio.sleep(self.cleanup_delay)
            
            if exc_type is not None:
                logger.error(
                    f"VAE decode failed with error: {exc_type.__name__}: {exc_val}"
                )
            
        finally:
            self.lock.release()
            
            if self.waiting_count > 2:
                logger.info(f"VAE released. {self.waiting_count} remaining in queue")
    
    def get_stats(self) -> dict:
        stats = self.stats.copy()
        
        if stats['total_decodes'] > 0:
            stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_decodes']
            stats['avg_decode_time'] = stats['total_decode_time'] / stats['total_decodes']
        else:
            stats['avg_wait_time'] = 0
            stats['avg_decode_time'] = 0
        
        stats['current_waiting'] = self.waiting_count
        stats['is_active'] = self.active_decodes > 0
        
        if self.enable_memory_tracking and stats['memory_peaks']:
            stats['avg_memory_peak_gb'] = sum(stats['memory_peaks']) / len(stats['memory_peaks'])
            stats['max_memory_peak_gb'] = max(stats['memory_peaks'])
        
        return stats
    
    async def wait_for_idle(self, timeout: float = 10.0):
        start_time = time.time()
        while self.active_decodes > 0 or self.waiting_count > 0:
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for VAE to be free")
            await asyncio.sleep(0.1)
    
    @asynccontextmanager
    async def priority_decode(self):
        logger.warning("Using priority VAE decoding")
        async with self.lock:
            self.active_decodes = 1
            try:
                yield self
            finally:
                self.active_decodes = 0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()