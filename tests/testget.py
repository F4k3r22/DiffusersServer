import requests

server_url = "http://localhost:8500/api/models"
memory = 'http://localhost:8500/api/status'

def list_models():
    url = server_url
    reseponse = requests.get(url=url)
    reseponse.json()
    print(reseponse.json())

list_models_api = list_models()

def get_memory_usage():
    url = memory
    response = requests.get(url=url)
    response.json()
    print(response.json())

memory_list = get_memory_usage()

""" 
Ejemplo de respuesta:

list_models:
{'all_models': {'Flux': ['black-forest-labs/FLUX.1-dev', 
    'black-forest-labs/FLUX.1-schnell'], 
    'SD3': ['stabilityai/stable-diffusion-3-medium'], 
    'SD3_5': ['stabilityai/stable-diffusion-3.5-large', 
    'stabilityai/stable-diffusion-3.5-large-turbo', 
    'stabilityai/stable-diffusion-3.5-medium'], 
    'WanT2V': ['Wan-AI/Wan2.1-T2V-14B-Diffusers', 
    'Wan-AI/Wan2.1-T2V-1.3B-Diffusers']}, 
    'current_model': 'stabilityai/stable-diffusion-3.5-medium', 'type': 't2im'}

get_memory_usage:
{'current_model': 'stabilityai/stable-diffusion-3.5-medium', 
    'memory': {'device': 'NVIDIA L40S', 
    'memory_allocated_gb': 17.09, 'memory_reserved_gb': 17.31}, 'type_models': 't2im'}
"""