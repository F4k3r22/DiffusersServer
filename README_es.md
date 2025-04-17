<h1 align="center">DiffusersServer</h1>

<div align="center">
  <img src="static/Diffusers_Server.png" alt="DiffusersServer Logo" width="200"/>
</div>

<p align="center">
  🚀 Nueva solución tipo Ollama, pero diseñada específicamente para modelos de generación de imágenes (Text-to-Image).
</p>

---

## 🌟 ¿Qué es DiffusersServer?

**DiffusersServer** es un servidor de inferencia basado en Flask y Waitress que permite generar imágenes a partir de texto (*Text-to-Image*) utilizando modelos avanzados de difusión.

Compatible con **Stable Diffusion 3**, **Stable Diffusion 3.5**, **Flux**, y **Stable Diffusion v1.5**, proporciona una API REST eficiente para integrar generación de imágenes en tus aplicaciones.

## ⚡ Características principales

✅ **Soporte para múltiples modelos**

- Stable Diffusion 3 *(Medium)*
- Stable Diffusion 3.5 *(Large, Large-Turbo, Medium)*
- Flux *(Flux 1 Schnell, Flux 1 Dev)*
- Stable Diffusion v1.5

✅ **Compatibilidad con GPU y MPS**

- Aceleración con CUDA (GPUs NVIDIA)
- Compatibilidad con MPS (Macs con chips M1/M2)

✅ **Servidor eficiente y escalable**

- Implementación con Flask + Waitress
- Soporte para múltiples hilos
- Carga los modelos en memoria una sola vez

✅ **API REST fácil de usar**

- Endpoint para inferencia: `POST /api/diffusers/inference`
- Parámetros personalizables: prompt, modelo, tamaño de imagen, cantidad de imágenes

✅ **Gestión optimizada de memoria**

- *CPU offloading* en modelos Flux para reducir uso de VRAM
- Monitoreo opcional de consumo de memoria

---

## 🚀 DiffusersServer está diseñado para ofrecer una solución ligera, rápida y flexible para la generación de imágenes a partir de texto.

Si te gusta el proyecto, ¡considera darle una ⭐!

## 🚀Instalar DiffusersServer

```bash
git clone https://github.com/F4k3r22/DiffusersServer.git
cd DiffusersServer
pip install .
```

## 🚀Instalar DiffusersServer via Pypi

```bash
pip install DiffusersServer
```

## 🖥️Iniciar tu servidor

```python
from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='black-forest-labs/FLUX.1-schnell',
    type_model='t2im',
    threads=3,
    enable_memory_monitor=True
)
```

Asi de facil es levantar tu servidor de inferencia local con DiffusersServer en menos de 20 lineas de código

## ⚡Peticiones a tu servidor

### Generar una imagen

```python
import requests
import json
import os
from datetime import datetime
import re
import urllib.parse
import platform

# URL del servidor
server_url = "http://localhost:8500/api/diffusers/inference"
base_url = "http://localhost:8500"  

# Datos para enviar
data = {
    "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
    "num_inference_steps" : 30,
    "num_images" : 1
}

# Toma en cuenta que hay un funcionamiento raro con el num_images si es mayor que 1, se va llenando la memoria
# En proporción de 4.833 GB por imagen (Con stabilityai/stable-diffusion-3.5-medium)
# Igual se limpia la memoria automaticamente despues de la inferencia para no saturar la memoria excesivamente

# Es decir SD3.5 memdium usa 19.137GB de VRAM cargado en memoria, y cuando se pide una imagen sube 23.970GB de VRAM
# Y cuando se termina de generar esta imagen el uso de memoria vuelve al 19.137GB de la carga inicial

# Realizar la solicitud POST
print(f"Enviando prompt: \"{data['prompt']}\"")
print("Generando imagen... (esto puede tomar un tiempo)")
response = requests.post(server_url, json=data)

# Verificar la respuesta
if response.status_code == 200:
    result = response.json()
    image_url = result['response']
    print("¡Solicitud exitosa!")
    print(f"URL de la imagen generada: {image_url}")
```

## Stats del Servidor

### Listar modelos disponibles

```python
import requests

server_url = "http://localhost:8500/api/models"

def list_models():
    url = server_url
    reseponse = requests.get(url=url)
    reseponse.json()
    print(reseponse.json())

list_models_api = list_models()
```

### Obtener el uso de Memoria del Servidor

```python
import requests

memory = 'http://localhost:8500/api/status'

def get_memory_usage():
    url = memory
    response = requests.get(url=url)
    response.json()
    print(response.json())

memory_list = get_memory_usage()
```

---

## 🚀 Planes a Futuro

Actualmente estamos trabajando en:

- **Mejorar y optimizar la integración de los modelos Text-to-Image (T2Img) y Text-to-Video (T2V)**
- **Desarrollar un sistema que permita a los usuarios crear Pipelines personalizados para sus propias API's**
- **Ampliar las capacidades de personalización según las necesidades específicas de cada proyecto**

---

## Para modelos T2V

Para utilizar los modelos T2V, primero debes instalar la versión más reciente de `diffusers` directamente desde el repositorio principal:

```bash
pip install git+https://github.com/huggingface/diffusers
```

### Requisitos minimos para una inferencia óptima

Para una inferencia óptima, se recomienda:

- **GPU con al menos 48GB de VRAM**
- **Sistema con 64GB de RAM**

### Levantar tu servidor para modelos T2V

```python
from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
    type_model='t2v'
)
```

### Peticiones

#### Generar un video

```python
import requests
import os

def generate_video(prompt : str):
    url = 'http://127.0.0.1:8500/api/diffusers/video/inference'
    payload = {
        "prompt": prompt
    }

    print(f"Petición con el prompt: {prompt}")

    try:
        response = requests.post(url=url, json=payload)
        result = response.json()
        video_url = result['response']
        print("¡Solicitud exitosa!")
        print(f"URL del video generado: {video_url}")
  
    except Exception as e:
        print(str(e))

generate_video("Police cars chasing a Ferrari in Miami at dusk with gunshots, explosions and lots of chaos and speed, cinematic and action style")
```

#### Descargar el video generado

```python
import requests
import os

def download_video(filename, ruta_destino=None):
    """
    Descarga un video usando la API de Flask.
  
    Args:
        filename: Nombre del archivo a descargar
        ruta_destino: Ruta donde guardar el archivo (opcional)
  
    Returns:
        str: Ruta donde se guardó el archivo
    """
    base_url = "http://127.0.0.1:8500"
  
    # Construir URL completa
    url = f"{base_url}/video/{filename}"
  
    # Realizar la petición
    response = requests.get(url, stream=True)
  
    # Verificar si la petición fue exitosa
    if response.status_code == 200:
        # Determinar ruta de destino
        if ruta_destino is None:
            ruta_destino = os.path.join(os.getcwd(), filename)
      
        # Guardar el archivo
        with open(ruta_destino, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
      
        print(f"Video descargado y guardado en: {ruta_destino}")
        return ruta_destino
    else:
        print(f"Error al descargar el video. Código de estado: {response.status_code}")
        return None

download = download_video("videoff52dbc5.mp4")
```

# Donaciones 💸

Si deseas apoyar este proyecto, puedes hacer una donación a través de PayPal:

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=KZZ88H2ME98ZG)

Tu donativo permite mantener y expandir nuestros proyectos de código abierto en beneficio de toda la comunidad.
