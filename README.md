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

Estamos trabajando en la integración de una API para modelos Text-to-Video (T2V), comenzando con Wan 2.1. Esto permitirá generar videos a partir de texto, ampliando las capacidades de DiffusersServer más allá de la generación de imágenes.

Tambien estamos trabajando en una mejor integración en los modelos pre existente T2Img de Diffusers

---

## Para modelos T2V
Primero instala diffusers desde la rama main del repositorio de diffusers
```bash
pip install git+https://github.com/huggingface/diffusers
```

# Donaciones 💸

Si deseas apoyar este proyecto, puedes hacer una donación a través de PayPal:

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=KZZ88H2ME98ZG)

Tu donativo permite mantener y expandir nuestros proyectos de código abierto en beneficio de toda la comunidad.
