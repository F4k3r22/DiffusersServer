<h1 align="center">DiffusersServer</h1>

<div align="center">
  <img src="static/Diffusers_Server.png" alt="DiffusersServer Logo" width="200"/>
</div>

<p align="center">
  🚀 New Ollama-type solution, but specifically designed for image generation models (Text-to-Image).
</p>

<p align="center">
  <i><a href="README_es.md">También disponible en Español</a> | <a href="README.md">Also available in Spanish</a></i>
</p>

---

## 🌟 What is DiffusersServer?

**DiffusersServer** is an inference server based on Flask and Waitress that allows generating images from text (*Text-to-Image*) using advanced diffusion models.

Compatible with **Stable Diffusion 3**, **Stable Diffusion 3.5**, **Flux**, and **Stable Diffusion v1.5**, it provides an efficient REST API to integrate image generation into your applications.

## ⚡ Main features

✅ **Support for multiple models**

- Stable Diffusion 3 *(Medium)*
- Stable Diffusion 3.5 *(Large, Large-Turbo, Medium)*
- Flux *(Flux 1 Schnell, Flux 1 Dev)*
- Stable Diffusion v1.5

✅ **GPU and MPS compatibility**

- Acceleration with CUDA (NVIDIA GPUs)
- MPS compatibility (Macs with M1/M2 chips)

✅ **Efficient and scalable server**

- Implementation with Flask + Waitress
- Support for multiple threads
- Loads models in memory only once

✅ **Easy-to-use REST API**

- Inference endpoint: `POST /api/diffusers/inference`
- Customizable parameters: prompt, model, image size, number of images

✅ **Optimized memory management**

- *CPU offloading* in Flux models to reduce VRAM usage
- Optional memory consumption monitoring

---

## 🚀 DiffusersServer is designed to offer a lightweight, fast, and flexible solution for text-to-image generation.

If you like the project, consider giving it a ⭐!

## 🚀Install DiffusersServer

```bash
git clone https://github.com/F4k3r22/DiffusersServer.git
cd DiffusersServer
pip install .
```

## 🚀Install DiffusersServer via Pypi

```bash
pip install DiffusersServer
```

## 🖥️Start your server

```python
from DiffusersServer import DiffusersServerApp

app = DiffusersServerApp(
    model='black-forest-labs/FLUX.1-schnell',
    type_model='t2im',
    threads=3,
    enable_memory_monitor=True
)
```

It's that easy to set up your local inference server with DiffusersServer in less than 20 lines of code

## ⚡Requests to your server

### Generate an image

```python
import requests
import json
import os
from datetime import datetime
import re
import urllib.parse
import platform

# Server URL
server_url = "http://localhost:8500/api/diffusers/inference"
base_url = "http://localhost:8500"  

# Data to send
data = {
    "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
    "num_inference_steps" : 30,
    "num_images" : 1
}

# Keep in mind that there's a strange behavior with num_images if it's greater than 1, the memory keeps filling up
# In proportion of 4.833 GB per image (With stabilityai/stable-diffusion-3.5-medium)
# The memory is automatically cleaned after inference to avoid excessive memory saturation

# That is, SD3.5 medium uses 19.137GB of VRAM loaded in memory, and when an image is requested it goes up to 23.970GB of VRAM
# And when this image is finished generating, the memory usage returns to the initial 19.137GB load

# Send the POST request
print(f"Sending prompt: \"{data['prompt']}\"")
print("Generating image... (this may take some time)")
response = requests.post(server_url, json=data)

# Check the response
if response.status_code == 200:
    result = response.json()
    image_url = result['response']
    print("Request successful!")
    print(f"Generated image URL: {image_url}")
```

## Server Stats

### List available models

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

### Get Server Memory Usage

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

## 🚀 Future Plans

We are currently working on:

- **Improving and optimizing the integration of Text-to-Image (T2Img) models**
- **Developing a system that allows users to create custom Pipelines for their own APIs**
- **Expanding customization capabilities according to the specific needs of each project**


# Donations 💸

If you wish to support this project, you can make a donation through PayPal:

[![Donate with PayPal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=KZZ88H2ME98ZG)

Your donation allows us to maintain and expand our open source projects for the benefit of the entire community.