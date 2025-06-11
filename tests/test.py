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



# Crear una carpeta para guardar las imágenes si no existe
download_folder = "imagenes_generadas"
os.makedirs(download_folder, exist_ok=True)

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
    
# Verificar si la respuesta es una lista de URLs
if isinstance(image_url, list):
    for idx, url in enumerate(image_url):
        file_name = os.path.basename(urllib.parse.urlparse(url).path)
        direct_url = f"{base_url}/images/{file_name}"
        
        # Crear un nombre de archivo para guardar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_slug = data["prompt"][:20].replace(" ", "_")
        safe_prompt_slug = re.sub(r'[^\w\-_]', '', prompt_slug)
        save_filename = f"{timestamp}_{safe_prompt_slug}_{idx}.png"
        save_path = os.path.join(download_folder, save_filename)
        
        # Descargar la imagen
        try:
            print(f"Descargando imagen desde: {direct_url}")
            img_response = requests.get(direct_url)
            if img_response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(img_response.content)
                print(f"Imagen descargada exitosamente a: {save_path}")
            else:
                print(f"Error al descargar la imagen: {img_response.status_code}")
                print(img_response.text)
        except Exception as e:
            print(f"Error al descargar la imagen: {e}")
else:
    # Caso en el que solo haya una URL
    file_name = os.path.basename(urllib.parse.urlparse(image_url).path)
    direct_url = f"{base_url}/images/{file_name}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = data["prompt"][:20].replace(" ", "_")
    safe_prompt_slug = re.sub(r'[^\w\-_]', '', prompt_slug)
    save_filename = f"{timestamp}_{safe_prompt_slug}.png"
    save_path = os.path.join(download_folder, save_filename)
    
    try:
        print(f"Descargando imagen desde: {direct_url}")
        img_response = requests.get(direct_url)
        if img_response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(img_response.content)
            print(f"Imagen descargada exitosamente a: {save_path}")
        else:
            print(f"Error al descargar la imagen: {img_response.status_code}")
            print(img_response.text)
    except Exception as e:
        print(f"Error al descargar la imagen: {e}")
