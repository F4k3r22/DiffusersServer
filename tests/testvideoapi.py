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

#generate_video("Police cars chasing a Ferrari in Miami at dusk with gunshots, explosions and lots of chaos and speed, cinematic and action style")

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