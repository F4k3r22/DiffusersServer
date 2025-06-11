import os
import re
import uuid
import urllib.parse
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración
SERVER_URL = "http://localhost:8500/api/diffusers/inference"
BASE_URL   = "http://localhost:8500"
DOWNLOAD_FOLDER = "imagenes_generadas"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def download_and_save(url: str, prompt: str, idx: int = 0) -> str:
    """
    Descarga la imagen de `url` y la guarda en DOWNLOAD_FOLDER.
    Devuelve la ruta local donde se guardó.
    """
    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    direct_url = f"{BASE_URL}/images/{file_name}"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = re.sub(r'[^\w\-_]', '', prompt[:20].replace(" ", "_"))
    suffix = f"_{idx}" if isinstance(idx, int) else ""
    save_filename = f"{timestamp}_{prompt_slug}{suffix}.png"
    save_path = os.path.join(DOWNLOAD_FOLDER, save_filename)

    resp = requests.get(direct_url)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path

def single_request(data: dict) -> list:
    """
    Hace una petición a la API y devuelve la lista de URLs recibidas.
    """
    resp = requests.post(SERVER_URL, json=data)
    resp.raise_for_status()
    body = resp.json().get("response", [])
    # Asegurarse de que sea lista
    return body if isinstance(body, list) else [body]

def test_parallel_requests(
    data: dict,
    n_requests: int = 5,
    max_workers: int = 5
) -> None:
    """
    Lanza `n_requests` peticiones en paralelo y descarga todas las imágenes resultantes.
    
    Args:
        data: payload JSON para cada petición (ej. prompt, num_inference_steps...)
        n_requests: número total de peticiones concurrentes.
        max_workers: tamaño del pool de hilos.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Lanzar peticiones
        future_to_i = {
            executor.submit(single_request, data): i
            for i in range(n_requests)
        }
        for future in as_completed(future_to_i):
            i = future_to_i[future]
            try:
                urls = future.result()
                print(f"[Req {i}] Recibido {len(urls)} URL(s).")
                # Descargar cada URL
                for idx, url in enumerate(urls):
                    save_path = download_and_save(url, data["prompt"], idx)
                    print(f"[Req {i}] Imagen {idx} guardada en {save_path}")
            except Exception as e:
                print(f"[Req {i}] Error: {e}")

if __name__ == "__main__":
    # Ejemplo de uso: 10 peticiones en paralelo con 5 hilos
    data = {
        "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
        "num_inference_steps": 30,
        "num_images": 1
    }
    test_parallel_requests(data, n_requests=10, max_workers=5)
