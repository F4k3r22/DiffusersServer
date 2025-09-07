import os
import re
import uuid
import urllib.parse
import requests
import random
import time
import statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuración
SERVER_URL = "http://localhost:8500/api/diffusers/inference"
BASE_URL   = "http://localhost:8500"
DOWNLOAD_FOLDER = "imagenes_generadas"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

SAVE_PROBABILITY = 0.1  # 10%

# Métricas globales
latencies = []
success_count = 0
error_count = 0
request_count = 0

def download_and_save(url: str, prompt: str, idx: int = 0) -> str:
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

def single_request(data: dict) -> tuple[bool, float, list]:
    """
    Hace una petición y devuelve (success, latency, urls).
    """
    start = time.time()
    try:
        resp = requests.post(SERVER_URL, json=data, timeout=480)
        resp.raise_for_status()
        latency = time.time() - start
        body = resp.json().get("response", [])
        urls = body if isinstance(body, list) else [body]
        return True, latency, urls
    except Exception as e:
        latency = time.time() - start
        return False, latency, []

def run_load_test(data: dict, duration: int = 30):
    global latencies, success_count, error_count, request_count
    latencies = []
    success_count = 0
    error_count = 0
    request_count = 0

    start_time = time.time()
    elapsed = 0
    rate = 5

    with ThreadPoolExecutor(max_workers=500) as executor:
        futures = []

        while elapsed < duration:
            now = time.time()
            elapsed = now - start_time

            # Ajustar tasa
            if elapsed >= 20:
                rate = 500
            elif elapsed >= 10:
                rate = 50
            else:
                rate = 5

            # Lanzar 'rate' peticiones este segundo
            for _ in range(rate):
                request_count += 1
                futures.append(executor.submit(single_request, data))

            time.sleep(1)

        # Procesar respuestas
        for i, future in enumerate(as_completed(futures)):
            ok, latency, urls = future.result()
            latencies.append(latency)

            if ok:
                success_count += 1
                print(f"[Req {i}] OK ({latency:.2f}s, {len(urls)} URL(s)).")

                if random.random() < SAVE_PROBABILITY:
                    for idx, url in enumerate(urls):
                        try:
                            save_path = download_and_save(url, data["prompt"], idx)
                            print(f"[Req {i}] Imagen guardada en {save_path}")
                        except Exception as e:
                            print(f"[Req {i}] Error guardando imagen: {e}")
            else:
                error_count += 1
                print(f"[Req {i}] ERROR ({latency:.2f}s).")

    # --- Reporte ---
    total_time = time.time() - start_time
    rps = success_count / total_time if total_time > 0 else 0

    if latencies:
        avg = statistics.mean(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # ~p95
        print("\n📊 Resultados del test de carga:")
        print(f"  Total requests: {request_count}")
        print(f"  Éxitos: {success_count}, Errores: {error_count}")
        print(f"  RPS efectivos: {rps:.2f}")
        print(f"  Latencia promedio: {avg:.2f}s")
        print(f"  Latencia mínima: {min(latencies):.2f}s")
        print(f"  Latencia máxima: {max(latencies):.2f}s")
        print(f"  Latencia p95: {p95:.2f}s")
    else:
        print("\n⚠️ No se midieron latencias (ninguna request exitosa).")

if __name__ == "__main__":
    data = {
        "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
        "num_inference_steps": 30,
        "num_images_per_prompt": 1
    }
    run_load_test(data, duration=20)
