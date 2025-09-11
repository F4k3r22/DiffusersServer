import os
import re
import urllib.parse
import requests
import time
import statistics
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8500/api/diffusers/inference"
BASE_URL   = "http://localhost:8500"
DOWNLOAD_FOLDER = "imagenes_generadas"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def download_and_save(url: str, prompt: str, idx: int = 0) -> str:
    file_name = os.path.basename(urllib.parse.urlparse(url).path)
    direct_url = f"{BASE_URL}/images/{file_name}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = re.sub(r'[^\w\-_]', '', (prompt or "")[:20].replace(" ", "_"))
    suffix = f"_{idx}" if isinstance(idx, int) else ""
    save_filename = f"{timestamp}_{prompt_slug}{suffix}.png"
    save_path = os.path.join(DOWNLOAD_FOLDER, save_filename)

    resp = requests.get(direct_url, timeout=120)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path

def single_request(data: dict, timeout: int = 480) -> tuple[bool, float, list]:
    start = time.time()
    try:
        resp = requests.post(SERVER_URL, json=data, timeout=timeout)
        resp.raise_for_status()
        latency = time.time() - start
        body = resp.json().get("response", [])
        urls = body if isinstance(body, list) else [body]
        return True, latency, urls
    except Exception as e:
        latency = time.time() - start
        return False, latency, []

def run_n_requests(data: dict, n: int = 10, concurrency: int | None = None, download_all: bool = True, timeout: int = 480):
    if concurrency is None:
        concurrency = min(n, 10)

    latencies = []
    success_count = 0
    error_count = 0
    saved_files = []

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request, data, timeout) for _ in range(n)]

        for i, future in enumerate(as_completed(futures), start=1):
            ok, latency, urls = future.result()
            latencies.append(latency)

            if ok:
                success_count += 1
                print(f"[Req {i}/{n}] OK ({latency:.2f}s) - {len(urls)} URL(s).")
                if download_all:
                    for idx, url in enumerate(urls):
                        try:
                            path = download_and_save(url, data.get("prompt", ""), idx)
                            saved_files.append(path)
                            print(f"    -> Imagen guardada: {path}")
                        except Exception as e:
                            print(f"    -> Error guardando imagen (req {i}): {e}")
            else:
                error_count += 1
                print(f"[Req {i}/{n}] ERROR ({latency:.2f}s).")

    total_time = time.time() - start_all
    rps = success_count / total_time if total_time > 0 else 0

    metrics = {}
    if latencies:
        avg = statistics.mean(latencies)
        mn = min(latencies)
        mx = max(latencies)
        # p95 manual:
        s = sorted(latencies)
        idx95 = max(0, min(len(s)-1, math.ceil(0.95 * len(s)) - 1))
        p95 = s[idx95]
        metrics = {
            "total_requests": n,
            "success": success_count,
            "errors": error_count,
            "total_time_s": total_time,
            "rps": rps,
            "latency_avg_s": avg,
            "latency_min_s": mn,
            "latency_max_s": mx,
            "latency_p95_s": p95,
        }
    else:
        metrics = {
            "total_requests": n,
            "success": success_count,
            "errors": error_count,
            "total_time_s": total_time,
            "rps": rps,
        }


    print("\n=== Resumen del test ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"Imágenes guardadas: {len(saved_files)} (carpeta: {DOWNLOAD_FOLDER})")

    return {"metrics": metrics, "saved_files": saved_files}

if __name__ == "__main__":
    data = {
        "prompt": "The T-800 Terminator Robot Returning From The Future, Anime Style",
        "num_inference_steps": 30,
        "num_images_per_prompt": 1
    }
    resultado = run_n_requests(data, n=200, concurrency=50, download_all=True, timeout=480)