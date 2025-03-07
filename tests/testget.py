import requests

server_url = "http://localhost:8500/api/models"

def list_models():
    url = server_url
    reseponse = requests.get(url=url)
    reseponse.json()
    return reseponse

list_models_api = list_models()
print(list_models)