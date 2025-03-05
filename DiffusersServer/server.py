from flask import Flask, request, jsonify
from flask_cors import CORS
from .Pipelines import *
import random
import uuid
import tempfile
from dataclasses import dataclass

service_url = 'http://localhost:8500'

image_dir = os.path.join(tempfile.gettempdir(), "images")
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

def save_image(image):
    filename = "draw" + str(uuid.uuid4()).split("-")[0] + ".png"
    image_path = os.path.join(image_dir, filename)
    logger.info(f"Saving image to {image_path}")
    image.save(image_path)
    return os.path.join(service_url, "images", filename)

@dataclass
class ServerConfigModels:
    model: str = None

def create_app(config=None):
    app = Flask(__name__)
    CORS(app)
    app.config['SERVER_CONFIG'] = config or ServerConfigModels()
    # Carga del modelo en memoria al iniciar la aplicación
    model_pipeline = TextToImagePipelineSD3()
    model_pipeline.start()
    app.config["MODEL_PIPELINE"] = model_pipeline

    @app.route('/api/inference', methods=['POST'])
    def api():
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Recuperamos el pipeline cargado previamente
        server_config = app.config['SERVER_CONFIG']
        model_pipeline = server_config.model if server_config.model is not None else data['MODEL_PIPELINE']

        prompt = data.get("prompt")
        # Si tu modelo o scheduler no son thread-safe, es recomendable clonar el scheduler
        scheduler = model_pipeline.pipeline.scheduler.from_config(model_pipeline.pipeline.scheduler.config)
        pipeline = StableDiffusion3Pipeline.from_pipe(model_pipeline.pipeline, scheduler=scheduler)
        
        generator = torch.Generator(device=model_pipeline.device)
        generator.manual_seed(random.randint(0, 10000000))
        output = pipeline(prompt, generator=generator)
        
        # Aquí guardarías la imagen y devolverías la respuesta
        image_url = save_image(output.images[0])
        return jsonify({'response': image_url})
    
    return app
