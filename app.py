from flask import Flask, request, send_file
from carvekit.api.high import HiInterface
from carvekit.ml.wrap.u2net import U2NET
from carvekit.ml.wrap.bgm import BGM
from carvekit.ml.wrap.fba import FBA
from carvekit.ml.wrap.tracer_b7 import TracerB7
from carvekit.ml.wrap.deeplab import DeepLabV3
from carvekit.ml.wrap.basnet import BASNET
from carvekit.pipelines.preprocessing import Preprocessing
from carvekit.pipelines.postprocessing import Postprocessing
from carvekit.pipelines.pipeline import Pipeline
import io
from PIL import Image
import requests

app = Flask(__name__)

# Configuration des modèles et des pipelines
preprocessing = Preprocessing()
postprocessing = Postprocessing()
model = U2NET(pretrained=True)  # Utilisation du modèle U2NET
interface = HiInterface(preprocessing, postprocessing, model)

@app.route('/remove-background', methods=['POST'])
def remove_background():
    image_url = request.json.get('image_url')
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content))
    result = interface([image])[0]
    img_io = io.BytesIO()
    result.save(img_io, 'PNG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
