import json
import urllib.request

from flask import Flask
from flask import request
from fastai.vision import *

app = Flask(__name__)


@app.route('/')
def handler():
    defaults.device = torch.device('cpu')

    path = Path('.')
    learner = load_learner(path, 'model.pkl')

    image = request.args['image']
    urllib.request.urlretrieve(image, './images/image.jpg')
    img = open_image('./images/image.jpg')
    pred_class, pred_idx, outputs = learner.predict(img)

    return json.dumps({
        "predictions": sorted(
            zip(learner.data.classes, map(float, outputs)),
            key=lambda p: p[1],
            reverse=True
        )
    })