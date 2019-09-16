from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.config import Config
import uvicorn
import os
from fastai import *
from fastai.vision import *
import urllib
import shutil
import requests
import binascii

app = Starlette(debug=True)

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])

### EDIT CODE BELOW ###

answer_question_1 = """ 
How can you detect overfitting or underfitting?

Overfitting can be observed if the model does very good on the training set but poorly on the testing set. In this case we want it to generalize more, which in the case of neural networks could be achieved by adding dropout layers or using a less complex architecture.
Underfitting means that we generalize too much, achieving a low accuracy rate. In this case we want to use more advanced models or obtain more data.
"""

answer_question_2 = """ 
Describe how gradient decent works.

Gradient descent is a method to find a (local) minimum of a function. We move incrementally in small steps into the direction of the steepest descent until no more improvements can be made. The step-length is important, in order to not overshoot the target over and over again. This can be used for neural networks in order to adjust their weights. 
"""

answer_question_3 = """ 
What is the goal of regression?

Regression is used to predict continuous data, for example changes in temperature or time series, where one dependent variable is modeled in terms of one or more independent ones. Given some data points we want to find a function that approximates/fits the data the best, minimizing error.
"""

## Replace none with your model
pred_model = 'export.pkl' 
learn = load_learner(".") # load pkl

def get_bytes(url):
    result = None
    if url.startswith("data:"):
        result = binascii.a2b_base64(url.partition('base64,')[2])
    else:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            result = r.content
        else:
            print("Could not load url")
    return result


@app.route("/api/answers_to_hw", methods=["GET"])
async def answers_to_hw(request):
    return JSONResponse([answer_question_1, answer_question_2, answer_question_3])

@app.route("/api/class_list", methods=["GET"])
async def class_list(request):
    return JSONResponse(['hygrophoropsis_aurantiaca','cantharellus_cibarius','paxillus_involutus','jack_o_lantern'])

@app.route("/api/classify", methods=["POST"])
async def classify_url(request):
    body = await request.json()
    url_to_predict = body["url"]

    bytes = get_bytes(url_to_predict)
    img = open_image(BytesIO(bytes))
    cat, tensor, losses = learn.predict(img)

    return JSONResponse({
        "predictions": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: round(p[1],2),
            reverse=True
        )
    })

### EDIT CODE ABOVE ###

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['PORT']))

