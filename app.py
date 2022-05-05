import os
import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import Image

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

model = tf.keras.models.load_model("models/model.h5")

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return "Welcome to the API", 200


@app.route("/predict", methods=["POST"])
def predict():
    image = Image.open(request.files["image"].stream)
    image = image.resize((224, 224))
    imgarr = np.asarray(image)
    imgarr = np.reshape(imgarr, (224, 224, 3))
    imgarr = np.expand_dims(imgarr, axis=0)

    x = model.predict(imgarr)

    return f"{np.argmax(x[0])}"


if __name__ == "__main__":
    app.run(HOST, PORT, True)
