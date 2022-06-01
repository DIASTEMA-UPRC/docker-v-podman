import os
import numpy as np
import tensorflow as tf

from flask import Flask, request
from PIL import Image

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))

model = tf.keras.models.load_model("models/model.h5")

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():
    image = Image.open(request.files["image"].stream)
    image = image.resize((224, 224))
    x = np.asarray(image)
    x = np.reshape(x, (224, 224, 3))
    x = np.expand_dims(x, axis=0)

    y = model.predict(x)

    return f"{np.argmax(y[0])}", 200


if __name__ == "__main__":
    app.run(HOST, PORT, True)
