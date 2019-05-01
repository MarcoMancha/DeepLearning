from flask import Flask, request, make_response
from werkzeug.utils import secure_filename
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import flask
import cv2
import os
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True

app = Flask(__name__)
app.config['upload_folder'] = '/home/mm/Documents/deeplearning/dogs-keras/uploads/'

print("----------------------- [LOADING INPUT FILES] -----------------------")
global graph
graph = tf.get_default_graph()
model = load_model("output/dogs.model")
label = pickle.loads(open("output/label.pickle", "rb").read())

@app.route('/', methods=['POST', 'GET'])
def upload():
    data = {}
    if request.method == 'GET':
        data = {'html' : "Function to upload image and classify"}

    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['upload_folder'], filename))
        nombre = os.path.join(app.config['upload_folder'], filename)
        with graph.as_default():
            print("----------------------- [LOADING IMAGE] -----------------------")
            image = cv2.imread(nombre)
            image = cv2.resize(image, (96, 96))
            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            print("----------------------- [PREDICTING LABEL] -----------------------")
            probability = model.predict(image)[0]
            index = np.argmax(probability)
            predicted = label.classes_[index]

            print("----------------------- [PREDICTION] -----------------------")
            prediction = ("I am {:.1f}% sure that is a " + predicted).format(probability[index] * 100)
            data = {"prediction":prediction}

    return flask.jsonify(data)

if __name__ == '__main__':
    app.debug = False
    app.run(host = '0.0.0.0',port=5000)
