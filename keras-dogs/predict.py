from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# Input files
flags = argparse.ArgumentParser()
flags.add_argument("--label", required=True, help="path to sabe label file")
flags.add_argument("--model", required=True, help="path to save model for future classifications")
flags.add_argument("--test", required=True, help="path to test image (ex. LABELIMAGENAME.jpg)")
arguments = vars(flags.parse_args())

print("----------------------- [LOADING IMAGE] -----------------------")
image = cv2.imread(arguments["test"])
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

print("----------------------- [LOADING INPUT FILES] -----------------------")
model = load_model(arguments["model"])
label = pickle.loads(open(arguments["label"], "rb").read())

print("----------------------- [PREDICTING LABEL] -----------------------")
probability = model.predict(image)[0]
index = np.argmax(probability)
predicted = label.classes_[index]

print("----------------------- [PREDICTION] -----------------------")
print("\nPREDICTION: " + predicted + " {:.1f}%").format(probability[index] * 100)
