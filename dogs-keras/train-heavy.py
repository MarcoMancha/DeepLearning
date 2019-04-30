from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
from imutils import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# File names  to generate output files passed as arguments
flags = argparse.ArgumentParser()
flags.add_argument("--dataset", required=True, help="path to carpet with images")
flags.add_argument("--label", required=True, help="path to sabe label file")
flags.add_argument("--model", required=True, help="path to save model for future classifications")
flags.add_argument("--plot", type=str, default="accuracy_plot.png", help="path to save image of accuracy and lost plot")
arguments = vars(flags.parse_args())

# Global variables
EPOCHS = 100
LR = 1e-3
BS = 32

IMAGE_DIMS = (96,96,3)
width = IMAGE_DIMS[1]
height = IMAGE_DIMS[0]
depth = IMAGE_DIMS[2]

data = []
labels = []

print("----------------------- [LOADING PATHS] -----------------------")
img_paths = list(paths.list_images(arguments["dataset"]))
random.seed(1)
# shuffle image paths to use them on a different order
random.shuffle(img_paths)

print("----------------------- [LOADING IMAGES] -----------------------")

for path in img_paths:
	# Try to load image if it is not corrupted
	try:
		# load image, resize it to default dimensions and append to data set
		file = cv2.imread(path)
		file = cv2.resize(file, (IMAGE_DIMS[0], IMAGE_DIMS[1]))
		file = img_to_array(file)
		data.append(file)
		# retrieve label from image path and append to labels list
		label = path.split(os.path.sep)[-2]
		labels.append(label)
	except Exception as e:
		continue

print("----------------------- [DATA NORMALIZATION] -----------------------")
# Normalize data set value into 0 - 1 range
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("Size of dataset: {:.2f} MB".format(data.nbytes / (1024000)))

print("----------------------- [BINARIZE LABELS] -----------------------")

label = LabelBinarizer()
labels = label.fit_transform(labels)
classes = len(label.classes_)

print("----------------------- [SPLITING DATASET] -----------------------")
# Partiton data into 80% train data and 20% test data to test with unseen data later
# Partiton data into 80% train data and 20% validation data to see how our model behaves

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=1)
(trainX, valX, trainY, valY) = train_test_split(trainX, trainY, test_size=0.2, random_state=1)

print("----------------------- [DATA AUGMENTATION] -----------------------")
# Image data generator for using data augmentation and obtain more images from dataset

aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

print("----------------------- [INITIALIZE MODEL] -----------------------")

model = Sequential()

# shape depending on keras backend
if K.image_data_format() == "channels_first":
	images_shape = (depth, height, width)
	channel_dim = 1
else:
	images_shape = (height, width, depth)
	channel_dim = -1

print("----------------------- [CONSTRUCT CNN - MODEL] -----------------------")

# Simpler VGG structure for Image Processing
model.add(Conv2D(32, (3, 3), padding="same", input_shape=images_shape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(256, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(512, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(4096))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(classes))
model.add(Activation("softmax"))
opt = Adam(lr=LR, decay=LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics = ["accuracy"])

print("----------------------- [TRAINING] -----------------------")

# Model fit generator to train the model already compiled using data augmentation

dogs_model = model.fit_generator( aug.flow(trainX, trainY, batch_size=BS), validation_data=(valX, valY), steps_per_epoch=len(trainX) // BS, epochs=EPOCHS, verbose=1)

# dogs_model = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1, validation_data=(testX, testY))

print("----------------------- [SAVING MODEL] -----------------------")

# Output the model results to file
model.save(arguments["model"])

print("----------------------- [SAVING LABELS] -----------------------")

# Output label binarizer to file
f = open(arguments["label"], "wb")
f.write(pickle.dumps(label))
f.close()

print("----------------------- [TESTING] -----------------------")
model = load_model(arguments["model"])
lb = pickle.loads(open(arguments["label"], "rb").read())
n = len(testX)
count = 0.0
testY = label.inverse_transform(testY)
for i in range(n):
	image = np.expand_dims(testX[i], axis=0)
	proba = model.predict(image)[0]
	idx = np.argmax(proba)
	label = lb.classes_[idx]
	if label == testY[i]:
		count = count + 1.0

print("TESTING ACCURACY: " + str((count / n) * 100.0) + "%")

print("----------------------- [PLOT ACCURACY - LOSS] -----------------------")

# Plot accuracy and loss on training and validation
plt.style.use("dark_background")
plt.figure()
plt.plot(np.arange(0, EPOCHS), dogs_model.history["loss"], label="Training loss")
plt.plot(np.arange(0, EPOCHS), dogs_model.history["val_loss"], label="Validation loss")
plt.plot(np.arange(0, EPOCHS), dogs_model.history["acc"], label="Training accuracy")
plt.plot(np.arange(0, EPOCHS), dogs_model.history["val_acc"], label="Validation accuracy")
plt.title("Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy - Loss")
plt.legend(loc="upper left")
plt.savefig(arguments["plot"])
