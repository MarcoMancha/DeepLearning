# Predicting Dog Breeds

Keras application to predict a dogs breed.

## Tools

Python version 2.7.1
Keras
Tensorflow
Cuda
cuDNN

Other requirements:
* imutils
* opencv-python
* scipy
* sklearn
* matplotlib

## Training

### train.py

Script used to train the deep neural network using CNN networks and deep learning techniques.

The script needs the following flags:

#### Flags

--dataset -> path with images divided by it's class

--label   -> where you want to store label's file

--model   -> where you want to store the model's file for further predictions

--plot    -> where you want to store the plot image with loss/accuracy (optional)

### Example
`$ python train.py --dataset dataset --label label.pickle --model dogs.model`

## Predicting

### predict.py

Script used to predict a dogs breed.

The script needs the following flags:

#### Flags

--label   -> where the label's file is

--model   -> where the model's file is

--test    -> path of the image to predict breed

### Example
`$ python predict.py --label output/label.pickle --model output/dogs.model --test test/chihuahua.jpg`

### Running the scripts

* Clone the repo and search for the titanic.py under neural_networks/ carpet.
* Run the script with the needed flags


