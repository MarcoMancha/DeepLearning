# Titanic

Titanic problem using neural networks.

## Tools

Python version 2.7.1

## Scripts

### data.py

Script used to transform and alter our .csv file, in order to choose columns, apply one hot encoding and normalize dataset.

Input -> .csv file

Output -> modified .csv file

### Running the script

* Clone the repo and search for the titanic.py under neural_networks/ carpet.

* Run the script: `python data.py`

Input files and output files must be **.csv** files

### neural.py

Script used to train our model using neural networks and logistic regression as the activation function.

#### Training

Input -> .csv file

Output -> accuracy scores

#### Testing

Input -> .csv file

Output -> predictions on results.csv

### Running the script

* Clone the repo and search for the titanic.py under neural_networks/ carpet.

* Run the script: `python neural.py`

Input files and output files must be **.csv** files

### Example
`$ python neural.py`<br>
File name of dataset: mod_datasets/train.csv<br>
Accuracy: 81.3692480359<br>
File name of testing dataset: mod_datasets/test.csv<br>



