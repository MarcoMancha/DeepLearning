# Titanic

Titanic problem using python and logistic regression.

## Machine Learning Techniques

One hot encoding - to use non numeric data.

Minimum and maximum - normalize data

Cross-validation - to check if the model is overfitting and to estimate the skill of the model on new data.

Stochastic gradient descent - to optimize the cumulative error on our model.

## Tools

Python version 2.7.1

## Modes

### Training with training coefficients

On this mode, the purpose is to train the model from scratch:

#### Training

* Select training data
* Choose the columns to work with
* Apply one hot encoding to columns (optional)
* Cross validation on dataset
* Gradient descent with whole dataset and generate coefficients
* Predict values using generated coefficients

#### Testing

* Select test data
* Choose the columns to work with (same columns as training)
* Apply one hot encoding to columns (optional)
* Use existing coefficients and predict values
* Write predictions on output file

### Training with existing coefficients

#### Training

No training is made on this option, coefficients are gathered from the script.

#### Testing

* Coefficients are loaded from file
* Select test data
* Choose the columns to work with (same columns as training)
* Apply one hot encoding to columns (optional)
* Use existing coefficients and predict values
* Write predictions on output file

## Running the script

* Clone the repo and search for the titanic.py under kaggle/titanic carpet.

* Run the script: `python titanic.py`

Input files and output files must be **.csv** files

### Example
![running script](https://github.com/MarcoMancha/deep_learning/blob/2e41cdb3da8804a6dcae2576a147cd4dd807fe5e/kaggle/titanic/example_run.png)


