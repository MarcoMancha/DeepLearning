import numpy as np
from numpy import genfromtxt
from csv import reader
import csv


# Calculate accuracy percentage

def accuracy_metric(actual, predicted):
    correct = 0
    id = 0
    dataset = list()
    length = len(actual)
    for i in range(length):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Apply logistic regression foorumula to obtain the predicted value using function

def logistic(x):
    return 1.0 / (1 + np.exp(-x))


# Derivative of logistic function, the partial derivative

def logistic_derivative(x):
    return x * (1.0 - x)


class NeuralNetwork:

    # Neural network of 3 layers (without input layer)
    # Constructor, receive numpy array of inputs and of labels
    # Input n*n - Output n*1

    # Example:
    # Input layer: N nodes
    # Hidden layer 1: 3 nodes - N weights per node
    # Hidden layer 2: 3 nodes - 3 weights per node
    # Output layer: 1 node - 3 weights

    def __init__(self, inputs, labels):
        self.input = inputs
        self.labels = labels
        self.output = np.zeros(self.labels.shape)
        self.weights_hidden1 = np.random.rand(self.input.shape[1], 3)
        self.weights_hidden2 = np.random.rand(3, 3)
        self.weights_output = np.random.rand(3, 1)

    # Feed forward for calculating output values for each row

    def feedForward(self):

        # For each layer  -> multiply input values with their weights to obtain next layer

        self.hiddenLayer1 = logistic(np.dot(self.input,
                self.weights_hidden1))
        self.hiddenLayer2 = logistic(np.dot(self.hiddenLayer1,
                self.weights_hidden2))
        self.output = logistic(np.dot(self.hiddenLayer2,
                               self.weights_output))

    def backPropagation(self):

        # Learning and regularization rate

        learning = 0.1
        regularization = 1

        # Number of input samples

        samples = self.labels.shape[0]

        # ------------- OUTPUT LAYER ---------------
        # Total error

        error = self.labels - self.output

        # Deltas = Total error * x * (1 - x) where x is a node

        deltas_output = error * logistic_derivative(self.output)

        # Gradients = Deltas * hidden layer 2 values

        gradients_output = np.dot(self.hiddenLayer2.T, deltas_output)

        # Calculate new weight using learning and regularization rate

        new_weights_output = (1 - learning * (regularization
                              / samples)) * self.weights_output \
            + learning * gradients_output

        # ------------- HIDDEN LAYER 2 ---------------

        # Deltas = (Deltas_output * weight_output) * (1 - x) where x are values of hidden layer 2

        deltas_hidden = np.dot(deltas_output, self.weights_output.T) \
            * logistic_derivative(self.hiddenLayer2)

        # Gradients = Deltas * hidden layer 1 values

        gradients_hidden = np.dot(self.hiddenLayer1.T, deltas_hidden)

        # Calculate new weight using learning and regularization rate

        new_weights_hidden2 = (1 - learning * (regularization
                               / samples)) * self.weights_hidden2 \
            + learning * gradients_hidden

        # ------------- HIDDEN LAYER 1 ---------------

        # Deltas = (Deltas_output * weight_output) * (1 - x) where x are values of hidden layer 1

        deltas_hidden = np.dot(deltas_output, self.weights_output.T) \
            * logistic_derivative(self.hiddenLayer1)

        # Gradients = Deltas * input layer values

        gradients_hidden = np.dot(self.input.T, deltas_hidden)

        # Calculate new weight using learning and regularization rate

        new_weights_hidden1 = (1 - learning * (regularization
                               / samples)) * self.weights_hidden1 \
            + learning * gradients_hidden

        # Update weights

        self.weights_hidden1 = new_weights_hidden1
        self.weights_hidden2 = new_weights_hidden2
        self.weights_output = new_weights_output


def main():

    # Load file from csv

    filename = raw_input('File name of dataset: ')
    x = genfromtxt(filename, delimiter=',')

    # Separate id's, labels and data columns from titanic dataset

    y = x[:, 1]
    ids = x[:, 0]
    y = y.tolist()
    y = [[i] for i in y]
    y = np.array(y)

    # Remove id's and labels column for training

    x = x[:, 2:]

    for row in x:
        row = list(row)

    # Create neural network from labels and data

    network = NeuralNetwork(x, y)
    for i in range(2000):
        network.feedForward()
        network.backPropagation()

    # Generate array with actual values on labels column

    length = y.shape[0]
    y = y.tolist()
    actual = list()
    predicted = list()
    for i in range(length):
        actual.append(int(y[i][0]))

    # Round values to have discrete values

    network.output = np.round(network.output)

    # Generate array with predicted values

    results = network.output.tolist()
    length = network.output.shape[0]
    for i in range(length):
        predicted.append(int(results[i][0]))

    # Calculate accuracy score for training

    accuracy = accuracy_metric(actual, predicted)
    print 'Accuracy: ' + str(accuracy)

    # Load csv test file and separate columns

    test = raw_input('File name of testing dataset: ')
    x = genfromtxt(test, delimiter=',')
    ids = x[:, 0]
    x = x[:, 1:]
    for row in x:
        row = list(row)

    # Change input from network to calculate new output using existing weight's

    # Feed forward with new input data

    network.input = x
    network.feedForward()

    # Round for discrete values

    network.output = np.round(network.output)
    output = network.output.tolist()

    # Write results using id's column

    with open('results.csv', 'w') as file:
        writer = csv.writer(file)
        length = len(output)
        for i in range(length):
            y = int(ids[i])
            x = int(output[i][0])
            row = [y, x]
            writer.writerow(row)


if __name__ == '__main__':
    main()
