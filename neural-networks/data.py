from csv import reader
import csv


# Load csv file

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Apply one hot encoding to selected columns

def one_hot_encoding(dataset, encoding):

    # Change columns with rows in order to obtain all posible values

    transpose = [[dataset[j][i] for j in range(len(dataset))] for i in
                 range(len(dataset[0]))]

    # Alphabet for each row

    alphabet = list()
    for (i, row) in enumerate(transpose):

        # If row is list, save all posible elements

        if i in encoding:
            values = list()
            for element in row:
                if element not in values and element != '':
                    values.append(element)
            alphabet.append(values)

    new_dataset = list()
    for (i, row) in enumerate(transpose):
        if i in encoding:
            values = next(iter(alphabet), None)

            # Generate a column for each value on alphabet, put 1 if exist, 0 if not

            for value in values:
                column = []
                for element in row:
                    if element == value:
                        column.append('1')
                    else:
                        column.append('0')
                new_dataset.append(column)
            alphabet.pop(0)
        else:
            new_dataset.append(row)

    # Change columns with rows again, in order to return to original matrix with extra columns

    dataset = [[new_dataset[j][i] for j in range(len(new_dataset))]
               for i in range(len(new_dataset[0]))]
    return dataset


# Choose the columns you want to train your data with

def choose_columns(dataset):
    column_ids = []
    labels_col = []
    labels = -1
    first = dataset[0]
    data = list()

    # Select the column to store the identificator for each element and return a result

    ids = int(raw_input('Column with ids (0..n): '))

    # Select the column to store the real values of training dataset

    labels = \
        int(raw_input('Column with labels (0..n) (-1 if no column): '))
    columns = []
    print 'Columns for training - testing, choose (y/n)'
    i = 0

    # For each column on the header, choose if column will be used for training

    for label in first:
        use = raw_input(label + ': ')
        if use == 'y':
            columns.append(i)
        i = i + 1

    # List to store the columns that will be applied the one hot encoding

    encode = []
    i = 0
    j = 0
    print 'Select the columns to apply one hot encoding, choose (y/n)'
    for label in first:

        # If the column is on the list of columns for training

        if i in columns:
            use = raw_input(label + ': ')
            if use == 'y':
                encode.append(j)
            j = j + 1
        i = i + 1
    last = ''

    # Generate the new dataset, with the selected columns

    for (j, row) in enumerate(dataset):
        if j != ids:
            data_row = list()
            for (i, element) in enumerate(row):

                # If index on columns list

                if i in columns:
                    data_row.append(element)

                # Save all the labels to compare later

                if i == int(labels):
                    last = element
                    labels_col.append(int(last))

                # Save all the ids to show results later

                if i == int(ids):
                    column_ids.append(element)

            data.append(data_row)
    data = one_hot_encoding(data, encode)
    return (data, column_ids, labels_col)


# Convert string to float

def str_to_float(dataset, column):
    for row in dataset:
        if row[column] == '':
            row[column] = 0.0
        else:
            row[column] = float(row[column].strip())


# Find the min and max values for each column

def min_max(dataset):
    min_max = list()
    length = len(dataset[0])
    for i in range(length):
        column = [row[i] for row in dataset]
        value_min = min(column)
        value_max = max(column)
        min_max.append([value_min, value_max])
    return min_max


# Normalize data in a scale of 0 to 1

def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1]
                    - minmax[i][0])


def main():

    filename = raw_input('File name of dataset: ')

    file_new = raw_input('File name to put transformed dataset: ')

    # Load file

    dataset = load_csv(filename)

    # Choose the columns to train the model using one hot encoding

    (dataset, column_ids, labels) = choose_columns(dataset)
    length = len(dataset[0])

    # Transform into float values

    for i in range(length):
        str_to_float(dataset, i)

    # Find minimum and maximum value on each column in order to normalize data

    minmax = min_max(dataset)

    # Normalize data using minimum and maximum values

    normalize_dataset(dataset, minmax)

    with open(file_new, 'w') as file:
        writer = csv.writer(file)
        length = len(dataset)
        for i in range(length):
            if labels:
                row = [column_ids[i], labels[i]]
            else:
                row = [column_ids[i]]
            for element in dataset[i]:
                row.append(float(element))
            writer.writerow(row)

    print filename + ' ready'


if __name__ == '__main__':
    main()
