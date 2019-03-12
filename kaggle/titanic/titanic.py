from random import seed
from random import randrange
from csv import reader
from math import exp
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

#Apply one hot encoding to selected columns
def one_hot_encoding(dataset, encoding):
	#Change columns with rows in order to obtain all posible values
	transpose = [[dataset[j][i] for j in range(len(dataset))] for i in range(len(dataset[0]))]
	#Alphabet for each row
	alphabet = list()
	for i,row in enumerate(transpose):
		#If row is list, save all posible elements
		if i in encoding:
			values = list()
			for element in row:
				if element not in values:
					values.append(element)
			alphabet.append(values)
	new_dataset = list()
	for i,row in enumerate(transpose):
		if i in encoding:
			values = next(iter(alphabet), None)
			#Generate a column for each value on alphabet, put 1 if exist, 0 if not
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
	#Change columns with rows again, in order to return to original matrix with extra columns
	dataset = [[new_dataset[j][i] for j in range(len(new_dataset))] for i in range(len(new_dataset[0]))]
	return dataset

#Choose the columns you want to train your data with
def choose_columns(dataset):
	column_ids = []
	labels_col = []
	labels = -1
	first = dataset[0]
	data = list()
	#Select the column to store the identificator for each element and return a result
	ids = int(raw_input("Column with ids (0..n): "))
	#Select the column to store the real values of training dataset
	labels = int(raw_input("Column with labels (0..n) (-1 if no column): "))
	columns = []
	print("Columns for training - testing, choose (y/n)")
	i = 0
	#For each column on the header, choose if column will be used for training
	for label in first:
		use = raw_input(label + ": ")
		if use == "y":
			columns.append(i)
		i = i + 1
	#List to store the columns that will be applied the one hot encoding
	encode = []
	i = 0
	j = 0
	print("Select the columns to apply one hot encoding, choose (y/n)")
	for label in first:
		#If the column is on the list of columns for training
		if i in columns:
			use = raw_input(label + ": ")
			if use == "y":
				encode.append(j)
			j = j + 1
		i = i + 1
	last = ""
	#Generate the new dataset, with the selected columns
	for j,row in enumerate(dataset):
		if j != ids:
			data_row = list()
			for i, element in enumerate(row):
				#If index on columns list
				if i in columns:
					data_row.append(element)
				#Save all the labels to compare later
				if i == int(labels):
					last = element
					labels_col.append(int(last))
				#Save all the ids to show results later
				if i == int(ids):
					column_ids.append(element)
			#If labels column exists
			if labels != -1:
				data_row.append(last)
			data.append(data_row)
	data = one_hot_encoding(data,encode)
	return data, column_ids, labels_col

# Convert string to float
def str_to_float(dataset, column):
	for row in dataset:
		if row[column] == "":
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
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	length = len(dataset)
	#Integer Fold size, size will be missing the last N elements
	fold_size = int(length / n_folds)
	#Calculate number of missing elements to add to the last fold
	missing = length - (fold_size * n_folds)
	for i in range(n_folds):
		fold = list()
		#Last fold
		if i == n_folds - 1:
			#Append the last N elements
			while len(fold) < fold_size + missing:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
		else:
			while len(fold) < fold_size:
				index = randrange(len(dataset_copy))
				fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

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

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm_folds(dataset, algorithm, n_folds, l_rate, n_epoch):
	#Divide folds for training
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	total_p = []
	for fold in folds:
		# N-1 training folds
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		# 1 fold for training
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		#Predict values and get coefficients
		predicted, coef = algorithm(train_set, test_set, l_rate, n_epoch)
		total_p = total_p + predicted
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores, total_p


# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	#Multiply each coefficient with each value of the row
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	#Apply logistic regression formula
	return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			#First coefficient
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			#Every other coefficient, depending on the size of the row
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return predictions, coef

def train(filename):
	# Values for training
	n_folds = 10
	l_rate = 0.01
	n_epoch = 1000

	#Load file
	dataset = load_csv(filename)

	#Choose the columns to train the model using one hot encoding
	dataset, column_ids, labels = choose_columns(dataset)
	length = len(dataset[0])

	# Transform into float values
	for i in range(length):
		str_to_float(dataset, i)

	# Find minimum and maximum value on each column in order to normalize data
	minmax = min_max(dataset)

	# Normalize data using minimum and maximum values
	normalize_dataset(dataset, minmax)

	#Obtain scores and predicted training values
	scores, predicted = evaluate_algorithm_folds(dataset, logistic_regression, n_folds, l_rate, n_epoch)

	print('Train scores using cross-validation: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

	coef = []
	column_ids = []

	# Ask file names for testing the coefficients
	filename = raw_input("File name of test dataset: ")
	results = raw_input("File name to put results of test: ")
	# Test model on unseen data
	test(dataset,filename,results,coef,labels)

def test(train,filename,results,coef,labels):

	l_rate = 0.01
	n_epoch = 1000

	#Load file
	test = load_csv(filename)

	#Choose the columns to test the model using one hot encoding
	test, column_ids, not_labels = choose_columns(test)
	length = len(test[0])

	# Transform into float values
	for i in range(length):
		str_to_float(test, i)

	# Find minimum and maximum value on each column in order to normalize data
	minmax = min_max(test)

	# Normalize data using minimum and maximum values
	normalize_dataset(test, minmax)

	if not coef:
		# Train model with the whole dataset and obtain coefficients
		train_predicted = []
		coef = coefficients_sgd(train, l_rate, n_epoch)

		# Predict values for training data
		for row in train:
			yhat = predict(row, coef)
			yhat = round(yhat)
			train_predicted.append(yhat)

		accuracy = accuracy_metric(labels, train_predicted)
		print('Train score using whole dataset: %.3f%%' % accuracy)

	# Predict values for test data
	test_predicted = []
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		test_predicted.append(yhat)

	#Write results using id's column
	with open(results, 'w') as file:
		writer = csv.writer(file)
		length = len(test_predicted)
		for i in range(length):
			y = int(column_ids[i])
			x = int(test_predicted[i])
			row = [y,x]
			writer.writerow(row)

def main():
    # Seed in order to start with the same numbers when using random
	seed(1)
	print("1. Testing with training coefficients")
	print("2. Testing with existing coefficients")
	option = int(raw_input())
	# Calculate coefficients and test using training file
	if option == 1:
		# Read filename ex.'train_mod.csv'
		filename = raw_input("File name of train dataset: ")
		train(filename)
	# Test with new data with existing coefficients
	else:
		#Coefficients using pclass, sex, age, sibsp, parch, cabin and embarked
		# Sex, cabin and embarked using one hot encoding
		coef=[
		1.0021798418352148,-1.806516571847624,-0.8562062405993144,1.8583860824344995,
		-1.0900134203624139,-1.668616679963269,-0.4990674915954358,0.1402318063844269,
		-0.49149528280850674,0.05421043260508719,-0.5260029354201596,-0.6301961980824836,
		-1.128917060415986,0.11080575853176691,1.3823493322411158,0.99319793385285,
		-0.4159081503865266,0.030510488327015715,1.0255809251229506,-0.7076786732702475,
		1.3595270397639434,0.16361686913758655,-0.46527588937834496,0.5188885900066735,
		-0.2694196409649548,-0.5842075018749662,-0.7065300846621648,0.799789620189117,
		-1.0797687009760386,-0.6579608621694224,-0.8832285801925486,0.4347799646507854,
		0.3316257047532618,0.07089064723418508,-1.0058489704447076,1.49129156915597,
		-0.6030280060480395,0.11271990360236607,-0.5796837543422574,-0.7976886611215674,
		-0.8469684302860353,1.478324452102963,-0.9474871833949388,0.04948868223816828,
		0.058133363020908636,0.8738556532917257,0.08419517574276811,0.04068741775113203,
		1.1668012505051808,-0.6419393266789012,1.211325689753156,-0.5764417235875058,
		0.13386875466340045,-0.40000674169024086,-0.7057163110583374,0.18720980979016227,
		0.07530542626909234,-0.8829726238343806,0.14863485191293066,-0.9483456951248199,
		0.8625217203972303,-0.3525535442469912,-2.501914903218151,0.8235716684032026,
		-0.9634237010878757,0.03741322221049875,0.0,0.11057263414056735,0.08489233849704506,
		0.06395792003887325,0.039858060889189216,0.0836131264601983,-0.9522940961018641,
		-0.689804724649039,0.0,-0.682424773992999,-0.956429097012995,0.07978674407364598,
		0.06811998167440833,0.8703113991349076,-0.9274626293682896,1.7114725154603987,
		1.2821640057063477,-0.45513181626405724,0.9147490623008178,1.1436862962538803,
		-0.9660606389637755,0.029919094994370975,-0.5571702391306803,0.05466519072554572,
		1.1177694289867004,-0.6573324359441786,-0.9588422812533847,-0.779760530715711,
		-0.6141402285577052,0.13232347904746214,0.05627222329866941,1.5656891499485672,
		-0.6431293010325146,0.0698878864154711,-0.974809349529896,-0.6585953523944078,
		0.0,-0.32781857117305496,-0.7657783130320479,0.7993100691992827,0.06446181776887874,
		0.15276505036465018,0.06193749556329033,-0.9184053187538518,0.06964863962145128,
		1.121531198872456,1.0203017710821227,1.1728206694234782,-0.5821666663392633,
		0.0627096545070257,1.2842672900570236,0.8393192811124104,0.9731808732143234,
		-0.7164676240433868,-0.664805732506599,1.2147746096681946,-0.675087349600845,
		0.07431884843687751,0.7928412181999182,0.1185495932775027,0.0734479652195583,
		-0.1076111554243771,0.0353687998081743,1.5777140463478614,0.03547159026745801,
		0.04175768001908377,1.0910883746214168,0.8053846515206128,0.8252203074398244,
		-0.6570925759575033,-0.7578452178628872,1.6941678862695748,0.12060775747755684,
		-1.383394256965522,-0.3116091469086709,0.0,-0.769167608659708,-0.8771114854131796,
		0.18043817954256705,-0.703802793213247,0.0,0.1316877610873389,0.06026932454417265,
		0.6391656324250118,0.06380252300171874,1.1297895634601949,-0.765285982823842,0.0,
		0.05829151654463251,0.7823821769518484,0.03334283282764744,0.39845560742612196,
		0.40676453244383515,0.16361686913758655]
		filename = raw_input("File name of test dataset: ")
		results = raw_input("File name to put results of test: ")
		t_d = []
		labels = []
		test(t_d,filename,results,coef,labels)

if __name__ == "__main__":
    main()

