# Find the average and standard deviation of each column
def average_std(dataset):
	avg_std = list()
	for row in dataset:
		average = 0
		n = 0
		for element in row:
			average = average + element
			n = n + 1
		average = average / n
		v = 0
		for element in row:
			v = v + pow(average - element, 2)
		v = v / (n-1)
	  	std = pow(v,0.5)
		avg_std.append([average, std])
	return avg_std

# Standarize data using z-score
def standarize_dataset(dataset, avg_std):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - avg_std[i][0]) / avg_std[i][1]
