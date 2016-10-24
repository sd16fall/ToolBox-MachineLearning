""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

data = load_digits()
#print data.DESCR
num_trials = 10
train_percentages = range(5,95,5)
test_accuracies = numpy.zeros(len(train_percentages))
# train a model with training percentages between 5 and 90 (see train_percentages) and evaluate
# the resultant accuracy.
# You should repeat each training percentage num_trials times to smooth out variability
# for consistency with the previous example use model = LogisticRegression(C=10**-10) for your learner
i = 0
for x in train_percentages: #Loops through the range of values specified in the train_percentages list
	j = 0
	summation = 0
	while j < num_trials: #For each value in train_percentages, the test accuracy is calculated num_trials times and averaged together
		X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = (x/100.0))
		model = LogisticRegression(C=10**-10)
		model.fit(X_train, y_train)
		summation += model.score(X_test,y_test) #Adds the result of the test to summation every time through the loop
		j += 1
	test_accuracies[i] = summation/num_trials #Divides the total by the number of trials and stores it in the ith position in test_accuracies
	i += 1 #increments i every time through the loop

fig = plt.figure()
plt.plot(train_percentages, test_accuracies)
plt.xlabel('Percentage of Data Used for Training')
plt.ylabel('Accuracy on Test Set')
plt.show()
