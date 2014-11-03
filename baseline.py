# COMP 598 Project 3
# Baseline


import numpy as np
import csv
import matplotlib.pyplot as plt

from baselineGNB import *
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import normalize



######################################## load ########################################

# Load all train inputs to a python list
train_inputs = np.load('train_inputs.npy')
print "Done loading train_inputs"

print train_inputs.shape      

# Load all train outputs to a python list
train_outputs = np.load('train_outputs.npy')
print "Done loading train_outputs"

print train_outputs.shape

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(train_inputs, train_outputs, test_size=0.20, random_state=0)


# normalize the data
s = preprocessing.Normalizer(norm='l2').fit(train_inputs)

train_inputs_normalized = s.transform(train_inputs)
X_train_normalized = s.transform(X_train)
X_test_normalized = s.transform(X_test)


classifier = GNB(0)
classifier.train(X_train, y_train)
print "Done training"
accuracy, predictions = classifier.score(X_test, y_test)
print "Accuracy is: ", accuracy


# Compute confusion matrix
y_test_pred = classifier.predict(X_test_normalized)
cm = confusion_matrix(y_test_pred, y_test)
cm = cm.astype('float')

# normalize the confusion matrix
Row_Normalized = normalize(cm, norm='l1', axis=1)


# draw the confusion matrix
plt.matshow(Row_Normalized)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()





# ######################################## predict ########################################


# # Load all train inputs to a python list
# test_inputs = np.load('test_inputs.npy')
# print "Done loading test_inputs" 
# print test_inputs.shape 

# test_inputs_normalized = preprocessing.normalize(test_inputs, norm='l2')

# # predict outputs for the test set
# predicted = clf.predict(test_inputs_normalized)
# print "Done predictions"


# ######################################## write ########################################

# # Write a  output for every test_input using the model
# test_output_file = open('test_outputs_normalized_gnb.csv', "wb")
# writer = csv.writer(test_output_file, delimiter=',') 
# writer.writerow(['Id', 'Prediction']) # write header
# for idx, test_input in enumerate(test_inputs):
#     predict_int = predicted[idx]
#     row = [idx+1, predict_int]
#     writer.writerow(row)
# test_output_file.close()

# print "Done writing test outputs"






