# COMP 598 Project 3
# Baseline: Gaussian Navie Bayes


import numpy as np
import scipy.stats

class GNB:

   
    def __init__(self, covar):
        # Should initialized it to zero in this project since we assume covariance is different across all i and j
        self.covar = covar
    
        
    def train(self, data, labels):
        """
        Given a dataset and corresponding labels
        Learn all necessary parameters in GNB
        """
        
        # total number of training examples
        self.num_data = len(data) 

        # number of features
        self.num_features = len(data[0]) 

        #---------- Calculate the prior probability P(Y = y) for labels y = 0 ... 9 ----------#
        # Initialize an array of total times each label occurs in the dataset
        self.array_sum = np.zeros(10, dtype = np.float) 

        # traverse all labels, y can be 0 ... 9
        for y in labels: 
            self.array_sum[y] += 1

        # divide to get probability each label occurs in the dataset
        self.array_prior = self.array_sum / self.num_data 

        print self.array_prior

        #---------- Calculate the normal mean mu_ij ----------#
        # initialize an 10 * num_features matrix to represent mu_ij
        self.matrix_mean = np.zeros((10, self.num_features), dtype = np.float)
        
        for j in range(self.num_features):
            for i in range(self.num_data):
                y = labels[i]
                self.matrix_mean[y][j] += data[i][j]       

        for y in range(10):                
            self.matrix_mean[y] =  self.matrix_mean[y]  / self.array_sum[y] 

        print self.matrix_mean                


        #---------- Calculate the variance sigma_ij ----------#

        #---------- Case 0: variance is different for all i and j ----------#
        if self.covar == 0:
            # initialize an 10 * num_features matrix to represent mu_ij
            self.matrix_variance = np.zeros((10, self.num_features), dtype = np.float)
            
            for j in range(self.num_features):
                for i in range(self.num_data):
                    y = labels[i]
                    self.matrix_variance[y][j] += (data[i][j] - self.matrix_mean[y][j])**2

            for y in range(10):                
                self.matrix_variance[y] =  self.matrix_variance[y] / self.array_sum[y]  

            print self.matrix_variance 

        #---------- Case 1: variance is different for i (classes), but same for all j (features) ----------#
        if self.covar == 1:
            # initialize an 10 * num_features matrix to represent mu_ij
            self.matrix_variance = np.zeros(10, dtype = np.float)
            
           
            for i in range(self.num_data):
                y = labels[i]
                mean = np.sum(data[i]) / self.num_features
                self.matrix_variance[y] += (mean - self.matrix_mean[y][j])**2

            for y in range(10):                
                self.matrix_variance[y] =  self.matrix_variance[y] / self.array_sum[y]  

            print self.matrix_variance 


    def score(self, data, labels):
        """
        Given a dataset which only has data (without labels)
        Apply Gaussian Naive Bayes method
        Return a list of lables as predications and accuracy
        """

        # initialize a list as prediction errors for test set
        prediction_errors = np.zeros(len(data), int)

        predicted_labels = np.zeros(len(data), int)

        # initialize an array of size 10 to represent probability of each labels for a given input
        array_predictions = np.zeros(10)
        
        for i, x in enumerate(data):   
            for y in range(10):  
                # Get p(y) for class j
                array_predictions[y] = np.log(self.array_prior[y])

                #---------- Case 0: variance is different for all i and j ----------#
                if self.covar == 0:   
                    for j in range(self.num_features):
                        # sum logs                    
                        array_predictions[y] += np.log(scipy.stats.norm(self.matrix_mean[y][j], np.sqrt(self.matrix_variance[y][j])).pdf(x[j]))
                
                #---------- Case 1: variance is same all j ----------#
                if self.covar == 1:
                    for j in range(self.num_features):
                        # sum logs                   
                        array_predictions[y] += np.log(scipy.stats.norm(self.matrix_mean[y][j], np.sqrt(self.matrix_variance[y])).pdf(x[j]))
                
            predicted_labels[i] = np.argmax(array_predictions)  # Find the highest probability
            print i
            if(predicted_labels[i] != labels[i]):
                prediction_errors[i] = 1     
                
        # Compute accuracy
        accuracy = 1.0 - (np.sum(prediction_errors) * 1.0 / len(prediction_errors))    

            
        # return predicated lables for the test set, and the accuracy
        return accuracy, predicted_labels
    


    def predict(self, data):
        """
        Given a dataset which only has data (without labels)
        Apply Gaussian Naive Bayes method
        Return a list of lables as predications for each input in the dataset.
        """


        # initialize a list as prediction errors for test set
        prediction_errors = np.zeros(len(data), int)

        predicted_labels = np.zeros(len(data), int)

        # initialize an array of size 10 to represent probability of each labels for a given input
        array_predictions = np.zeros(10)
        
        for i, x in enumerate(data):   
            for y in range(10):  
                # Get p(y) for class j
                array_predictions[y] = np.log(self.array_prior[y])

                #---------- Case 0: variance is different for all i and j ----------#
                if self.covar == 0:   
                    for j in range(self.num_features):
                        # Sum logs                  
                        array_predictions[y] += np.log(scipy.stats.norm(self.matrix_mean[y][j], np.sqrt(self.matrix_variance[y][j])).pdf(x[j]))
                
                #---------- Case 1: variance is different for all i ----------#
                if self.covar == 1:
                    for j in range(self.num_features):
                        # Sum logs                   
                        array_predictions[y] += np.log(scipy.stats.norm(self.matrix_mean[y][j], np.sqrt(self.matrix_variance[y])).pdf(x[j]))
                
            predicted_labels[i] = np.argmax(array_predictions)  # Find the highest probability
            
        # return predicated lables 
        return predicted_labels
    
