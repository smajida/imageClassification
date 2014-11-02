# Loading transformed MNIST
# Author: Yuting Wen

import os
import numpy as np
from pylearn2.datasets import dense_design_matrix

class MNIST(dense_design_matrix.DenseDesignMatrix):
    
    def __init__(self,which_set,one_hot=True,all_labelled=True,supervised=True,extra=False,
                 shuffle=False,center=False,
                 binarize=False,start=None,
                 stop=None,axes=['b', 0, 1, 'c'],
                 preprocessor=None,
                 fit_preprocessor=False,
                 fit_test_preprocessor=False):
        
        path = "./data_and_scripts/"
        
        if all_labelled == True:
            """
            Using 80% training set as training set; 20% as labelled test set
            """  
            self.X = np.load(os.path.join(path,'train_inputs.npy'), 'r').astype('float32')
            self.y = np.load(os.path.join(path,'train_outputs.npy'), 'r').astype('float32')
    
            if which_set == 'train':
                self.X = self.X[:40000,:]
                self.y = self.y[:40000]
            elif which_set == 'test':
                self.X = self.X[40000:,:]
                self.y = self.y[40000:]
                
            if one_hot:
                one_hot = np.zeros((self.y.shape[0], 10), dtype='float32')
                for i in xrange(self.y.shape[0]):
                    one_hot[i, self.y[i]] = 1.
                self.y = one_hot
        
        elif all_labelled == False and supervised == True:
            """
            Using full training set as training set and unlabelled test set
            """
            
            if which_set == 'train':
                self.X = np.load(os.path.join(path,'train_inputs.npy'), 'r').astype('float32')
                self.y = np.load(os.path.join(path,'train_outputs.npy'), 'r').astype('float32')
                
                if extra:
                    train_mnist_X = np.load(os.path.join(path,'train_inputs_mnist.npy'), 'r').astype('float32')
                    train_mnist_y = np.load(os.path.join(path,'train_outputs_mnist.npy'), 'r').astype('float32')
                    self.X = np.vstack([self.X,train_mnist_X])
                    self.y = np.vstack([self.y,train_mnist_y])

                if one_hot:
                    one_hot = np.zeros((self.y.shape[0], 10), dtype='float32')
                    for i in xrange(self.y.shape[0]):
                        one_hot[i, self.y[i]] = 1.
                    self.y = one_hot
                
            elif which_set == 'test':
                self.X = np.load(os.path.join(path,'test_inputs.npy'), 'r').astype('float32')
                if one_hot:
                    self.y = np.zeros((self.X.shape[0],10),dtype='float32')
                else:
                    self.y = None
        
        else:
            """
            Using full training set and unlabelled test set as training set
            """
            if which_set == 'train':
                X_train = np.load(os.path.join(path,'train_inputs.npy'), 'r').astype('float32')
                X_test = np.load(os.path.join(path,'test_inputs.npy'), 'r').astype('float32')
                self.X = np.vstack([X_train,X_test])
                
                if extra:
                    train_mnist_X = np.load(os.path.join(path,'train_inputs_mnist.npy'), 'r').astype('float32')
                    self.X = np.vstack([self.X,train_mnist_X])
                
            elif which_set == 'test':
                self.X = np.load(os.path.join(path,'test_inputs.npy'), 'r').astype('float32')
                print 'train shape ', self.X.shape[0]
                            
            
            if one_hot:
                self.y = np.zeros((self.X.shape[0],10),dtype='float32')
            else:
                self.y = None
        
        view_converter = dense_design_matrix.DefaultViewConverter((48, 48, 1))

        super(MNIST,self).__init__(
            X=self.X, y=self.y, view_converter=view_converter)
