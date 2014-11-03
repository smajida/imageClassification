# Implementation of MLP using SGD and cross-validation
# Author: Yuting Wen

import csv
import numpy as np

rng = np.random.RandomState(201314)

class HiddenLayer(object):

    def __init__(self,
                n_in,
                n_out,
                activation):
        
        bound = np.sqrt(6./(n_in+n_out))
        self.W = np.asarray(rng.uniform(-bound,bound,(n_in,n_out)),dtype='float32')
        self.b = np.zeros((n_out,),dtype='float32')
        self.params = [self.W,self.b]
	self.activation = activation

    def predict(self,x):

	#print x.shape,self.W.shape,self.b.shape
        return self.activation(np.dot(x,self.W)+self.b)

class MLP(object):

    def __init__(self,
                layer_size_list,
                decay=1e-4,
                batch_size=50):
        
	self.decay = decay
        self.batch_size = batch_size
        self.n_hidden_layer = len(layer_size_list) - 1
        
        self.layer_list = []
        for i in range(self.n_hidden_layer):
            
            if i<self.n_hidden_layer-1:
                activation = np.tanh
            else:
                activation = _softmax
            
            layer = HiddenLayer(n_in=layer_size_list[i],
                                n_out=layer_size_list[i+1],
                                activation=activation)
            self.layer_list.append(layer)
            
    def feedforward(self,x):

	self.layer_outputs = []
	for layer in self.layer_list:
	    x = layer.predict(x)
	    self.layer_outputs.append(x)
	return self.layer_outputs[-1]

    def cost(self,x,y):

	MSE = np.mean((y-self.layer_outputs[-1])**2)
        regularization = self.decay*np.sum([np.sum(layer.W**2) for layer in self.layer_list])
        return MSE + regularization/x.shape[0]

    def sgd(self,x,y,lr):
	
        o = self.feedforward(x)
	deltas = [None]*self.n_hidden_layer
        deltas[-1] = (y-o)*o*(1.0-o)
	#print o.shape,y.shape,deltas[-1].shape

        for i in range(self.n_hidden_layer-1):
	    o = self.layer_outputs[-i-2]
	    deltas[-i-2] = o*(1.0-o)*np.dot(deltas[-i-1],self.layer_list[-i-1].W.T)
	    #print o.shape,self.layer_list[-i-1].W.shape,deltas[-i-2].shape
	    self.layer_list[-i-1].W += lr*np.dot(o.T,deltas[-i-1])
	
	return self.cost(x,y)	
        
    def predict(self,x):
        
        y_list = []    
        for k in range(x.shape[0] // self.batch_size):
            y_list.append(self.feedforward(x[k*self.batch_size:(k+1)*self.batch_size]))
        return np.vstack(y_list)

    def train(self,
                x,
                y,
                customized_test,
                n_epoch=2000,
                lr=0.01,
                factor=80):
    
        n_batch = x.shape[0]//self.batch_size

        print 'Training...'

        for epoch in xrange(n_epoch):

            cost = 0

            for batch in xrange(n_batch):
                x_batch = x[batch*self.batch_size:(batch+1)*self.batch_size]
                x_batch += np.asarray(rng.uniform(-4./255,4./255,x_batch.shape),
                                        dtype='float32')
                y_batch = y[batch*self.batch_size:(batch+1)*self.batch_size]
                #print x_batch.shape,y_batch.shape
                cost += self.sgd(x_batch,
                                y_batch,
                                lr*factor/(epoch*1.+factor))
        
            print 'Epoch:', epoch, 'Cost:', cost
            
            if epoch%10 == 0:
                customized_test()
                    
    def train_valid(self,train_X,train_Y,valid_X,valid_Y,test_X):
        
        train_MSE_CE = self.accuracy(train_X,train_Y)
        valid_MSE_CE = self.accuracy(valid_X,valid_Y)
        
        print 'Train MSE,CE:',train_MSE_CE
        print 'Valid MSE,CE:',valid_MSE_CE
        
        self.write_test(test_X)
        return train_MSE_CE + valid_MSE_CE
        
    def write_test(self,test_X):
        
        test_pred = self.feedforward(test_X)
        print test_pred
        f = open('/home/yw/imageClassification/results/mlp_test_outputs.csv','wb')
        writer = csv.writer(f, delimiter=',') 
        writer.writerow(['Id', 'Prediction']) 
        for i,pred in enumerate(test_pred):
            row = [i+1,np.argmax(pred)]
            writer.writerow(row)
        f.close()        

    def accuracy(self,x,y):
        
        y_hat = self.feedforward(x)
        sqr_err = ((y_hat-y)**2).sum() / y.shape[0]
        mis_class = 1-1.*np.equal(np.argmax(y,axis=1),(np.argmax(y_hat,axis=1))).sum() / y.shape[0]
        return sqr_err,mis_class

def _softmax(x):

    x = np.exp(x)
    if len(x.shape) == 1:
        return x/x.sum()
    else:
        return x/x.sum(axis=1).reshape((x.shape[0],1))

def _onehot(y,n_col):
    
    z = np.zeros((y.shape[0],n_col),dtype='int32')
    z[range(y.shape[0]),y.flatten()] = 1
    return z

def main(batch_size=50,n_fold=5):
    
    print 'Loading data...'
    
    all_train_X = np.float32(np.load('/home/yw/imageClassification/data_and_scripts/train_inputs.npy'))
    all_train_Y = _onehot(np.load('/home/yw/imageClassification/data_and_scripts/train_outputs.npy').reshape((all_train_X.shape[0],1)), 10)
    test_X = np.float32(np.load('/home/yw/imageClassification/data_and_scripts/test_inputs.npy')) 
    print all_train_X[:10]
    print all_train_Y[:10]
    
    N,M = all_train_X.shape 
    n = N // n_fold

    record = {}
    
    for lr in [0.01,0.02,0.03,0.04,0.05]:
        print 'Learning rate:',lr
        
        for layer_size_list in [[M,100,10],[M,1000,10],[M,500,200,10],[M,1000,500,100,10]]:
            print 'Layer size list:', layer_size_list
            
            err_list = []
            
            for f in range(n_fold):
                print 'Fold:', f
                
                train_X = np.vstack([all_train_X[:f*n,:],all_train_X[(f+1)*n:,:]])
                train_Y = np.vstack([all_train_Y[:f*n],all_train_Y[(f+1)*n:]])
                valid_X = all_train_X[f*n:(f+1)*n,:]
                valid_Y = all_train_Y[f*n:(f+1)*n]

                def customized_test():
                    model.train_valid(train_X,train_Y,valid_X,valid_Y,test_X)    

                model = MLP(batch_size=batch_size,layer_size_list=layer_size_list) 
    
                model.train(x=train_X,y=train_Y,customized_test=customized_test,lr=lr)
                
                err_list.append(customized_test())
            
            best_fold = np.argmin(err_list[:,3])
            record[(lr,layer_list)] = err_list[best_fold]

if __name__=="__main__":
    main()
