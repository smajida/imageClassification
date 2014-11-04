# Implementation of MLP using SGD and cross-validation

import csv

import theano
from theano import function
import theano.tensor as T

import numpy as np

rng = np.random.RandomState(201314)

#theano.config.compute_test_value = 'warn'

# debug functions
def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs],

def inspect_outputs(i, node, fn):
    print "output(s) value(s):", [output[0] for output in fn.outputs]

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

class HiddenLayer:

    def __init__(self,
                input,
                n_in,
                n_out,
                activation):
        #print n_in,n_out
        
        bound = np.sqrt(6./(n_in+n_out))
        
        W = np.asarray(rng.uniform(-bound,bound,(n_in,n_out)),
                       dtype=theano.config.floatX)
        W = theano.shared(value=W,name='Whidden',borrow=True)
        
        b = np.zeros((n_out,),dtype=theano.config.floatX)
        b = theano.shared(value=b,name='bhidden',borrow=True)
        
        self.params = [W,b]

        self.output = activation(T.dot(input,W)+b)
        #output_print = T.printing.Print('output:')(self.output)
        #output_print

class MLP:

    def __init__(self,
                layer_size_list,
                decay=1e-3,
                batch_size = 50):
        
        self.batch_size = batch_size
        self.n_hidden_layer = len(layer_size_list) - 1
        
        x = T.fmatrix('x')
        y = T.imatrix('y')
        lr = T.fscalar('lr')

        #x.tag.test_value = np.float32(np.random.rand(50,2304))
        #y.tag.test_value = _onehot(np.random.randint(10,size=50),10)
        #lr.tag.test_value = np.random.rand()

        params = []
        layer_list = []
        for i in range(self.n_hidden_layer):
            
            if i==0:
                input = x
            else:
                input = prev_layer.output
            
            if i<self.n_hidden_layer-1:
                activation = T.tanh
            else:
                activation = T.nnet.softmax
            
            layer = HiddenLayer(input=input,
                                n_in=layer_size_list[i],
                                n_out=layer_size_list[i+1],
                                activation=activation)
                      
            params += layer.params
            #print params
            layer_list.append(layer)
            
            prev_layer = layer
        
        #print layer_list[-1].output
        cross_entropy = T.mean(-T.sum(y*T.log(layer_list[-1].output),axis=1))

        regularization = decay*sum([T.sum(i**2) for i in params if 'W' in i.name])

        cost = cross_entropy + regularization
        
        grad_params = T.grad(cost,params)

        updates = [(p,p-lr*g) for p,g in zip(params,grad_params)]

        self.prediction = function([x],
                                    layer_list[-1].output
                                    #mode=theano.compile.MonitorMode(
                                        #pre_func=inspect_inputs,
                                        #post_func=inspect_outputs
                                        #post_func=detect_nan)
                                    )
        
        self.sgd = function([x,y,lr],
                            cost,
                            updates=updates
                            #mode=theano.compile.MonitorMode(
                                #pre_func=inspect_inputs,
                                #post_func=inspect_outputs
                                #post_func=detect_nan)
                            )
        
    def predict(self,x):
        
        y_list = []
        
        for k in range(x.shape[0] // self.batch_size):
            y_list.append(self.prediction(x[k*self.batch_size:(k+1)*self.batch_size]))
        
        return np.vstack(y_list)

    def train(self,
                x,
                y,
                customized_test,
                n_epoch=1500,
                lr=0.01,
                factor=80):
    
        n_batch = x.shape[0]//self.batch_size

        print 'Training...'

        for epoch in xrange(n_epoch):

            cost = 0

            for batch in xrange(n_batch):
                x_batch = x[batch*self.batch_size:(batch+1)*self.batch_size]
                x_batch += np.asarray(rng.uniform(-4./255,4./255,x_batch.shape),
                                        dtype=theano.config.floatX)
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
        
        test_pred = self.predict(test_X)
        #print test_pred
        f = open('/home/yw/imageClassification/results/mlp_test_outputs.csv','wb')
        writer = csv.writer(f, delimiter=',') 
        writer.writerow(['Id', 'Prediction']) 
        for i,pred in enumerate(test_pred):
            row = [i+1,np.argmax(pred)]
            writer.writerow(row)
        f.close()        

    def accuracy(self,x,y):
        
        y_hat = self.predict(x)
        
        sqr_err = ((y_hat-y)**2).sum() / y.shape[0]
        mis_class = 1-1.*np.equal(np.argmax(y,axis=1),(np.argmax(y_hat,axis=1))).sum() / y.shape[0]

        return sqr_err,mis_class

def _onehot(y,n_col):
    
    z = np.zeros((y.shape[0],n_col),dtype='int32')
    z[range(y.shape[0]),y.flatten()] = 1
    return z

def main(batch_size=50,n_fold=5):
    
    print 'Loading data...'
    
    all_train_X = np.float32(np.load('/home/yw/imageClassification/data_and_scripts/train_inputs.npy'))
    all_train_Y = _onehot(np.load('/home/yw/imageClassification/data_and_scripts/train_outputs.npy').reshape((all_train_X.shape[0],1)), 10)
    test_X = np.float32(np.load('/home/yw/imageClassification/data_and_scripts/test_inputs.npy')) 
    #print all_train_X[:10]
    #print all_train_Y[:10]
    
    N,M = all_train_X.shape 
    n = N // n_fold

    record = []
    
    for lr in [0.01,0.02,0.03,0.04,0.05]:
        print 'Learning rate:',lr
        
        for layer_size_list in [[M,3000,10],[M,5000,10],[M,3000,1000,10],[M,5000,2000,100,10]]:
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
            record.append([lr,layer_size_list,err_list[best_fold]])
	    print record

if __name__=="__main__":
    main()
