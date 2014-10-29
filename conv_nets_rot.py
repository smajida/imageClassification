import time
import numpy as np
import scipy.misc

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

theano.config.floatX = 'float32'

def make_hidden_layer(x, nin, nout, activation):
    print (nin,nout),(nout,)
    k = np.sqrt(6./(nin+nout))
    W = theano.shared(np.random.uniform(-k,k,(nin,nout)),name='W')
    b = theano.shared(np.zeros((nout)),name='b')
    return [W,b], activation(T.dot(x,W)+b)


def make_conv_layer(input, filter_shape, image_shape, poolsize=(3, 3)):
    print filter_shape, image_shape, "(",((image_shape[2]-filter_shape[2]+1)/poolsize[0])**2 * filter_shape[0],")"

    fan_in = np.prod(filter_shape[1:])
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
               np.prod(poolsize))

    k = np.sqrt(6./(fan_in+fan_out))
    W = theano.shared(np.float32(np.random.uniform(-k, k, size=filter_shape)),name='Wconv',borrow=True)
    
    # bias on individual filters
    b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='bconv',borrow=True)

    conv_out = conv.conv2d(
        input=input,
        filters=W,
        filter_shape=filter_shape,
        image_shape=image_shape
    )

    # downsample feature maps
    pooled_out = downsample.max_pool_2d(
        input=conv_out,
        ds=poolsize,
        ignore_border=True
    )

    # broadcast bias accross batch and activations (so each bias is
    # used for a single filter)
    output = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

    return [W, b], output


class ConvNet:
    def __init__(self,
                 batch_size = 200,
                 nfilters = [40,40,200],
                 sizeA = 7,
                 sizeB = 5,
                 sizeC = 5,
                 nhidden = 100,
                 gamma = 1e-6,
                 imgW = 48):
        self.batch_size = batch_size
        x = T.matrix()
        y = T.matrix()
        lr = T.scalar()
        
        img_x = x.reshape((batch_size,1,imgW,imgW))
        # 
        c_p, c = make_conv_layer(img_x, [nfilters[0],1,sizeA,sizeA], 
                                 [batch_size, 1, imgW, imgW],
                                 (2,2))
        new_size = (imgW-sizeA+1)/2
        #
        d_p, d = make_conv_layer(c, [nfilters[1],nfilters[0],sizeB,sizeB], 
                                 [batch_size, nfilters[0], new_size, new_size],
                                 (2,2))
        new_size = (new_size-sizeB+1)/2
        #
        e_p, e = make_conv_layer(d, [nfilters[2],nfilters[1],sizeC,sizeC], 
                                 [batch_size, nfilters[1], new_size, new_size],
                                 (2,2))
        new_size = (new_size-sizeC+1)/2

        flat_size = nfilters[-1] * new_size**2
        # 

        f = e.reshape((batch_size,flat_size))
        h_p, h = make_hidden_layer(f,
                                   flat_size, nhidden, T.tanh)
        #
        o_p, o = make_hidden_layer(h, nhidden,
                                   10, T.nnet.softmax)

        
        params = c_p + o_p + d_p + e_p + h_p 
        print params
        L2 = sum([T.sum(i**2) for i in params if 'W' in i.name])
        
        cost = T.sum((y-o)**2) / y.shape[0] + gamma * L2

        grads = T.grad(cost, params)
        updates = [[p, p - lr * g] for g,p in zip(grads, params)]
        
        self.learn = theano.function([x,y,lr],
                                     cost,
                                     updates=updates)
        self.evaluate = theano.function([x],
                                        o)
        
    def fit(self,data, step_test, testfunc, nepochs=800, tau = 70):
        X, Y = data
        lr = 0.1

        nbatches = X.shape[0] / self.batch_size

        print "training."

        def rot(X):
            return np.float32([1./255*scipy.misc.imrotate(
                i.reshape((48,48)),np.random.uniform(0,360)).flatten() for i in X])

        range_len_X = range(len(X))
        # train for a while
        for i in range(nepochs):
            c = 0
            t0 = time.time()
            for batch in range(nbatches):
                x = X[batch*self.batch_size:(batch+1)*self.batch_size]
                #x = X[np.random.choice(range_len_X,self.batch_size)]
                x = rot(x)
                #x += np.float32(np.random.uniform(-4./255,4./255,x.shape))
                c += self.learn(x,
                                Y[batch*self.batch_size:(batch+1)*self.batch_size],
                                lr * tau / (i * 1. + tau))
            print i,c,time.time()-t0
            if i%step_test == 0:
                testfunc(i)

    def test(self, data):
        x,Y = data
        y = self.eval(x)
        cerr = 1-1.*np.equal(np.argmax(y,axis=1),(np.argmax(Y,axis=1))).sum() / Y.shape[0]
        # return mean square error and classification error
        return ((Y-y)**2).sum() / y.shape[0], cerr

    def test_rot(self, data):
        x,Y = data
        y = self.eval_rot(x)
        cerr = 1-1.*np.equal(np.argmax(y,axis=1),(np.argmax(Y,axis=1))).sum() / Y.shape[0]
        # return mean square error and classification error
        return ((Y-y)**2).sum() / y.shape[0], cerr, y

    def eval(self, x):
        y_i = []
        for k in range(x.shape[0] / self.batch_size):
            y_i.append(self.evaluate(x[k*self.batch_size:(k+1)*self.batch_size]))
        y = np.vstack(y_i)
        return y


    def eval_rot(self, x):
        def rot(X,r):
            return np.float32([1./255*scipy.misc.imrotate(
                i.reshape((48,48)),r).flatten() for i in X])
        y_i = []
        for k in range(x.shape[0] / self.batch_size):
            y_k = []
            for irot in range(0,360,36):
                y_k.append(self.evaluate(rot(x[k*self.batch_size:(k+1)*self.batch_size],irot)))
            y_i.append(np.float32(y_k).mean(axis=0))
        y = np.vstack(y_i)
        return y

def onehot(a,maxv):
    z = np.zeros((a.shape[0],maxv),dtype='float32')
    z[range(a.shape[0]),a.flatten()] = 1
    return z

def main():
    n_folds = 5
    
    all_X = np.float32(np.load('train_inputs.npy') / 255.)
    all_Y = np.load('train_outputs.npy').reshape((all_X.shape[0],1))

    n_examples = all_X.shape[0]
    fs = foldsize = n_examples / n_folds
    # k folds
    for k in range(n_folds):
        trainX = np.vstack([all_X[:k*fs],all_X[(k+1)*fs:]])#[:1000]
        trainY = onehot(np.vstack([all_Y[:k*fs],all_Y[(k+1)*fs:]]), 10)#[:1000]
        testX = all_X[k*fs:(k+1)*fs]#[:1000]
        testY = onehot(all_Y[k*fs:(k+1)*fs], 10)#[:1000]
        print trainX.shape,trainY.shape,testX.shape,testY.shape
        

        net = ConvNet()
        def showerr():
            print net.test([trainX,trainY]), net.test([testX, testY]), net.test_rot([testX, testY])
        net.fit([trainX,trainY], 5, showerr)
        print "End of train for fold",k
        print net.test([testX, testY])


def main_test():
    n_folds = 5
    
    all_X = np.float32(np.load('train_inputs.npy') / 255.)
    all_Y = np.load('train_outputs.npy').reshape((all_X.shape[0],1))
    test_X = np.float32(np.load('test_inputs.npy') / 255.)
    
    n_examples = all_X.shape[0]
    fs = foldsize = n_examples / n_folds
    # k folds
    for k in range(n_folds):
        trainX = np.vstack([all_X[:k*fs],all_X[(k+1)*fs:]])#[:1000]
        trainY = onehot(np.vstack([all_Y[:k*fs],all_Y[(k+1)*fs:]]), 10)#[:1000]
        testX = all_X[k*fs:(k+1)*fs]#[:1000]
        testY = onehot(all_Y[k*fs:(k+1)*fs], 10)#[:1000]
        print trainX.shape,trainY.shape,testX.shape,testY.shape
        

        net = ConvNet()
        best_err = [1]
        def showerr(epoch):
            q = (net.test([trainX,trainY]), net.test([testX, testY]), net.test_rot([testX, testY]))
            print q[0],q[1],(q[2][0],q[2][1])
            file('log.txt','a').write(str("["+str(time.time()))+"] "+str((q[0],q[1],(q[2][0],q[2][1])))+"\n")
            if q[2][1] < best_err[0]:
                best_err[0] = q[2][1]
                y = net.eval_rot(test_X)
                f = file('test_output_40_40_200_755_100_%dep_%.4f.csv'%(epoch,100*best_err[0]),'w')
                f.write("Id,Prediction\n")
                for i,p in enumerate(y):
                    f.write("%d,%d\n"%(i+1,np.argmax(p)))
                f.close()
        net.fit([trainX,trainY], 10, showerr)
        print "End of train for fold",k
        print net.test([testX, testY])
        #####
        break


def test():
    def q(X):
        return np.array([scipy.misc.imresize(i.reshape((48,48)),(28,28)).flatten() for i in X])

    all_X = np.float32(np.load('train_inputs.npy') / 255.)
    all_Y = onehot(np.load('train_outputs.npy').reshape((all_X.shape[0],1)), 10)
    test_X = np.float32(np.load('test_inputs.npy') / 255.)#[:1000]
    net = ConvNet()
    #all_X = all_X[:1000]
    #all_Y = all_Y[:1000]
    def showerr():
        print net.test([all_X,all_Y])
        y = net.eval(test_X)
        f = file('test_output_40_13.3_5.2_400.csv','w')
        f.write("Id,Prediction\n")
        for i,p in enumerate(y):
            f.write("%d,%d\n"%(i+1,np.argmax(p)))
        f.close()
    net.fit([all_X,all_Y], 10, showerr)
        
    

if __name__=="__main__":
    main_test()
    #main()
    #test()
