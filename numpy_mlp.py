# -*- encoding: utf-8 -*-
# Auteurs : Emmanuel Bengio et Patrick Thémens
import numpy
import cPickle as pickle
import gzip, time

class Softmax:
    def __call__(self, x):
        x = numpy.exp(x)
        if len(x.shape) == 1:
            return x/x.sum()
        else:
            return x/x.sum(axis=1).reshape((x.shape[0],1))
    
    def eval(self, x):
        return self(x)

    def grad(self, dLdf, x, h):
        es = h.sum()
        return [dLdf * ((h * (es - h)) / (es**2))]
        
    def apply_grad(self, *args):
        pass

softmax = Softmax()

class Tanh:
    def __call__(self, x):
        return numpy.tanh(x)

    def eval(self, x):
        return self(x)

    def grad(self, dLdf, x, h):
        return [dLdf * (1 - h**2)]

    def apply_grad(self, *args):
        pass

tanh = Tanh()


class LogLoss:
    def eval(self, x, t):
        return -numpy.log(x[t])

    def grad(self, t, x, o):
        dLdos = numpy.zeros(o.shape)
        dLdos[t] = -1/o[t]
        # dLdos = numpy.array([-1/o[i] if i==t else 0 for i in range(o.shape[0])])
        return dLdos

    def merge_effect(self, x):
        return x

class SoftmaxLogLoss:
    def eval(self, x, t):
        return -numpy.log(x[numpy.arange(x.shape[0]),t])
        
    def grad(self, t, x, o):
        onehot = numpy.zeros(o.shape)
        onehot[numpy.arange(o.shape[0]),t] = 1
        return softmax(o) - onehot

    def merge_effect(self, x):
        return softmax(x)

class HiddenLayer:
    def __init__(self, n_in, n_out, decay):
        k = numpy.sqrt(6.0/(n_in + n_out))
        self.W = numpy.random.uniform(-k,k,(n_in, n_out))
        self.b = numpy.zeros((n_out,))

        self.params = [self.W, self.b]
        self.decay_params = [self.W]
        self.decay = decay

    def eval(self, x):
        return numpy.dot(x,self.W)+self.b

    def grad(self, dLdf, x, h):
        #Version avec boucle en commentaires
        #dLdW = numpy.sum([numpy.outer(x[i], dLdf[i]) for i in range(x.shape[0])],axis=0) + self.decay*2*self.W
        dLdW = numpy.dot(x.T, dLdf) + self.decay*2*self.W
        
        #dLdb = sum([i for i in dLdf])
        dLdb = dLdf.sum(axis=0)
        
        #dLdh = numpy.asarray([numpy.dot(dLdf[i], self.W.T) for i in range(x.shape[0])])
        dLdh = numpy.dot(dLdf, self.W.T)
        return dLdW, dLdb, dLdh

    def apply_grad(self, grads):
        for grad,param in zip(grads,[self.W, self.b]):
            param -= grad

class Network:
    def __init__(self, layers, activations, loss, decay):
        self.layers = []
        self.params = []
        self.decay_params = []
        self.decay = decay
        for i in range(len(layers)-1):
            self.layers.append( HiddenLayer(layers[i], layers[i+1], decay) )
            self.params += self.layers[-1].params
            self.decay_params += self.layers[-1].decay_params
            if activations[i]:
                self.layers.append( activations[i] )
        self.lossf = loss
    
    def predict(self, x):
        p = self.eval(x)
        return numpy.argmax(p, axis=1)
    
    def eval(self, x):
        for i in self.layers:
            x = i.eval(x)
        return self.lossf.merge_effect(x)
    
    def error(self, x, t):
        ev = self.eval(x)
        return -numpy.log(ev[numpy.arange(x.shape[0]),t])

    def score(self, x, t):
        ev = self.eval(x)
        return 1.0*(t != numpy.argmax(ev, axis=1)).sum() / x.shape[0]

    def loss(self, o, t):
        return self.lossf.eval(o, t) + self.decay/o.shape[0]*sum(numpy.sum(i**2) for i in self.decay_params)
    
    def grad(self, x, t):
        o = self.eval(x)
        return self.lossf.grad(t, x, o)

    def num_grad(self, x , t):
        wgrads = []
        baseline_loss = self.loss(self.eval(x), t).sum()
        epsilon = 1e-6
        for i in self.params:
            nflat = numpy.prod(i.shape)
            grad = numpy.zeros(nflat)
            copy = i.copy()
            flat = i.reshape((nflat,))
            for j in range(flat.shape[0]):
                f = flat[j]
                flat[j] += epsilon
                grad[j] = (self.loss(self.eval(x), t).sum() - baseline_loss) / epsilon
                flat[j] = f
            if (copy != i).any():
                print "Error!"
            wgrads.append(grad.reshape(i.shape))
        return wgrads, baseline_loss
              
    def grads(self, x, t):
        outputs = [x]
        for i in self.layers:
            outputs.append(i.eval(outputs[-1]))
        grads = [self.lossf.grad(t, x, outputs[-1])]
        wgrads = []
        for i,p in zip(self.layers[::-1],range(len(self.layers)-1,-1,-1)):
            gds = i.grad(grads[-1], outputs[p], outputs[p+1])
            grads.append(gds[-1])
            wgrads = list(gds[:-1]) + wgrads
        return wgrads, self.loss(self.lossf.merge_effect(outputs[-1]), t).mean()

    def update(self, grads, lr):
        for i,p in enumerate(self.params):
            p += -grads[i]*lr

    
    def step(self, x, t, lr):
        grads, loss = self.grads(x, t)
        self.update(grads, lr)
        return loss


def load_2moons():
    data = numpy.loadtxt(file("2moons.txt",'r'))
    
    ntrain = data.shape[0] * 0.8
    ntest = data.shape[0] - ntrain
    trainX, trainY = data[:ntrain,:2], numpy.int32(data[:ntrain,2])
    testX , testY  = data[:ntest, :2], numpy.int32(data[:ntest, 2])

    return trainX, trainY, testX, testY, testX, testY

def load_mnist():
    f=gzip.open('mnist.pkl.gz')
    data=pickle.load(f)

    return data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]

def main(data,
         n_epochs,
         n_hidden,
         n_classes,
         weight_decay,
         K,
         lr,
         measureTime=True,
         fileOutput=None):
    print (n_epochs, n_hidden, n_classes, weight_decay, K, lr)
    trainX, trainY, testX, testY, validX, validY = data
    
    net = Network([trainX.shape[1],n_hidden,n_classes],[tanh, None], SoftmaxLogLoss(), weight_decay)
    
    nbatches = trainX.shape[0] / K
    scurve = []
    lcurve = []

    t0 = time.time()
    for i in range(n_epochs):
        losses = 0
        for j in range(nbatches):
            losses += net.step(trainX[j*K:j*K+K],trainY[j*K:j*K+K], lr)
        t1 = time.time()
        
        errtr, errte, errva = [net.error(trainX, trainY).mean(), 
                               net.error(testX, testY).mean(),
                               net.error(validX, validY).mean()]
        scotr, scote, scova = [net.score(trainX, trainY), 
                               net.score(testX, testY),
                               net.score(validX, validY)]
        lcurve.append((errtr, errte, errva))
        scurve.append((scotr, scote, scova))
        
        losses /= nbatches
        s = ("%f %f %f %.2f%% %.2f%% %.2f%%")%(
            errtr, errte, errva,
            scotr*100, scote*100, scova*100)
        print "epoch",i, losses, s,
        if measureTime:
            print int((t1-t0)/60),"m",int(t1-t0)%60,"s (",((t1-t0)/(i+1)),"s )"
        else:
            print
        if fileOutput:
            s = ("%f %f %f %f %f %f")%(errtr, errte, errva,
                                       scotr, scote, scova)
            fileOutput.write(s+"\n")
        
    print "Erreur de test:", net.score(testX, testY)*100,"%"

    return net, [lcurve, scurve]

def test_grads():
    
    net = Network([4,5,3],[tanh, None], SoftmaxLogLoss(), 0.001)

    ngrads = net.num_grad(numpy.array([[1,2,3,4]]), [2])[0]
    cgrads = net.grads(numpy.array([[1,2,3,4]]), [2])[0]
    diff = [(ngrads[i]/cgrads[i]) for i in range(len(ngrads))]
    dmax = max([numpy.max(i) for i in diff])
    dmin = min([numpy.min(i) for i in diff])

    if dmax > 1.01 or dmin < 0.99:
        print "\n".join(str(i) for i in diff)
        raise Exception("Numerical gradient does not match computed gradient")

    x = numpy.random.random((10,4))
    y = numpy.int32(numpy.random.random((10,))*3)

    ngrads = net.num_grad(x,y)[0]
    cgrads = net.grads(x,y)[0]
    diff = [(ngrads[i]/cgrads[i]) for i in range(len(ngrads))]
    dmax = max([numpy.max(i) for i in diff])
    dmin = min([numpy.min(i) for i in diff])


    if dmax > 1.01 or dmin < 0.99:
        print "\n".join(str(i) for i in diff)
        raise Exception("Numerical gradient does not match computed gradient")

if __name__ == "__main__":
    numpy.random.seed(42)

    test_grads()

    if 0:
        data = load_2moons()
        """
        Sur l'ensemble de valid/test:
        Avec (200, 10, 2, 0.001, 5, 0.005)  on obtient <1% d'erreur
        """
        main(data,
             n_epochs = 200,
             n_hidden = 10,
             n_classes = 2,
             weight_decay = 0.001,
             K = 5,
             lr = 0.005)


    if 1:
        import utilz
        data = load_mnist()
        """
        Sur l'ensemble de validation:
        Avec (200,  20, 10, 0.001, 50, 0.005) on atteint 4.85% d'erreur
        Avec (100, 120, 10, 0.001, 10, 0.002) on atteint 1.80% d'erreur

        Avec (100, 120, 10, 0.001, 10, 0.002) on atteint 1.93% d'erreur sur l'ensemble de test
        """
        net, curves = main(data,
                           n_epochs = 100,
                           n_hidden = 120,
                           n_classes = 10,
                           weight_decay = 0.001,
                           K = 10,
                           lr = 0.002)
        utilz.plotCurves(curves[0],["train","test","valid"],"Perte", "loss.png")
        utilz.plotCurves(numpy.array(curves[1])*100,"Score (%)", "score.png")

    if 0:
        import utilz
        data = load_2moons()
        wd = 0.001
        n_epochs = 40
        n_hidden = 5
        net,_ = main(data,
                     n_epochs = n_epochs,
                     n_hidden = n_hidden,
                     n_classes = 2,
                     weight_decay = wd,
                     K = 100,
                     lr = 0.01)
        utilz.showDR(net, "n_hidden %d, decay %f, n epoques %d"%(n_hidden,wd,n_epochs),
                     -1,2.8,
                     data[0], data[1],
                     data[0][:,0].min(), data[0][:,1].min(), 
                     data[0][:,0].max(), data[0][:,1].max(),
                     "foo.png")
