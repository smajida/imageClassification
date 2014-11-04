import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import normalize


all_X = np.float32(np.load('train_inputs.npy') / 255.)
all_Y = np.load('train_outputs.npy')

n_folds = 5

n_examples = all_X.shape[0]
fs = foldsize = n_examples / n_folds
p = 0
for k in range(n_folds):
    print "Fitting fold",k
    trainX = np.vstack([all_X[:k*fs],all_X[(k+1)*fs:]])
    trainY = np.hstack([all_Y[:k*fs],all_Y[(k+1)*fs:]])
    testX = all_X[k*fs:(k+1)*fs]
    testY = all_Y[k*fs:(k+1)*fs]
    
    print trainX.shape,trainY.shape,testX.shape,testY.shape
    m = sklearn.svm.LinearSVC(C=100.,dual=False)
    m.fit(trainX, trainY)
    ypred = m.predict(testX)
    p += 1.*(ypred == testY).sum()/ypred.shape[0]
    print "Accuracy", 1.*(ypred == testY).sum()/ypred.shape[0]
    if k == 0:
        cm = confusion_matrix(ypred, testY)
        cm = cm.astype('float')
        
        Row_Normalized = normalize(cm, norm='l1', axis=1)
        
        plt.matshow(Row_Normalized)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

print p / n_folds
