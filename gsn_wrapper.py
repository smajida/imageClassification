# Wrapper for generative stochastic network

import cPickle as pickle
import csv

import numpy as np

from pylearn2.expr.activations import rescaled_softmax
from pylearn2.costs.autoencoder import MeanBinaryCrossEntropy
from pylearn2.costs.gsn import GSNCost
from pylearn2.corruption import (BinomialSampler, GaussianCorruptor,
                                 MultinomialSampler, SaltPepperCorruptor,
                                 SmoothOneHotCorruptor)
from pylearn2.models.gsn import GSN, JointGSN
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train
from pylearn2.training_algorithms.sgd import SGD, MonitorBasedLRAdjuster
from pylearn2.distributions.parzen import ParzenWindows

from mymnist import MNIST

HIDDEN_SIZE = 1000
SALT_PEPPER_NOISE = 0.3
GAUSSIAN_NOISE = 0.5

WALKBACK = 1

LEARNING_RATE = 0.015
MOMENTUM = 0.65

MAX_EPOCHS = 5500
BATCHES_PER_EPOCH = None
BATCH_SIZE = 50
MONITORING_BATCHES = 10

ALL_LABELLED = False
SUPERVISED = True

def test_train_supervised():
    ds = MNIST(which_set='train',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)

    gsn = GSN.new(
        layer_sizes=[ds.X.shape[1], HIDDEN_SIZE, ds.y.shape[1]],
        activation_funcs=["sigmoid", "tanh", rescaled_softmax],
        pre_corruptors=[GaussianCorruptor(GAUSSIAN_NOISE)] * 3,
        post_corruptors=[SaltPepperCorruptor(SALT_PEPPER_NOISE), None, SmoothOneHotCorruptor(GAUSSIAN_NOISE)],
        layer_samplers=[BinomialSampler(), None, MultinomialSampler()],
        tied=False
    )

    _rcost = MeanBinaryCrossEntropy()
    reconstruction_cost = lambda a, b: _rcost.cost(a, b) / ds.X.shape[1]

    _ccost = MeanBinaryCrossEntropy()
    classification_cost = lambda a, b: _ccost.cost(a, b) / ds.y.shape[1]

    c = GSNCost(
        [
            (0, 1.0, reconstruction_cost),(2, 2.0, classification_cost)
        ],
        walkback=WALKBACK,
        mode="supervised"
    )

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=MONITORING_BATCHES
    )

    trainer = Train(ds, gsn, algorithm=alg,
                    save_path="./results/gsn_sup_trained.pkl", save_freq=10,
                    extensions=[MonitorBasedLRAdjuster()])
    trainer.main_loop()
    print "done training"

def test_train_ae():
    ds = MNIST(which_set='train',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)

    gsn = GSN.new(
        layer_sizes=[ds.X.shape[1], HIDDEN_SIZE,ds.X.shape[1]],
        activation_funcs=["sigmoid", "tanh", rescaled_softmax],
        pre_corruptors=[GaussianCorruptor(GAUSSIAN_NOISE)] * 3,
        post_corruptors=[SaltPepperCorruptor(SALT_PEPPER_NOISE), None,SmoothOneHotCorruptor(GAUSSIAN_NOISE)],
        layer_samplers=[BinomialSampler(), None, MultinomialSampler()],
        tied=False
    )

    _mbce = MeanBinaryCrossEntropy()
    reconstruction_cost = lambda a, b: _mbce.cost(a, b) / ds.X.shape[1]

    c = GSNCost([(0, 1.0, reconstruction_cost)], walkback=WALKBACK)

    alg = SGD(
        LEARNING_RATE,
        init_momentum=MOMENTUM,
        cost=c,
        termination_criterion=EpochCounter(MAX_EPOCHS),
        batches_per_iter=BATCHES_PER_EPOCH,
        batch_size=BATCH_SIZE,
        monitoring_dataset=ds,
        monitoring_batches=MONITORING_BATCHES
   )

    trainer = Train(ds, gsn, algorithm=alg, save_path="./results/gsn_ae_trained.pkl",
                    save_freq=5, extensions=[MonitorBasedLRAdjuster()])
    trainer.main_loop()
    print "done training"

def test_classify():
    with open("./results/gsn_sup_trained.pkl") as f:
        gsn = pickle.load(f)

    gsn = JointGSN.convert(gsn)
    gsn._corrupt_switch = False

    ds = MNIST(which_set='test',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)
    mb_data = ds.X
    y = ds.y

    outfile = open("./results/gsn_train_outputs.csv","wb")
    writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL) 

    for i in xrange(1, 10):
        y_hat = gsn.classify(mb_data, trials=i)
        errors = np.abs(y_hat - y).sum() / 2.0
        errors_normalize = errors / mb_data.shape[0]

        writer.writerow([i, errors, errors_normalize])
        writer.writerow(y_hat)
        print i, errors, errors_normalize
        
    outfile.close()

def test_unlabelled_classify():
    if SUPERVISED == True:
        outfile = './results/gsn_sup_test_outputs.csv'
        with open("./results/gsn_sup_trained.pkl") as f:
            gsn = pickle.load(f)
    else:
        outfile = './results/gsn_ae_test_outputs.csv'
        with open("./results/gsn_ae_trained.pkl") as f:
            gsn = pickle.load(f)
            
    gsn = JointGSN.convert(gsn)
    gsn._corrupt_switch = False

    ds = MNIST(which_set='test',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)
    
    mean = gsn._get_aggregate_classification(ds.X)
    am = np.argmax(mean, axis=1).astype(int)
    print 'am shape: ', am.shape
    
    test_output_file = open(outfile, "wb")
    writer = csv.writer(test_output_file, delimiter=',') 
    writer.writerow(['Id', 'Prediction']) 
    for idx, predict in enumerate(am):
        row = [idx+1, predict]
        writer.writerow(row)
    test_output_file.close()

if __name__ == '__main__':
    if SUPERVISED == True:
        test_train_supervised()
        if ALL_LABELLED == True:
            test_classify()
        else:
            test_unlabelled_classify()
    else:
        test_train_ae()
        test_unlabelled_classify()