# Wrapper for multilayer perception 

import csv

import numpy as np

from pylearn2.train import Train
from pylearn2.models import mlp
from pylearn2.training_algorithms.sgd import SGD,MonitorBasedLRAdjuster,LinearDecayOverEpoch
from pylearn2.costs.cost import LpPenalty, MethodCost, SumOfCosts
from pylearn2.termination_criteria import EpochCounter,MonitorBased
from pylearn2.train_extensions import best_params
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor

from theano import function
from theano import tensor as T

from mymnist import MNIST

ALL_LABELLED = True
SUPERVISED = True

rng = np.random.RandomState([2014,10,20])

def rectifier_bias():
    if rng.randint(2):
        return 0
    return rng.uniform(0, .3)

h0 = mlp.RectifiedLinear(layer_name='h0',max_col_norm=rng.uniform(1.,5.),
                         dim=rng.randint(250,5000),sparse_init=15,
                         init_bias=rectifier_bias())
                         
h1 = mlp.RectifiedLinear(layer_name='h1',max_col_norm=rng.uniform(1.,5.),
                         dim=rng.randint(250,5000),sparse_init=15,
                         init_bias=rectifier_bias())

ylayer = mlp.RectifiedLinear(layer_name='y',max_col_norm=rng.uniform(1.,5.),
                             dim=10,sparse_init=5)

layers = [h0,h1,ylayer]
 
md = mlp.MLP(layers=layers,batch_size=100,nvis=2304)

train_set = MNIST('train',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)
valid_set = MNIST('test',one_hot=True,all_labelled=ALL_LABELLED,supervised=SUPERVISED)

monitoring = dict(valid=valid_set)

costs = [mlp.Default(),mlp.L1WeightDecay(coeffs=[ 1.0E-6,1.0E-6,1.0E-6,1.0E-6,1.0E-6,1.0E-6, ])]
cost = SumOfCosts(costs)
#cost = LpPenalty(variables=md.get_params(), p=2)

#termination = MonitorBased(channel_name='valid_y_misclass',prop_decrease=0.,N=100)
termination = EpochCounter(4000)

if rng.randint(2):
    msat = 2
else:
    msat = rng.randint(2, 1000)

alg = SGD(batch_size=100,learning_rate=10.0**rng.uniform(-2.,-.5),cost=cost,
            monitoring_dataset = monitoring,termination_criterion=termination,
            #learning_rule=MomentumAdjustor(start=1,saturate=msat,final_momentum=rng.uniform(.5, .9))
            )


extensions = [#best_params.MonitorBasedSaveBest(channel_name='valid_y_misclass',store_best_model=True),
              LinearDecayOverEpoch(start=1,saturate=rng.randint(200, 1000),decay_factor=10. ** rng.uniform(-3, -1)),
              MonitorBasedLRAdjuster()]


save_path = './results/mlp_trained.pkl'
trainer = Train(dataset=train_set,model=md,algorithm=alg,extensions=extensions,save_path=save_path,save_freq=5)
trainer.main_loop()

print 'Done training'

X = md.get_input_space().make_batch_theano()
Y = md.fprop(X)
 
y = T.argmax(Y, axis=1)
f = function([X], y)

test = MNIST('test',one_hot=False,all_labelled=False,supervised=SUPERVISED)
yhat = f(test.X)

outfile = './results/mlp_test_outputs.csv'
test_output_file = open(outfile, "wb")
writer = csv.writer(test_output_file, delimiter=',') 
writer.writerow(['Id', 'Prediction']) 
for idx, predict in enumerate(yhat):
    row = [idx+1, predict]
    writer.writerow(row)
test_output_file.close()