from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet

import numpy as np
import matplotlib.pyplot as plt

import pdb

ds = ClassificationDataSet(11,1, nb_classes=10)

tf = open('winequality-white.csv', 'r')

for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[:11])
    outdata = tuple(data[11:])
    ds.addSample(indata,outdata)

tstdata, trndata = ds.splitWithProportion( 0.25 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

n = buildNetwork(trndata.indim,8,8,trndata.outdim,recurrent=True)
t = BackpropTrainer(n,learningrate=0.01,momentum=0.5,verbose=True,weightdecay=0.01)
t.trainUntilConvergence(trndata,maxEpochs=100, verbose=True)
pdb.set_trace()
t.testOnData(verbose=True)
