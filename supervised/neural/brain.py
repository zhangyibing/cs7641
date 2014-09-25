from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet

ds = SupervisedDataSet(11,1)

tf = open('winequality-red.csv', 'r')

for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[:11])
    outdata = tuple(data[11:])
    ds.addSample(indata,outdata)

n = buildNetwork(ds.indim,8,8,ds.outdim,recurrent=True)
t = BackpropTrainer(n,learningrate=0.01,momentum=0.5,verbose=True)
t.trainOnDataset(ds,1000)
t.testOnData(verbose=True)
