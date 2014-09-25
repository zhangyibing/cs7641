from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.xml.networkwriter import NetworkWriter

#For Classification
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.utilities           import percentError
from sklearn.metrics import accuracy_score

#Plotting Libraries
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal

input_size = 784
target_size = 1
hidden_size = 50   # arbitrarily chosen

ds = ClassificationDataSet(input_size, target_size, nb_classes=10)

tf = open('train1000.csv', 'r')
output_model_file = 'Classmodel.xml'

for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[1:])
    outdata = data[0]
    ds.addSample(indata,outdata)

tstdata, trndata = ds.splitWithProportion( 0.25 )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]


#net = buildNetwork(input_size, hidden_size, target_size, bias=True)
#trainer = BackpropTrainer( net, ds )

net = buildNetwork( trndata.indim, hidden_size, trndata.outdim, outclass=SoftmaxLayer, bias=True )
trainer = BackpropTrainer( net, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

trainer.trainUntilConvergence( verbose = True, validationProportion = 0.15, maxEpochs = 250, continueEpochs = 5 )


trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

testOutput = trainer.testOnClassData( dataset=tstdata, verbose=True )
tstresult = percentError( trainer.testOnClassData( dataset=tstdata ), tstdata['class'] )

print "epoch: %4d" % trainer.totalepochs, \
      "  train error: %5.2f%%" % trnresult, \
      "  test error: %5.2f%%" % tstresult


NetworkWriter.writeToFile(net, output_model_file)