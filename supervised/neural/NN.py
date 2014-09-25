# NN.py  by Omar Metwally, with extensive code borrowed from
# the pybrain tutorial. This is a simple demonstration of 
# the pybrain library that uses the MNIST data to train
# a deep neural network via backpropagation. Then the feed-
# forward network is used to make predictions on a test set 
# of digits.
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# load raw data from csv files 
f = open("train.csv")
data = f.read()
f.close()
f = open("smallDigitTraining.csv")
test_data = f.read()
f.close()

# initialize output file 'output.txt' with predicted classes
fileout = open("output.txt","w")
data = data.split('\n')
test_data = test_data.split('\n')

# delete csv file headers 
del data[0]
del test_data[0]

features = []
test_features = []
targets= []

# load csv data into lists 'data' and 'test_data'
# note that python's csv library was not used to make
# this demo code clearer and keep the focus on pybrain
for row in data:
    row = row.split(',')

    if row[0]:
        temp = []
        temp.append(int(row[0]))

        temp2 = []
        num_col = len(row)
        i = 1
        while i < num_col:
            temp2.append(int(row[i])/255.0)
            i += 1
        
        features.append(temp2)
        targets.append(temp)

print "number of features: ",len(features[1])

for row in test_data:
    row = row.split(',')

    num_col = len(row)
    i = 0
    temp = [] 
    if num_col > 3:
        while i < num_col:
            temp.append(int(row[i])/255.0)
            i+=1
        
    test_features.append(temp)

# purge incomplete rows from 'data' and 'test_data'
temp = []
for row in data:
    row = row.split(',')
    if len(row) >3:
        temp.append(row)

data = temp
temp2 = []

for row in test_data:
    row = row.split(',')
    if len(row)>3:
        temp2.append(row)

test_data = temp2

num_input = len(data)
num_test = len(test_data)

# initialize a network with number of input units = number
# of pixels; two hidden units with 1000 units, and 10 output units
# default is sigmoid hidden units and linear input/ouput units
# very important to specify outclass=SoftmaxLayer
net = buildNetwork(len(features[1]),300,100,10,outclass=SoftmaxLayer)
print "number of inputs m: ",num_input

# initialize two classification data sets, one for training
# and cross-validation purposes, the other for the test data
# default parameter 'target' in method ClassificationDataSet
# is '1'
DS = ClassificationDataSet(len(features[1]),nb_classes=10)
test_DS = ClassificationDataSet(len(features[1]),nb_classes=10)

i = 0
# as written, the follwing 3 lines feed only the first
# 10000 training cases into the NN for training, for speed
# and demonstration purposes. For real training, use 
# while i < num_input:
while i < 1000:
    DS.appendLinked(features[i], targets[i])
    i+=1
i = 0

# as written, the following 3 lines predict only the first
# 50 test cases, for the sake of speed and demonstration
while i < 50:
    test_DS.appendLinked(test_features[i], 0)
    i+=1

# split up the classification data set 'DS' into training
# and cross-validation sets
cvdata, trndata = DS.splitWithProportion(0.2)

# the _convertToOneOfMany method
DS._convertToOneOfMany(bounds=[0,1])
test_DS._convertToOneOfMany(bounds=[0,1])
cvdata._convertToOneOfMany(bounds=[0,1])
trndata._convertToOneOfMany(bounds=[0,1])

trainer = BackpropTrainer(net, dataset=trndata,momentum=0.1,verbose=True, weightdecay=0.01)


# train the NN 1000 separate times, printing out train and CV errors
# each time

i =0
while i < 1000:
    # specify the number of epochs
    trainer.trainEpochs(1)
    trnresults = percentError(trainer.testOnClassData(),
            trndata['class'])
    cvresults = percentError(trainer.testOnClassData(dataset = cvdata), cvdata['class'])

    print "epoch: %4d" % trainer.totalepochs, \
        " train error: %5.2f%%" %trnresults, \
        " CV error: %5.2f%%" % cvresults
    
    i+=1

i = 0

# feed test data set into NN and print the results to screen
out = net.activateOnDataset(test_DS)
out = out.argmax(axis=1) # sl 
print "predicted digits: "
print out

