"get predictions for a test set"

import numpy as np
from pybrain.tools.xml.networkreader import NetworkReader

from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from sklearn.metrics import mean_squared_error as MSE

test_file = 'smallDigitTraining.csv'
model_file = 'Classmodel.xml'
output_predictions_file = 'predictions.txt'

# load model

net = NetworkReader.readFrom(model_file) 

# load data
tf = open(test_file, 'r')

input_size = 784
target_size = 1
ds = SDS(input_size, target_size)

y_test = []
for line in tf.readlines():
    data = [float(x) for x in line.strip().split(',') if x != '']
    indata =  tuple(data[1:])
    outdata = data[0]
    y_test.append(data[0])
    ds.addSample(indata,outdata)

# predict

p = net.activateOnDataset( ds )
print p
roundedVals = []
for n in p:
    roundedVals.append(round(n))
    
mse = MSE( y_test, roundedVals )
rmse = sqrt( mse )

print "testing RMSE:", rmse

error = np.mean( y_test != roundedVals )
print "Percent Error: ", error 

np.savetxt( output_predictions_file, np.c_[roundedVals, y_test])
