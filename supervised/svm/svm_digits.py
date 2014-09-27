print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

import csv
import pdb

from sklearn import svm
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from filehelper import Images

###############################################################################
# Load data
datafile = '../data/trainsmall.csv'
images = Images()
images.loadData(datafile)
X, y = shuffle(images.data, images.target, random_state=13)
X = X.astype(np.uint16)
y = y.astype(np.uint16)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


###############################################################################
# Fit regression model
correctPredictions = []
kernelFunctions = ['linear', 'poly', 'rbf', 'sigmoid']
#kernelFunctions = ['linear']

for i, l in enumerate(kernelFunctions):
  X_train, y_train = X[:offset], y[:offset]
  X_test, y_test = X[offset:], y[offset:]
  clf = svm.SVC(kernel=kernelFunctions[i])
  clf.fit(X_train, y_train)
  pred = clf.predict(X_test)
  mse = mean_squared_error(y_test, pred)
  print("MSE: %.4f" % mse)

  correct = 0
  for j in range(0, len(y_test)):
    if y_test[j] == round(pred[j]):
      correct = correct + 1

  correctPredictions.append((float(correct)/len(y_test)) * 100.0)
  print("Prediction Correct Rate: " + str(((float(correct)/len(y_test)) * 100.0)))


###############################################################################
# Plot training deviance

rects = plt.bar(np.arange(4), correctPredictions)

plt.xlabel('Kernels')
plt.ylabel('Prediction Correctness')
plt.title('Prediction Correctness by Kernel Type')
plt.xticks(np.arange(4) + .4, kernelFunctions)
plt.legend()

plt.tight_layout()
plt.show()