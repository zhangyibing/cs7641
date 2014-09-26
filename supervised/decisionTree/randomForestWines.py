print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from filehelper import Wines
from sklearn.metrics import confusion_matrix

###############################################################################
# Load data
datafile = '../data/winequality-white.csv'
wines = Wines()
wines.loadData(datafile)
X, y = shuffle(wines.data, wines.target, random_state=13)
X = X.astype(np.float64)
y = y.astype(np.float64)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


###############################################################################
# Fit regression model
correctPredictions = []
correctTrainingPredictions = []
num_estimators = np.arange(1, 100, 1)

for i, l in enumerate(num_estimators):
  X_train, y_train = X[:offset], y[:offset]
  X_test, y_test = X[offset:], y[offset:]
  clf = ensemble.RandomForestClassifier(n_estimators =num_estimators[i], criterion="entropy")
  clf.fit(X_train, y_train)
  pred = clf.predict(X_test)
  predTraining = clf.predict(X_train)
  mse = mean_squared_error(y_test, pred)
  print("MSE: %.4f" % mse)

  correct = 0
  for j in range(0, len(y_test)):
    if y_test[j] == round(pred[j]):
      correct = correct + 1

  correctTraining = 0
  for j in range(0, len(y_train)):
    if y_train[j] == round(predTraining[j]):
      correctTraining = correctTraining + 1

  correctPredictions.append((float(correct)/len(y_test)) * 100.0)
  print("Prediction Correct Rate: " + str(((float(correct)/len(y_test)) * 100.0)))

  correctTrainingPredictions.append((float(correctTraining)/len(y_train)) * 100.0)
  print("Prediction Correct Rate Training: " + str(((float(correctTraining)/len(y_train)) * 100.0)))

# Compute confusion matrix
cm = confusion_matrix(y_test, pred)

###############################################################################
# Plot training deviance
# 
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

plt.figure()
plt.title('Random Forest: Performace vs Number of Estimators')
plt.plot(num_estimators, correctPredictions, lw=2, label = 'Correct Prediction Percentage Test')
plt.plot(num_estimators, correctTrainingPredictions, lw=2, label = 'Correct Prediction Percentage Training')
plt.legend(loc='best')
plt.xlabel('Number of Estimators')
plt.ylabel('Correct Prediction Percentage')
plt.show()