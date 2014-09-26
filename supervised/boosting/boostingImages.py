print(__doc__)

# Author: Nick Robinson  <nick@nlrobinson.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from filehelper import Images
from sklearn.metrics import confusion_matrix

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
max_learners = np.arange(2, 400, 20)
correctPredictions = []
correctTrainingPredictions = []

for i, l in enumerate(max_learners):
  X_train, y_train = X[:offset], y[:offset]
  X_test, y_test = X[offset:], y[offset:]
  clf = ensemble.GradientBoostingClassifier(n_estimators=l, max_depth=3)
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
  print("Prediction Correct Rate Test: " + str(((float(correct)/len(y_test)) * 100.0)))

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
plt.title('Boosting: Performace vs Number of Learners')
plt.plot(max_learners, correctPredictions, lw=2, label = 'Prediction Correctness Test')
plt.plot(max_learners, correctTrainingPredictions, lw=2, label = 'Prediction Correctness Training')
plt.legend()
plt.xlabel('Number of Learners')
plt.ylabel('Correct Prediction Percentage')
plt.show()