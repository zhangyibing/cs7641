print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

import csv
import pdb

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

###############################################################################
#Image Class
class Images:
    def __init__(self):
        self.data = []
        self.target = []
        self.feature_names = np.array([])

    def loadData(self, filename):
        with open(datafile) as csvfile:
            file_reader = csv.reader(csvfile)

            #Deal with headers
            features = file_reader.next()
            i = 0
            for fields in features:
              if i != 0:
                self.feature_names = np.append(self.feature_names, fields)
              i = i + 1

            #Skip header line and seed data/targets
            next(file_reader, None)
            for row in file_reader:
                self.data.append(row[1:])
                self.target.append(row[0])


###############################################################################
# Load data
#boston = datasets.load_boston()
#X, y = shuffle(boston.data, boston.target, random_state=13)
#X = X.astype(np.float32)
#offset = int(X.shape[0] * 0.9)
#X_train, y_train = X[:offset], y[:offset]
#X_test, y_test = X[offset:], y[offset:]

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

for i, l in enumerate(max_learners):
  X_train, y_train = X[:offset], y[:offset]
  X_test, y_test = X[offset:], y[offset:]
  #params[i] = {'n_estimators': i, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.1}
  clf = ensemble.GradientBoostingClassifier(n_estimators=l, max_depth=4)
  #clf = ensemble.AdaBoostClassifier(n_estimators=l, learning_rate=0.1)
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

plt.figure()
plt.title('Boosting: Performace vs Number of Learners')
plt.plot(max_learners, correctPredictions, lw=2, label = 'Prediction Correctness')
plt.legend()
plt.xlabel('Number of Learners')
plt.ylabel('Correct Prediction Percentage')
plt.show()

###############################################################################
# Plot feature importance
#feature_importance = clf.feature_importances_
# make importances relative to max importance
#feature_importance = 100.0 * (feature_importance / feature_importance.max())
#sorted_idx = np.argsort(feature_importance)
#pos = np.arange(sorted_idx.shape[0]) + .5
#plt.subplot(1, 2, 2)
#plt.barh(pos, feature_importance[sorted_idx], align='center')
#pdb.set_trace()
#plt.yticks(pos, images.feature_names[sorted_idx])
#plt.xlabel('Relative Importance')
#plt.title('Variable Importance')
plt.show()