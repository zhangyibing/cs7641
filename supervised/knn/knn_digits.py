print(__doc__)

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from filehelper import Images

###############################################################################
# Load data
datafile = '../data/train.csv'
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
num_neighbors = np.arange(1, 10, 1)
correctPredictionsUniform = []
correctPredictionsWeighted = []

for i, l in enumerate(num_neighbors):
  X_train, y_train = X[:offset], y[:offset]
  X_test, y_testUniform, y_testWeighted = X[offset:], y[offset:], y[offset:]
  clf1 = neighbors.KNeighborsClassifier(n_neighbors=l)
  clf2 = neighbors.KNeighborsClassifier(n_neighbors=l, weights='distance')
  clf1.fit(X_train, y_train)
  pred1 = clf1.predict(X_test)
  clf2.fit(X_train, y_train)
  pred2 = clf2.predict(X_test)
  mse1 = mean_squared_error(y_testUniform, pred1)
  mse2 = mean_squared_error(y_testWeighted, pred2)
  print("MSE Uniform: %.4f" % mse1)
  print("MSE Distance: %.4f" % mse2)

  correctUniform = 0
  for j in range(0, len(y_testUniform)):
    if y_testUniform[j] == round(pred1[j]):
      correctUniform = correctUniform + 1

  correctWeighted = 0
  for j in range(0, len(y_testWeighted)):
    if y_testWeighted[j] == round(pred2[j]):
      correctWeighted = correctWeighted + 1

  correctPredictionsUniform.append((float(correctUniform)/len(y_testUniform)) * 100.0)
  print("Prediction Correct Rate Uniform: " + str(((float(correctUniform)/len(y_testUniform)) * 100.0)))

  correctPredictionsWeighted.append((float(correctWeighted)/len(y_testWeighted)) * 100.0)
  print("Prediction Correct Rate Distance: " + str(((float(correctWeighted)/len(y_testWeighted)) * 100.0)))


###############################################################################
# Plot training deviance

plt.figure()
plt.title('Decision Trees: Performace vs Number of Estimators')
plt.plot(num_neighbors, correctPredictionsUniform, lw=2, label = 'Uniform Weights')
plt.plot(num_neighbors, correctPredictionsWeighted, lw=2, label = 'Distance Weights')
plt.legend()
plt.xlabel('Number of Neighbors')
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