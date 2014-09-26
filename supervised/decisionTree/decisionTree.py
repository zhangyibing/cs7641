from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import mean_squared_error
import pydot
import numpy as np

wine_data = np.loadtxt('trainingRedX.csv', delimiter=';')
wine_target = np.loadtxt('trainingRedY.csv', delimiter=';')

clf = tree.DecisionTreeClassifier("gini", max_depth=5)
clf = clf.fit(wine_data, wine_target)

wine_data_sample = np.loadtxt('sampleDataX.csv', delimiter=';')
wine_data_target_sample = np.loadtxt('sampleDataY.csv', delimiter=';')

predictedY = clf.predict(wine_data_sample)

score = clf.score(wine_data_sample, wine_data_target_sample)
print "Score is: " + str(score)

apScore = mean_squared_error(wine_data_target_sample, predictedY)
print "Mean Error Squared: " + str(apScore)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"])
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("wine.pdf")

