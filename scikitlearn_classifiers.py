from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


#data

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#classifiers

clf_randomforest = RandomForestClassifier()
clf_tree = tree.DecisionTreeClassifier()
clf_neighbors = KNeighborsClassifier()
clf_perceptron = Perceptron()
clf_svm = SVC()

#FIT the traning data

clf_randomforest.fit(X, Y)
clf_tree.fit(X, Y)
clf_neighbors.fit(X, Y)
clf_perceptron.fit(X, Y)
clf_svm.fit(X, Y)

#test the accurecy using the same data

predict_randomforest = clf_randomforest.predict(X)
accuracy_randomforest = accuracy_score(Y, predict_randomforest) * 100

print("accuracy for RandomForest Classifier:" , accuracy_randomforest)

predict_decisiontrees = clf_tree.predict(X)
accuracy_decisiontrees = accuracy_score(Y, predict_decisiontrees) * 100

print("accuracy for Decision Trees :" , accuracy_decisiontrees)

predict_neighbors = clf_neighbors.predict(X)
accuracy_neighbors = accuracy_score(Y, predict_neighbors)

print("accuracy for KNeighborsClassifier :", accuracy_neighbors)

predict_perceptron = clf_perceptron.predict(X)
accuracy_perceptron = accuracy_score(Y, predict_perceptron)

print("accuracy for perceptron :", accuracy_perceptron)

predict_svm = clf_svm.predict(X)
accuracy_svm = accuracy_score(Y, predict_svm)

print("accuracy for svm :", accuracy_svm)

#the best classifier 

best = np.argmax([accuracy_randomforest, accuracy_decisiontrees,
 accuracy_neighbors, accuracy_perceptron, accuracy_svm])

classifiers = {0: 'RandomForest', 1: 'DecisionTrees', 2: 'KNN', 3: 'Perceptron', 
4: 'SVM'}
print('Best gender classifier is {}'.format(classifiers[best]))




