from sklearn import datasets, model_selection
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    iris = datasets.load_iris()  # scikit-learn 自带的 iris 数据集
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)


maxdepth = 40

X_train, X_test, y_train, y_test = load_data()
depths = np.arange(1, maxdepth)
training_scores = []
testing_scores = []
for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    training_scores.append(clf.score(X_train, y_train))
    testing_scores.append(clf.score(X_test, y_test))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(depths, training_scores, label="traing score", marker='o')
ax.plot(depths, testing_scores, label="testing score", marker='*')
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Decision Tree Classification")
ax.legend(framealpha=0.5, loc='best')
plt.show()
