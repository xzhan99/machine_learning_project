import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as pl



def read_file():
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                 "result"]
    adult = pd.read_csv('/Users/andrewzhan/Downloads/adult.csv', names=col_names)
    adult_test = pd.read_csv('/Users/andrewzhan/Downloads/adult_test.csv', names=col_names, header=0)
    return adult, adult_test


def reformat(set):
    set = set.replace(regex=[r'<=50K\.'], value='<=50K')
    set = set.replace(regex=[r'>50K\.'], value='>50K')
    set['label'] = set.result.apply(lambda x: 1 if '<=50K' in x else 0)
    # print(set)
    return set


if __name__ == '__main__':
    train_set, test_set = read_file()
    print(train_set.shape)
    print(test_set.shape)
    train_set = reformat(train_set)
    print('-------------------------------')
    test_set = reformat(test_set)
    print(train_set.shape)
    print(test_set.shape)

    train_set_y = train_set['label']
    train_set_x = train_set[["age", "workclass", "education", "education-num", "marital-status", "occupation",
                             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                             "native-country"]]
    test_set_y = test_set['label']
    test_set_x = test_set[["age", "workclass", "education", "education-num", "marital-status", "occupation",
                           "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                           "native-country"]]
    dict_vect = DictVectorizer(sparse=False)
    train_set_x = dict_vect.fit_transform(train_set_x.to_dict(orient='record'))
    test_set_x = dict_vect.transform(test_set_x.to_dict(orient='record'))

    maxdepth = 40
    depths = np.arange(1, maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(train_set_x, train_set_y)
        training_scores.append(clf.score(train_set_x, train_set_y))
        testing_scores.append(clf.score(test_set_x, test_set_y))

    fig = pl.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(depths, training_scores, label="traing score", marker='o')
    ax.plot(depths, testing_scores, label="testing score", marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5, loc='best')
    pl.show()
