import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as pl

text_fields = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex",
               "native-country"]


def read_file():
    col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                 "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                 "result"]
    adult = pd.read_csv('/Users/andrewzhan/Downloads/adult.csv', names=col_names)
    adult_test = pd.read_csv('/Users/andrewzhan/Downloads/adult_test.csv', names=col_names, header=0)
    return adult, adult_test


def reformat(data_set):
    data_set = data_set.replace(regex=[r'<=50K\.'], value='<=50K')
    data_set = data_set.replace(regex=[r'>50K\.'], value='>50K')
    data_set['label'] = data_set.result.apply(lambda x: 1 if '<=50K' in x else 0)
    set_y = data_set['label']
    set_x = data_set[
        ["age", "workclass", "education", "education-num", "marital-status", "occupation", "relationship", "race",
         "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]]
    set_x = replace_nan(set_x)
    return set_x, set_y


def replace_nan(set_x):
    set_x = set_x.replace(regex=[r'\?|\.|\$'], value=np.nan)
    set_x = set_x.fillna(set_x.mean())
    mf_dic = most_frequent_text(set_x)
    for field in text_fields:
        set_x[field] = set_x[field].replace(np.nan, mf_dic[field])
    return set_x


def most_frequent_text(dataset):
    most_frequent = {}
    for field in text_fields:
        col = dataset[field]
        col_count = {}
        for str in col:
            if str in col_count:
                col_count[str] += 1
            else:
                col_count[str] = 0
        col_count = sorted(col_count.items(), key=lambda x: x[1], reverse=True)
        most_frequent[field] = col_count[0][0]
    print(most_frequent)
    return most_frequent


def split_train_set(data_set):
    data_set['label'] = data_set.result.apply(lambda x: 1 if '<=50K' in x else 0)

    splitted_set = []
    for i in range(10):
        sample = train_set.sample(frac=0.1, replace=False)
        splitted_set.append(sample)

    return splitted_set


def reform_train_test_set(splitted_set, test_pos):
    test = splitted_set[test_pos]
    train = pd.DataFrame(columns=["age", "workclass", "education", "education-num", "marital-status", "occupation",
                                  "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                                  "native-country"])
    for i in range(10):
        if i != test_pos:
            train = pd.concat([train, splitted_set[i]], ignore_index=True, sort=False)
    return train, test


if __name__ == '__main__':
    train_set, final_test_set = read_file()

    train_set_x, train_set_y = reformat(train_set)
    final_test_set_x, final_test_set_y = reformat(final_test_set)

    dict_vect = DictVectorizer(sparse=False)
    train_set_x = dict_vect.fit_transform(train_set_x.to_dict(orient='record'))
    final_test_set_x = dict_vect.transform(final_test_set_x.to_dict(orient='record'))

    classifier = DecisionTreeClassifier(max_depth=11)
    classifier.fit(train_set_x, train_set_y)
    kf = KFold(n_splits=10)
    result = cross_val_score(classifier, train_set_x, train_set_y, cv=kf, scoring='accuracy')
    print(result.mean())
    print(classifier.score(final_test_set_x, final_test_set_y))

    # splitted_train_set = split_train_set(train_set)
    # scores = []
    # for i in range(10):
    #     train_set, test_set = reform_train_test_set(splitted_train_set, i)
    #     train_set_x, train_set_y = reformat(train_set)
    #     test_set_x, test_set_y = reformat(test_set)
    #     final_test_set_x, final_test_set_y = reformat(final_test_set)
    #
    #     dict_vect = DictVectorizer(sparse=False)
    #     train_set_x = dict_vect.fit_transform(train_set_x.to_dict(orient='record'))
    #     test_set_x = dict_vect.transform(test_set_x.to_dict(orient='record'))
    #     final_test_set_x = dict_vect.transform(final_test_set_x.to_dict(orient='record'))
    #
    #     # train_set_x = svd_reconstruction(train_set_x)
    #     # test_set_x = svd_reconstruction(test_set_x)
    #     # final_test_set_x = svd_reconstruction(final_test_set_x)
    #
    #     classifier = DecisionTreeClassifier(max_depth=11)
    #     classifier.fit(train_set_x, train_set_y)
    #     scores.append({'i': i, 'score': classifier.score(test_set_x, test_set_y)})
    # print(scores)
    # scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    # print(scores)
    # print('aver = ', sum([x['score'] for x in scores]) / 10)
    # print('final = ', classifier.score(final_test_set_x, final_test_set_y))

    # print(classifier.score(train_set_x, train_set_y))
    # print(classifier.score(test_set_x, test_set_y))
