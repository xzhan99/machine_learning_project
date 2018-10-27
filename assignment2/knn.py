import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

# 数据读取----------
col_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
             "result"]
adult = pd.read_csv('/Users/andrewzhan/Downloads/adult.csv', names=col_names)
adult_test = pd.read_csv('/Users/andrewzhan/Downloads/adult_test.csv', names=col_names, header=0)

# 数据处理----------
adult_clean = adult.replace(regex=[r'\?|\.|\$'], value=np.nan)
adult = adult_clean.dropna(how='any')
adult = adult.drop(['fnlwgt'], axis=1)
adult['label'] = adult.result.apply(lambda x: 1 if '<=50K' in x else 0)

adult_clean_test = adult_test.replace(regex=[r'\?|\$'], value=np.nan)
adult_test = adult_clean_test.dropna(how='any')
adult_test = adult_test.drop(['fnlwgt'], axis=1)
adult_test['label'] = adult_test.result.apply(lambda x: 1 if '<=50K.' in x else 0)
adult['label'] = adult.result.apply(lambda x: 1 if '<=50K' in x else 0)

train_set_y = adult['label']
train_set_x = adult[["age", "workclass", "education", "education-num", "marital-status", "occupation",
                     "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]]

test_set_y = adult_test['label']
test_set_x = adult_test[["age", "workclass", "education", "education-num", "marital-status", "occupation",
                         "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week",
                         "native-country"]]

dict_vect = DictVectorizer(sparse=False)
# print(train_set_x.head(1))

train_set_x = dict_vect.fit_transform(train_set_x.to_dict(orient='record'))
test_set_x = dict_vect.transform(test_set_x.to_dict(orient='record'))

import matplotlib.pyplot as plt  # 可视化模块

# 建立测试参数集
k_range = range(1, 15)

k_scores = []
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`')
# 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = neighbors.KNeighborsClassifier(n_neighbors=2 * k - 1)
    k_scores.append(cross_val_score(knn, train_set_x, train_set_y, cv=10).mean())
    print('11111111111111111111111111111111111111')

# 可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

# knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
# print(cross_val_score(knn,train_set_x,train_set_y,cv=10).mean())
# print('11111111111111111111111111111111111111111111111111111111111111')
# knn.fit(train_set_x, train_set_y)
# res = knn.predict(test_set_x)  # 对测试集进行预测
# print(res.shape)
#
# error_num = np.sum(res != test_set_y)  # 统计分类错误的数目
# num = len(test_set_x)  # 测试集的数目
# print("Total num:", num, " Wrong num:", error_num, "  RightRate:", 1 - (error_num / float(num)))
