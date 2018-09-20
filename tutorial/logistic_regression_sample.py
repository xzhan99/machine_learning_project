import h5py

import numpy as np
from scipy.optimize import minimize


def read_file():
    folder_path = '/Users/andrewzhan/Downloads/'
    # folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
    with h5py.File(folder_path + 'images_training.h5', 'r') as H:
        train_set_x_orig = np.copy(H['data'])
        train_set_x_orig = train_set_x_orig / 255.
    with h5py.File(folder_path + 'labels_training.h5', 'r') as H:
        train_set_y_orig = np.copy(H['label'])
        # train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])

    with h5py.File(folder_path + 'images_testing.h5', 'r') as H:
        test_set_x_orig = np.copy(H['data'])[:2000]
        test_set_x_orig = test_set_x_orig / 255.
    with h5py.File(folder_path + 'labels_testing_2000.h5', 'r') as H:
        test_set_y_orig = np.copy(H['label'])
        # test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lost(theta, x, y):
    m = x.shape[1]
    z = np.dot(x, theta.T)
    exp_z = sigmoid(z)

    the_lost = np.sum(np.multiply(y, np.log(exp_z)) + np.multiply(1 - y, np.log(1 - exp_z))) / (-m)
    the_lost = np.squeeze(the_lost)
    # print(the_cost)

    if np.isnan(the_lost):
        return np.inf
    return the_lost


def gradient_descent(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)    # (n, 785)
    y = np.matrix(y)    # (n, 1)

    z = np.dot(x, theta.T)  #()
    exp_z = sigmoid(z)  # (n, 1)

    dloss = np.sum(np.multiply((exp_z - y), x), axis=0)    # (1, 785)

    return dloss.ravel()


def one_vs_all(x, y, num_labels):
    num_of_data = x.shape[0]
    num_of_param = x.shape[1]

    # 对于k个分类器，构建k*（n+1）维向量组 (10, 785)
    all_theta = np.random.random((num_labels, num_of_param + 1))

    # 在X第一列之前插入一列全为1的列向量作为常数项
    x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)

    # 对于y若将某个类别i拿出来之后剩下的类别构成一类
    for i in range(0, num_labels):
        theta = np.zeros(num_of_param + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (num_of_data, 1))

        fmin = minimize(fun=lost, x0=theta, args=(x, y_i), method='TNC', jac=gradient_descent)
        all_theta[i, :] = fmin.x

    return all_theta


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]
    # 与前相同，插入一列全部为1的列向量
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # 转换为矩阵
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # 计算每个训练实例上每个类的类概率
    h = sigmoid(X * all_theta.T)
    # 选取最高的那个概率为该实例的预测数字标签并构建数组
    h_argmax = np.argmax(h, axis=1)
    return h_argmax


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y = read_file()
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1)
    num = 100
    train_set_x = train_set_x[:num]
    train_set_y = train_set_y[:num]
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)

    all_theta = one_vs_all(train_set_x, train_set_y, 10)

    y_pred = predict_all(test_set_x, all_theta)
    # for a in zip(y_pred, test_set_y):
    #     print(a)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, test_set_y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
