import h5py

import numpy as np
from scipy.optimize import minimize


def read_file():
    # folder_path = '/Users/andrewzhan/Downloads/'
    folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
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


def lost(w, X, y, learning_rate):
    m = X.shape[1]
    z = np.dot(X, w.T)
    exp_z = sigmoid(z)
    the_lost = np.sum(np.multiply(y, np.log(exp_z)) + np.multiply(1 - y, np.log(1 - exp_z))) / (-m)
    the_lost = np.squeeze(the_lost)

    # print(the_cost)
    if np.isnan(the_lost):
        return np.inf
    return the_lost


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    print(X)
    y = np.matrix(y)
    m = len(X)

    # parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / m).T + ((learningRate / m) * theta)
    print(grad.shape)

    # intercept gradient is not regularized
    print(X[:, 0])
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / m
    print(grad.shape)
    return np.array(grad).ravel()


def logistic_loss_and_grad(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    temp = -lost(theta, X, y) + y
    print('temp: ', temp.shape)
    dloss = -np.sum(np.multiply(temp, X), axis=0)
    return dloss


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]  # X的行数
    params = X.shape[1]  # X的列数

    # 对于k个分类器，构建k*（n+1）维向量组 (10, 785)
    all_theta = np.zeros((num_labels, params + 1))

    # 在X第一列之前插入一列全为1的列向量作为常数项
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # 对于y若将某个类别i拿出来之后剩下的类别构成一类
    for i in range(0, num_labels):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # 采用梯度下降法最小化目标函数（cost）
        # fmin = minimize(fun=lost, x0=theta, args=(X, y_i), method='TNC', jac=logistic_loss_and_grad)
        fmin = minimize(fun=lost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
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

    data = {'X': train_set_x, 'y': train_set_y}

    all_theta = one_vs_all(data['X'], data['y'], 10, 0.000001)

    y_pred = predict_all(test_set_x, all_theta)
    # for a in zip(y_pred, test_set_y):
    #     print(a)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, test_set_y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))
