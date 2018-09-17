import h5py

import numpy as np
from scipy.optimize import minimize


def read_file():
    folder_path = '/Users/andrewzhan/Downloads/'
    with h5py.File(folder_path + 'images_training.h5', 'r') as H:
        data = np.copy(H['data'])
    with h5py.File(folder_path + 'labels_training.h5', 'r') as H:
        label = np.copy(H['label'])
    return data, label


def reduce_demension(data):
    new_data = []
    for pic in data:
        new_row = np.concatenate(pic, axis=None)
        new_data.append(new_row)
    return np.array(new_data)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg


def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]  # X的行数
    params = X.shape[1]  # X的列数

    # 对于k个分类器，构建k*（n+1）维向量组
    all_theta = np.zeros((num_labels, params + 1))

    # 在X第一列之前插入一列全为1的列向量作为常数项
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # 对于y若将某个类别i拿出来之后剩下的类别构成一类
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # 采用梯度下降法最小化目标函数（cost）
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

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
    # 由于该数组在训练时是基于图片的0-9而预测的，所以要+1以匹配y
    h_argmax = h_argmax + 1

    return h_argmax


if __name__ == "__main__":
    number_of_set = 1000
    data_set, labels = read_file()
    formatted_data = reduce_demension(data_set)
    formatted_data = formatted_data[:number_of_set]
    formatted_label = labels.T
    formatted_label = formatted_label[:number_of_set]
    data = {'X': formatted_data, 'y': formatted_label}

    rows = data['X'].shape[0]
    params = data['X'].shape[1]
    all_theta = np.zeros((10, params + 1))
    X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)
    theta = np.zeros(params + 1)
    y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
    y_0 = np.reshape(y_0, (rows, 1))
    X.shape, y_0.shape, theta.shape, all_theta.shape
    np.unique(data['y'])
    all_theta = one_vs_all(data['X'], data['y'], 10, 1)
    # print(all_theta)

    y_pred = predict_all(data['X'], all_theta)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print('accuracy = {0}%'.format(accuracy * 100))