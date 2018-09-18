import numpy as np

import h5py


def read_file():
    # folder_path = '/Users/andrewzhan/Downloads/'
    folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
    with h5py.File(folder_path + 'images_training.h5', 'r') as H:
        train_set_x_orig = np.copy(H['data'])
    with h5py.File(folder_path + 'labels_training.h5', 'r') as H:
        train_set_y_orig = np.copy(H['label'])
        train_set_y_orig = train_set_y_orig.reshape(1, train_set_y_orig.shape[0])

    with h5py.File(folder_path + 'images_testing.h5', 'r') as H:
        test_set_x_orig = np.copy(H['data'])[:2000]
    with h5py.File(folder_path + 'labels_testing_2000.h5', 'r') as H:
        test_set_y_orig = np.copy(H['label'])
        test_set_y_orig = test_set_y_orig.reshape(1, test_set_y_orig.shape[0])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def initialize_with_zeros(dim):
    # 初始化w,b
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def propagate(w, b, X, Y):
    """
    实现前向传播的代价函数及反向传播的梯度
    输入:
    w -- 权重, 一个numpy数组，大小为(图片长度 * 图片高度 * 3, 1)
    b -- 偏差, 一个标量
    X -- 训练数据，大小为 (图片长度 * 图片高度 * 3 , 样本数量)
    Y -- 真实"标签"向量，大小为(1, 样本数量)

    输出:
    cost -- 逻辑回归的负对数似然代价函数
    dw -- 相对于w的损失梯度，因此与w的形状相同
    db -- 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]
    z = np.dot(w.T, X) + b
    A = sigmod(z)
    # print(A)
    cost = np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1 - A))) / (-m)
    cost = np.squeeze(cost)
    # print('cost = ', cost)
    # 反向传播
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    return dw, db, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    此函数通过运行梯度下降算法来优化w和b
    输入:
    w -- 权重, 一个numpy数组，大小为(图片长度 * 图片高度 * 3, 1)
    b -- 偏差, 一个标量
    X -- 训练数据，大小为 (图片长度 * 图片高度 * 3 , 样本数量)
    Y -- 真实"标签"向量，大小为(1, 样本数量)
    num_iterations -- 优化循环的迭代次数
    learning_rate -- 梯度下降更新规则的学习率
    print_cost -- 是否每100步打印一次成本
    输出:
    params -- 存储权重w和偏见b的字典
    grads -- 存储权重梯度相对于代价函数偏导数的字典
    costs -- 在优化期间计算的所有损失的列表，这将用于绘制学习曲线。
    """
    costs = []
    for i in range(num_iterations):
        # 成本和梯度计算
        dw, db, cost = propagate(w, b, X, Y)
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 每100次训练迭代打印成本
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


if __name__ == '__main__':
    train_set_x, train_set_y, test_set_x, test_set_y = read_file()
    print(train_set_x.shape)

    train_set_x_flatten = train_set_x.reshape(train_set_x.shape[0], -1).T
    test_set_x_flatten = test_set_x.reshape(test_set_x.shape[0], -1).T
    # 数据标准化
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # X_train -- 由numpy数组表示的训练集，大小为 (图片长度 * 图片高度 * 3，训练样本数)
    # Y_train -- 由numpy数组（向量）表示的训练标签，大小为 (1, 训练样本数)
    # X_test -- 由numpy数组表示的测试集，大小为（图片长度 * 图片高度 * 3，测试样本数）
    # Y_test -- 由numpy数组（向量）表示的测试标签，大小为 (1, 测试样本数)
    print()
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)
