import h5py

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as pl


def read_file():
    # folder_path = '/Users/andrewzhan/Downloads/'
    folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
    with h5py.File(folder_path + 'images_training.h5', 'r') as H:
        train_set_x_orig = np.copy(H['data'])
        train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
    with h5py.File(folder_path + 'labels_training.h5', 'r') as H:
        train_set_y_orig = np.copy(H['label'])

    with h5py.File(folder_path + 'images_testing.h5', 'r') as H:
        test_set_x_orig = np.copy(H['data'])[:2000]
        test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)
    with h5py.File(folder_path + 'labels_testing_2000.h5', 'r') as H:
        test_set_y_orig = np.copy(H['label'])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def normalisation(data, minimum, maximum):
    return (data - minimum) / (maximum - minimum)


def svd_reconstruction(x):
    n_components = 50
    U, s, Vt = np.linalg.svd(x, full_matrices=False)
    S = np.diag(s)
    x_reconstructed = U[0:U.shape[0], 0:n_components].dot(S[0:n_components, 0:n_components]).dot(
        Vt[0:n_components, 0:Vt.shape[1]])
    SSE = np.sum((x - x_reconstructed) ** 2)
    print(x.shape[1], ' ', x.shape[0])
    comp_ratio = (x.shape[1] * n_components + n_components + x.shape[0] * n_components) / (x.shape[1] * x.shape[0])

    print(s[0])
    pl.figure(figsize=(15, 10))
    pl.subplot(111)
    pl.plot(np.arange(len(s)), s)
    pl.grid()
    pl.title('Singular values distribution')
    pl.xlabel('n')
    pl.show()

    return x_reconstructed


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lost(theta, x, y):
    m = x.shape[1]
    z = np.dot(x, theta.T)
    exp_z = sigmoid(z)

    the_lost = np.sum(np.multiply(y, np.log(exp_z)) + np.multiply(1 - y, np.log(1 - exp_z))) / (-m)
    # print(the_cost)

    # regularisation
    the_lambda = np.math.e ** -27
    the_lost = the_lost + 1 / 2 * the_lambda * theta.dot(theta.T)

    if np.isnan(the_lost):
        return np.inf
    return the_lost


def gradient_descent(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)  # (n, 785)
    y = np.matrix(y)  # (n, 1)

    z = np.dot(x, theta.T)  # ()
    exp_z = sigmoid(z)  # (n, 1)

    dloss = np.sum(np.multiply((exp_z - y), x), axis=0)  # (1, 785)

    return dloss


def train(x, y, num_labels):
    num_of_data = x.shape[0]
    num_of_param = x.shape[1]

    # build k classifiers, all_theta is a (10, 785) matrix
    all_theta = np.random.random((num_labels, num_of_param + 1))

    # insert an all one column as the first column
    x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)

    # classify by y, separately train 10 classifiers
    for i in range(0, num_labels):
        theta = np.zeros(num_of_param + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(num_of_data, 1)

        # for improving speed performance, train 100 data each time
        size = 100
        for j in range(size, num_of_data + 1, size):
            y_j = y_i[j - size: j]
            x_j = x[j - size: j]
            # minimize a scalar function using a truncated Newton (TNC) algorithm
            theta_min = minimize(fun=lost, x0=theta, args=(x_j, y_j), method='TNC', jac=gradient_descent)
            # print(theta_min.success)
            theta = theta_min.x
        all_theta[i, :] = theta_min.x

    return all_theta


def predict(x, all_theta):
    num_of_data = x.shape[0]
    # insert an all one column as the first column, same with training process
    x = np.insert(x, 0, values=np.ones(num_of_data), axis=1)
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)

    # calculate the possibility of every class
    possibilities = sigmoid(x.dot(all_theta.T))

    # choose the highest one
    result = np.argmax(possibilities, axis=1)

    return result


def show_pic(pic1, pic2):
    pic1 = pic1.reshape(28, 28)
    pic2 = pic2.reshape(28, 28)
    pl.figure(figsize=(15, 10))  # figsize=(15,10)
    pl.subplot(121)
    pl.imshow(pic1, cmap=pl.cm.gray)
    pl.title('Original image')
    pl.subplot(122)
    pl.imshow(pic2, cmap=pl.cm.gray)
    pl.title('Compressed image')
    pl.show()


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y = read_file()
    max_grey = np.max(train_set_x)
    min_grey = np.min(train_set_x)
    train_set_x = normalisation(train_set_x, min_grey, max_grey)
    test_set_x = normalisation(test_set_x, min_grey, max_grey)

    num = 30000
    train_set_x = train_set_x[:num]
    train_set_y = train_set_y[:num]
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)

    pic_ori = train_set_x[0]
    train_set_x = svd_reconstruction(train_set_x)
    # test_set_x = svd_reconstruction(test_set_x)
    # show_pic(pic_ori, train_set_x[0])

    all_theta = train(train_set_x, train_set_y, 10)

    results = predict(test_set_x, all_theta)
    correct_rate = [1 if y_hat == y else 0 for (y_hat, y) in zip(results, test_set_y)]
    accuracy = (sum(correct_rate) / test_set_y.shape[0])
    print('accuracy = ', accuracy * 100, '%')

    # print(list(zip(test_set_y, y_pred)))
