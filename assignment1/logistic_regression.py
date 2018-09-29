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

    with h5py.File(folder_path + 'images_testing.h5', 'r') as H:
        test_set_x_orig = np.copy(H['data'])[:2000]
        test_set_x_orig = test_set_x_orig / 255.
    with h5py.File(folder_path + 'labels_testing_2000.h5', 'r') as H:
        test_set_y_orig = np.copy(H['label'])
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def svd_reconstruction(x):
    n_components = 50
    U, s, Vt = np.linalg.svd(x, full_matrices=False)
    S = np.diag(s)
    x_reconstructed = U[0:U.shape[0], 0:n_components] \
        .dot(S[0:n_components, 0:n_components]) \
        .dot(Vt[0:n_components, 0:Vt.shape[1]])
    SSE = np.sum((x - x_reconstructed) ** 2)
    comp_ratio = (x.shape[1] * n_components + n_components + x.shape[0] * n_components) / (
            x.shape[1] * x.shape[0])
    print('If we choose {} dominant singular value/s;\n SSE = {} and\n compresion ratio = {}\n\n' \
          .format(n_components, SSE, np.round(comp_ratio, 10)))
    return x_reconstructed


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

    # classify by y, all y equals i in one class, all y doesn't equal i in one class
    for i in range(0, num_labels):
        theta = np.zeros(num_of_param + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = y_i.reshape(num_of_data, 1)

        size = 100
        for j in range(size, num_of_data + 1, size):
            y_j = y_i[j - size: j]
            x_j = x[j - size: j]
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


if __name__ == "__main__":
    train_set_x, train_set_y, test_set_x, test_set_y = read_file()
    train_set_x = train_set_x.reshape(train_set_x.shape[0], -1)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], -1)
    num = 30000
    train_set_x = train_set_x[:num]
    train_set_y = train_set_y[:num]
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)

    train_set_x = svd_reconstruction(train_set_x)

    all_theta = train(train_set_x, train_set_y, 10)

    results = predict(test_set_x, all_theta)
    correct_rate = [1 if y_hat == y else 0 for (y_hat, y) in zip(results, test_set_y)]
    accuracy = (sum(correct_rate) / test_set_y.shape[0])
    print('accuracy = ', accuracy * 100, '%')

    # print(list(zip(test_set_y, y_pred)))
