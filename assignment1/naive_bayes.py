import sys

import h5py
import numpy as np


def read_file():
    folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
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


def create_vocab_list(data, num_column):
    # build a set for showing all the possible value for every pixel
    # the shape of the set is [784, 256]
    vocab = []
    for i in range(num_column):
        grey_scales = list(set(data[:, i]))
        # print(grey_scales)
        vocab.append(grey_scales)
    return vocab


def set_of_pixel_to_vector(vocab, pic_data):
    return_vector = []
    for i, pixel in enumerate(pic_data):
        vocab_row = vocab[i]
        vec = np.zeros(len(vocab_row))
        if pixel in vocab_row:
            vec[vocab_row.index(pixel)] = 1
        return_vector.append(vec)
    return return_vector


def train_naive_bayes(train_data, label_data, num_pic):
    label_count = np.zeros(10)
    for i in label_data:
        label_count[i] += 1
    label_count = label_count / num_pic
    print(label_count)
    #拉普拉斯平滑
    p0 = p1 = p2 = p3 = p4 = p5 = p6 = p7 = p8 = p9 = np.ones(256)
    p0_demon = p1_demon = p2_demon = p3_demon = p4_demon = p5_demon = p6_demon = p7_demon = p8_demon = p9_demon = 2
    for i in range(num_pic):
        if label_data[i] == 0:
            



if __name__ == "__main__":
    data_set, labels = read_file()
    formed_data = reduce_demension(data_set)
    formed_data = formed_data[:1000]
    labels = labels[:1000]
    print(formed_data.shape)
    num_pic, num_pixel = formed_data.shape
    vocab_matrix = create_vocab_list(formed_data, num_pixel)
    # print(vocab_matrix)
    train_matrix = []
    for pic in formed_data:
        train_matrix.append(set_of_pixel_to_vector(vocab_matrix, pic))
    train_naive_bayes(train_matrix, labels, num_pic)



