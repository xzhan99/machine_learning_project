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
    vocab = list(range(0, 784))
    return vocab


def set_of_pixel_to_vector(vocab, pic_data):
    return_vector = [0] * len(vocab)
    for i, pixel in enumerate(pic_data):
        if pixel > 0:
            return_vector[i] = 1
    return return_vector


def train_naive_bayes(train_data, label_data, num_pic, num_pixel):
    label_count = np.zeros(10)
    for i in label_data:
        label_count[i] += 1
    label_per = label_count / num_pic
    # 拉普拉斯平滑
    pn = np.ones(shape=[10, num_pixel])
    pd = np.array([2] * 10)
    # p0n = p1n = p2n = p3n = p4n = p5n = p6n = p7n = p8n = p9n = np.ones(num_pixel)
    # p0_demon = p1_demon = p2_demon = p3_demon = p4_demon = p5_demon = p6_demon = p7_demon = p8_demon = p9_demon = 2
    for i in range(num_pic):
        if label_data[i] in range(0, 10):
            pn[label_data[i]] += train_data[i]
            pd[label_data[i]] += sum(train_data[i])
    pv = np.array([pn[i] / pd[i] for i in range(10)])
    return pv, label_per


def classify_naive_bayes(pic_data, pv, label_per):
    p = [sum(pic_data * pv[i]) * (label_per[i]) for i in range(10)]
    print(p)
    return p.index(max(p))


if __name__ == "__main__":
    data_set, labels = read_file()
    formed_data = reduce_demension(data_set)
    # formed_data = formed_data[:1000]
    # labels = labels[:1000]
    print(formed_data.shape)
    num_pic, num_pixel = formed_data.shape
    vocab_matrix = create_vocab_list(formed_data, num_pixel)
    # print(vocab_matrix)
    train_matrix = []
    for pic in formed_data:
        train_matrix.append(set_of_pixel_to_vector(vocab_matrix, pic))
    # print(train_matrix[0])
    pv, label_per = train_naive_bayes(train_matrix, labels, num_pic, num_pixel)
    print(pv.shape)

    #test
    count = 0
    for i in range(100):
        pic_v = set_of_pixel_to_vector(vocab_matrix, formed_data[1])
        result = classify_naive_bayes(pic_v, pv, label_per)
        if result == labels[i]:
            print(True)
            count += 1
        else:
            print(False)
    print(count/100)



