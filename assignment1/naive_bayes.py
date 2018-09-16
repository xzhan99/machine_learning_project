import h5py
import numpy as np


def read_file():
    folder_path = 'C:\\Users\\18701\\Desktop\\machine learning\\'
    with h5py.File(folder_path + 'images_training.h5', 'r') as H:
        data = np.copy(H['data'])
    with h5py.File(folder_path + 'labels_training.h5', 'r') as H:
        label = np.copy(H['label'])
    print(data[0].shape)
    print(label)


if __name__ == "__main__":
   read_file()