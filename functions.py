import gzip
import numpy as np
from struct import unpack


# this function has been taken from https://martin-thoma.com/classify-mnist-with-pybrain/
def get_data(im_file, label_file):
    images = gzip.open(im_file, 'rb')
    labels = gzip.open(label_file, 'rb')

    images.read(4)

    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]

    rows = images.read(4)
    rows = unpack('>I', rows)[0]

    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    labels.read(4)

    N = labels.read(4)
    N = unpack('>I', N)[0]

    x = np.zeros((N, rows, cols), dtype=np.float32)
    y = np.zeros((N, 1), dtype=np.uint8)

    for i in range(N):
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return x, y


def y_to_matrix(y, number_of_classes):
    y1 = np.zeros((len(y), number_of_classes))
    for ind, elem in enumerate(y):
        y1[ind][elem] = 1.
    return y1


def X_transformation(X):
    rows_len, cols_len = X[0].shape
    X1 = np.zeros((len(X), rows_len * cols_len))
    for i in range(len(X)):
        X1[i] = np.ravel(X[i])
    X1 = X1 / 255.
    e = np.ones(X1.shape[0])
    X1 = np.hstack([X1, e.reshape(-1, 1)])
    return X1


def delete_zero_features(X):
    zero_idx = np.argwhere(np.all(X[..., :] == 0, axis=0))
    X1 = np.delete(X, zero_idx, axis=1)
    return  zero_idx, X1

def softmax(weights, X):
    d = np.exp(np.dot(X, np.transpose(weights)))
    summ = np.sum(d, axis=1)
    for i in range(len(weights)):
        d[:, i] = d[:, i] / summ
    return d


def my_prediction(X, weights):
    p = softmax(weights, X)
    return np.argmax(p, axis=1)
