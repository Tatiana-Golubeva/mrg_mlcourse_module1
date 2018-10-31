import numpy as np
import argparse
from functions import X_transformation, y_to_matrix, get_data, softmax, my_prediction, delete_zero_features
from sklearn.metrics import classification_report


def my_loss(X, y, weights):
    p = softmax(weights, X)
    summa = -np.sum(y * np.log(p))
    return summa / len(X)


def gradient(X, weights, y):
    p = softmax(weights, X)
    return np.dot(np.transpose(p - y), X) / (len(X))


def grad_step(X, y, current_weights, step):
    while True:
        current_loss = my_loss(X, y, current_weights)
        grad = gradient(X, current_weights, y)
        new_weights = current_weights - step * grad
        new_loss = my_loss(X, y, new_weights)
        if new_loss >= current_loss:
            step = step / 2
        else:
            difference = current_loss - new_loss
            return new_weights, new_loss, difference


np.random.seed(0)


def GD(X_train, y_train, num_iter, step):
    np.random.seed(0)
    w = np.random.normal(scale=0.001, size=(10, X_train.shape[1]))
    eps = 1e-7
    for i in range(num_iter):
        w, loss, difference = grad_step(X_train, y_train, w, step)
        # print (i, loss)
        if difference <= eps:
            break
    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train_dir=', dest='x_train_dir', type=str,
                        help='путь к файлу, в котором лежат рекорды обучающей выборки', required=True)
    parser.add_argument('--y_train_dir=', dest='y_train_dir', type=str,
                        help='путь к файлу, в котором лежат метки обучающей выборки', required=True)
    parser.add_argument('--model_output_dir=', dest='model_output_dir', type=str,
                        help='путь к файлу, в который скрипт сохраняет обученную модель', required=True)

    args = parser.parse_args()
    X, y = get_data(args.x_train_dir, args.y_train_dir)
    y = y.reshape(-1)
    number_of_classes = 10
    y1 = y_to_matrix(y, number_of_classes)
    X1 = X_transformation(X)
    zero_idx, X1 = delete_zero_features(X1)
    w = GD(X1, y1, 1000, 0.5)
    np.save(args.model_output_dir, w)
    np.save('zero_ind.npy', zero_idx)
    y_pred = my_prediction(X1, w)
    y_pred_matrix = y_to_matrix(y_pred, number_of_classes)
    print(classification_report(y1, y_pred_matrix))
