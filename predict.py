import numpy as np
import argparse
from functions import X_transformation, y_to_matrix, get_data, my_prediction, softmax
from sklearn.metrics import classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test_dir=', dest='x_test_dir', type=str,
                        help='путь к файлу, в котором лежат рекорды тестовой выборки', required=True)
    parser.add_argument('--y_test_dir=', dest='y_test_dir', type=str,
                        help='путь к файлу, в котором лежат метки тестовой выборки', required=True)
    parser.add_argument('--model_input_dir=', dest='model_input_dir', type=str,
                        help='путь к файлу, из которого скрипт считывает обученную модель', required=True)

    args = parser.parse_args()
    weights = np.load(args.model_input_dir)
    idx_to_del = np.load('zero_ind.npy')
    X, y = get_data(args.x_test_dir, args.y_test_dir)
    y = y.reshape(-1)
    number_of_classes = 10
    y1 = y_to_matrix(y, number_of_classes)
    X1 = X_transformation(X)
    X1 = np.delete(X1, idx_to_del, axis=1)
    y_pred = my_prediction(X1, weights)
    y_pred_matrix = y_to_matrix(y_pred, number_of_classes)
    print(classification_report(y1, y_pred_matrix))
