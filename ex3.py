import sys
import numpy as np


def txt_to_np(text_file_path):
    with open(text_file_path, 'r') as f:
        txt_data = f.readlines()
        txt_data = [x.strip() for x in txt_data]
    n_samples = len(txt_data)
    samples_size = len([float(y) for y in txt_data[0].split()])
    np_data = np.zeros((n_samples, samples_size))
    for i, line in enumerate(txt_data):
        np_data[i, :] = np.array([float(y) for y in line.split()])
    return np_data


def read_data(train_x_path, train_y_path, test_x_path):
    train_x = txt_to_np(train_x_path)
    train_y = txt_to_np(train_y_path)
    test_x = txt_to_np(test_x_path)
    return train_x, train_y, test_x


def write_res():
    with open(test_y, 'w') as out:
        for i in range(test_x.shape[0]):
            out.write('{}\n'.format(results[i]))


def initialize_weights(input_size, n_hidden, n_classes, w_var, reg=0):
    W1 = w_var * np.random.randn(input_size, n_hidden)
    b1 = np.zeros(n_hidden)
    W2 = w_var * np.random.randn(n_hidden, n_classes)
    b2 = np.zeros(n_classes)
    return W1, b1, W2, b2


def train_nn(train_x, train_y, n_hidden=200, n_classes=10, T=30, lr=0.04, w_var=0.9, reg=0):
    # initialize weights
    n_train = train_x.shape[0]
    input_size = train_x.shape[1]
    W1, b1, W2, b2 = initialize_weights(input_size, n_hidden, n_classes, w_var)

    # train model
    for epoch in range(T):
        loss = 0
        for i in range(n_train):
            # forward pass
            h = np.dot(W1.T, train_x[i]) + b1
            h_relu = np.maximum(h, 0)
            scores = np.dot(W2.T, h_relu) + b2

            # calculate loss
            cur_class = int(train_y[i])
            loss += -np.log(np.exp(scores[cur_class]) / np.sum(np.exp(scores)))

            # back propagation
            dsoft = np.exp(scores) / np.sum(np.exp(scores))
            dsoft[cur_class] -= 1
            dW2 = np.expand_dims(h_relu, axis=1).dot(np.expand_dims(dsoft, axis=1).T)
            db2 = dsoft
            dW2 = dW2 + 2 * reg * W2
            dh = dsoft.dot(W2.T)

            # ReLu derivate
            dh = dh * (h_relu != 0)
            dW1 = np.expand_dims(train_x[i], axis=1).dot(np.expand_dims(dh, axis=1).T)
            db1 = dh
            dW1 = dW1 + 2 * reg * W1

            # update weights
            W1 -= lr * dW1
            W2 -= lr * dW2
            b1 -= lr * db1
            b2 -= lr * db2

        loss /= n_train
        loss += reg * np.sum(W2 * W2) + reg * np.sum(W1 * W1)
    return W1, b1, W2, b2


def test_nn(test_x, model):
    n_test = test_x.shape[0]
    results = []
    W1, b1, W2, b2 = model
    for i in range(n_test):
            h = np.dot(W1.T, test_x[i]) + b1
            h_relu = np.maximum(h, 0)
            scores = np.dot(W2.T, h_relu) + b2
            results.append(np.argmax(scores))
    return results


if __name__ == "__main__":
    train_x_path, train_y_path, test_x_path, test_y = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # data manipulations
    train_x, train_y, test_x = read_data(train_x_path, train_y_path, test_x_path)

    # normalize train_x
    for c in range(train_x.shape[1]):
        train_x_avg = np.mean(train_x[:, c])
        std = np.std(train_x[:, c])
        if std > 0:
            train_x[:, c] = (train_x[:, c] - train_x_avg) / std

    # normalize test_x
    for c in range(test_x.shape[1]):
        test_x_avg = np.mean(test_x[:, c])
        std = np.std(test_x[:, c])
        if std > 0:
            test_x[:, c] = (test_x[:, c] - test_x_avg) / std

    # train the model
    lr_list = [0.0001]
    w_var_list = [0.001]
    for lr in lr_list:
        for w_var in w_var_list:
            model = train_nn(train_x, train_y, lr=lr, w_var=w_var)
            results = test_nn(test_x, model)

    write_res()
