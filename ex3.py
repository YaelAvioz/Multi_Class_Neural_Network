import sys
import numpy as np
from tqdm import tqdm


def txt_to_np(text_file_path):
    with open(text_file_path, 'r') as f:
        txt_data = f.readlines()
        txt_data = [x.strip() for x in txt_data]
    num_data = []
    for line in txt_data:
        num_data.append([float(y) for y in line.split()])
    np_data = np.array(num_data)
    return np_data


def read_data(train_x_path, train_y_path, test_x_path):
    train_x = txt_to_np(train_x_path)
    train_y = txt_to_np(train_y_path)
    test_x = txt_to_np(test_x_path)
    return train_x, train_y, test_x


def read_data_fast(train_x_path, train_y_path, test_x_path):
    # TODO: delete this function before submission
    # TODO: validate that everything works with normal functions
    train_x = np.load('new_format/' + train_x_path + '.npy')
    train_y = np.load('new_format/' + train_y_path + '.npy')
    test_x = np.load('new_format/' + test_x_path + '.npy')
    return train_x, train_y, test_x


def normalize_data(data, mean=None, std=None):
    if not mean:
        mean = np.mean(data)
    if not std:
        std = np.sqrt(np.var(data))
    data -= mean
    data /= std
    return data, mean, std


def initialize_weights(input_size, n_hidden, n_classes, w_var, reg=0):
    W1 = w_var * np.random.randn(input_size, n_hidden)
    b1 = np.zeros(n_hidden)
    W2 = w_var * np.random.randn(n_hidden, n_classes)
    b2 = np.zeros(n_classes)
    return W1, b1, W2, b2


def train_nn(train_x, train_y, n_hidden=10, n_classes=10, T=1, lr=0.5, w_var=0.1, reg=0):
    # initialize weights
    print(f"w_var is: {w_var}. lr is: {lr}")
    n_train = train_x.shape[0]
    input_size = train_x.shape[1]
    W1, b1, W2, b2 = initialize_weights(input_size, n_hidden, n_classes, w_var)

    # train model
    for epoch in range(T):
        loss = 0
        for i in tqdm(range(n_train)):
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
        print(f'epoch {epoch}: current loss is:{loss}')
    return W1, b1, W2, b2


def test_nn(test_x, model):
    n_test = test_x.shape[0]
    results = []
    W1, b1, W2, b2 = model
    for i in tqdm(range(n_test)):
            # forward pass
            h = np.dot(W1.T, test_x[i]) + b1
            h_relu = np.maximum(h, 0)
            scores = np.dot(W2.T, h_relu) + b2
            results[i].append(n.argmax(scores))
    return results


if __name__ == "__main__":
    train_x_path, train_y_path, test_x_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    # data manipulations
    train_x, train_y, test_x = read_data_fast(train_x_path, train_y_path, test_x_path)
    train_x, mean, std = normalize_data(train_x)
    test_x, _, _ = normalize_data(test_x, mean=mean, std=std)

    # train the model
    lr_list = [0.0001]
    w_var_list = [0.001]
    for lr in lr_list:
        for w_var in w_var_list:
            model = train_nn(train_x, train_y, lr=lr, w_var=w_var)
            results = test_nn(test_x, model)
    # test the model
    # TODO: test on test data
