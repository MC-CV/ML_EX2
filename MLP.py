import numpy as np
import math
import torch
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plot
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.model_selection import KFold
from visdom import Visdom

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    for idx,i in enumerate(x):
        one_hot[idx, i] = 1
    return one_hot

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class MultilayerPerceptron():
    def __init__(self, n_hidden, n_iterations=200, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()

    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        _, n_outputs = y.shape
        # import pdb;pdb.set_trace()

        limit   = 1 / math.sqrt(n_features)
        self.W  = np.random.uniform(-limit, limit, (n_features, self.n_hidden)) # 108,16
        # self.w0 = np.zeros((1, self.n_hidden)) # 1,16

        limit   = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs)) # 16,2
        # self.v0 = np.zeros((1, n_outputs)) # 1,2
        # import pdb;pdb.set_trace()
    def fit(self, X, y):

        self._initialize_weights(X, y)

        for i in range(self.n_iterations):

            hidden_input = X.dot(self.W)
            hidden_output = self.hidden_activation(hidden_input)

            output_layer_input = hidden_output.dot(self.V)
            y_pred = self.output_activation(output_layer_input)

            viz.line([self.loss.acc(y, y_pred)], [i], win='accuracy', update='append')
            viz.line([max(sum(self.loss.loss(y, y_pred)))/len(X)], [i], win='train_loss', update='append')
            # print(max(sum(self.loss.loss(y, y_pred)))/len(X))
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            grad_v = hidden_output.T.dot(grad_wrt_out_l_input)
            # grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            # import pdb;pdb.set_trace()

            grad_wrt_hidden_l_input = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            # grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            self.V  -= self.learning_rate * grad_v
            # print(self.V)
            # self.v0 -= self.learning_rate * grad_v0
            self.W  -= self.learning_rate * grad_w
            # self.w0 -= self.learning_rate * grad_w0
            # print(self.W)
        # print(self.W,self.V)
    def predict(self, X):
        hidden_input = X.dot(self.W)
        hidden_output = self.hidden_activation(hidden_input)
        output_layer_input = hidden_output.dot(self.V)
        y_pred = self.output_activation(output_layer_input)
        return y_pred


def main():

    X_train = pd.read_csv(r'data1forEx\train1_icu_data.csv')
    y_train = pd.read_csv(r'data1forEx\train1_icu_label.csv')
    X_test = pd.read_csv(r'data1forEx\test1_icu_data.csv')
    y_test = pd.read_csv(r'data1forEx\test1_icu_label.csv')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)


    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)

    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test=scaler.transform(X_test)

    # normalizer=Normalizer(norm='l2')
    # X_train = normalizer.transform(X_train)
    # X_test = normalizer.transform(X_test)

    X_train = np.insert(X_train, 0, 1, axis=1)
    X_test = np.insert(X_test, 0, 1, axis=1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # import pdb;pdb.set_trace()


    # MLP
    clf = MultilayerPerceptron(n_hidden=32,
        n_iterations=500,
        learning_rate=0.0001)

    clf.fit(X_train, y_train)

    y_pred_train = np.argmax(clf.predict(X_train), axis=1)
    y_test_train = np.argmax(y_train, axis=1)

    y_pred = np.argmax(clf.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)

    # accuracy = accuracy_score(y_test_train, y_pred_train)
    # print ("train_Accuracy:", accuracy)
    # print ("train_error_rate:", 1-accuracy)
    accuracy = accuracy_score(y_test, y_pred)
    print ("test_Accuracy:", accuracy)
    print ("test_error_rate:", 1-accuracy)
    
    # plot.scatter(y_test,y_pred)
    # plot.show()

    kf = KFold(n_splits=5)
    i = 1
    for train_index, test_index in kf.split(X_train,y_train):
        X_train_1, X_test_1 = X_train[train_index], X_train[test_index]
        Y_train_1, Y_test_1 = y_train[train_index], y_train[test_index]
        clf.fit(X_train_1, Y_train_1)
        y_pred = np.argmax(clf.predict(X_test_1), axis=1)
        Y_test = np.argmax(Y_test_1, axis=1)
        print('第',i,'次:')
        accuracy = accuracy_score(Y_test, y_pred)
        print ("test_Accuracy:", accuracy)
        print ("test_error_rate:", 1-accuracy)
        i += 1


if __name__ == "__main__":
    viz = Visdom()
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0.], [0], win='accuracy', opts=dict(title='accuracy'))    
    main()