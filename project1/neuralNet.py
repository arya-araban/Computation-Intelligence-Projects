from sklearn import metrics
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from project1.misc import load_ds


class NN:

    def __init__(self, num_hidden_layers):
        self.num_hidden_layers = num_hidden_layers

    def train_and_predict(self, x_train, y_train, x_test, y_test):
        mlp = MLPClassifier(hidden_layer_sizes=(50, self.num_hidden_layers + 2), max_iter=10, alpha=1e-4,
                            solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

        mlp.fit(x_train, y_train)

        return mlp.score(x_test, y_test)


if __name__ == "__main__":
    alg1 = NN(2)
    xTrain, xTest, yTrain, yTest = load_ds()
    y_predict = alg1.train_and_predict(xTrain, yTrain, xTest, yTest)
    print(y_predict)
