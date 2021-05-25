from sklearn.neural_network import MLPClassifier

from project1.misc import load_ds


class NN:

    def __init__(self, num_hidden_layers, num_hidden_units):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_units = num_hidden_units

    def train_and_predict(self, x_train, y_train, x_test, y_test):
        mlp = MLPClassifier(hidden_layer_sizes=(self.num_hidden_units, self.num_hidden_layers + 2), max_iter=10, alpha=1e-4,
                            solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)

        mlp.fit(x_train, y_train)
        score = mlp.score(x_test, y_test)
        return (f"{score * 100}%")


if __name__ == "__main__":
    alg1 = NN(2,500)
    xTrain, xTest, yTrain, yTest = load_ds()
    y_predict = alg1.train_and_predict(xTrain, yTrain, xTest, yTest)
    print(f"score: {y_predict}")
