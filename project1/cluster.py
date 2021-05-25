from sklearn import metrics
from sklearn.metrics import accuracy_score, adjusted_rand_score
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from misc import load_ds


class KMeans:
    num_seeds = 0  # greater than or equal to 10

    def plot_k_means(self, x_train, labels):
        pca = PCA(2)
        df = pca.fit_transform(x_train)
        u_labels = np.unique(labels)

        for i in u_labels:
            plt.scatter(df[labels == i, 0], df[labels == i, 1], label=i)
        plt.legend()
        plt.show()

    def __retrieve_info(self, cluster_labels, y_train):
        reference_labels = {}
        # For loop to run through each label of cluster label
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)  # wherever cluster_label is i
            num = np.bincount(y_train[index == 1]).argmax()
            reference_labels[i] = num
        return reference_labels

    def print_metrics(self, y_pred, y_train):
        print(f"rand index: {adjusted_rand_score(y_pred, y_train) * 100 :.2f}%")
        # print(f"accuracy: {accuracy_score(y_pred, y_train) * 100 :.2f}%")
        contingency_matrix = metrics.cluster.contingency_matrix(y_train, y_pred)
        purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
        print(f"purity: {purity * 100 :.2f}%")

    def train_and_predict(self, x_train, y_train, x_test):
        model = MiniBatchKMeans(n_clusters=self.num_seeds)
        model.fit(x_train)

        labels = model.labels_
        mapping = self.__retrieve_info(labels, y_train)

        prediction = model.predict(x_test)

        # plot_k_means(x_train,labels)

        # print(mapping)
        # print(labels)
        # print(y_train)

        number_labels = np.zeros(len(prediction))
        for i in range(len(prediction)):  # performing the mapping for prediction
            number_labels[i] = mapping[prediction[i]]
        y_prediction = number_labels.astype('int')

        """IN CASE WE WANT PREDICTION FOR X_TEST """
        # number_labels = np.zeros(len(labels))
        # for i in range(len(labels)):  # performing the mapping for prediction
        #     number_labels[i] = mapping[labels[i]]
        # y_prediction = number_labels.astype('int')

        return y_prediction

    def __init__(self, num_seeds):
        self.num_seeds = num_seeds


# main program:
if __name__ == "__main__":
    alg1 = KMeans(256)
    xTrain, xTest, yTrain, yTest = load_ds()
    y_predict = alg1.train_and_predict(xTrain, yTrain, xTest)
    alg1.print_metrics(y_predict, yTest)
