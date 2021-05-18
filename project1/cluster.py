from sklearn import metrics
from sklearn.metrics import accuracy_score, adjusted_rand_score
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def load_ds():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # load the dataset

    #   PRE-PROCESSING  #
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    return x_train, x_test, y_train, y_test


class K_MEANS:
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

    def train(self, x_train, y_train):
        kmeans = MiniBatchKMeans(n_clusters=self.num_seeds)
        kmeans.fit(x_train)
        labels = kmeans.labels_
        # plot_k_means(x_train,labels)
        mapping = self.__retrieve_info(labels, y_train)
        # print(mapping)
        # print(labels)
        # print(y_train)

        number_labels = np.zeros(len(labels))
        for i in range(len(labels)):  # performing the mapping
            number_labels[i] = mapping[kmeans.labels_[i]]
        y_prediction = number_labels.astype('int')

        return y_prediction
        # print("****")
        # print(number_labels)
        # print(y_train)
        # self.print_metrics(number_labels, y_train)

    def __init__(self, num_seeds):
        # self.num_seeds = len(np.unique(y_train))
        self.num_seeds = num_seeds


if __name__ == "__main__":
    alg1 = K_MEANS(300)
    x_train, x_test, y_train, y_test = load_ds()
    y_predict = alg1.train(x_train, y_train)
    alg1.print_metrics(y_predict, y_train)
