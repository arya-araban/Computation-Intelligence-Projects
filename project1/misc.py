from tensorflow.python.keras.datasets import mnist


def load_ds():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # load the dataset

    #   PRE-PROCESSING  #
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    '''changing input into n*m vector for train and test'''
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    return x_train, x_test, y_train, y_test