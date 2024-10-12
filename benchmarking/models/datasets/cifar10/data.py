import tensorflow as tf

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, flat_features=False):
        super().__init__()

        self.flat_features = flat_features

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(flat_features)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = 10
        self.output_activation = "softmax"
        self.loss_function = "categorical_crossentropy"
        self.metrics = ["accuracy"]


    @staticmethod
    def load_dataset(flat_features=False):
        """
        Loads the dataset.

        Args:
            flat_features (bool): If True, the features are flattened.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

        train_x = train_x.astype('float32') / 255
        test_x = test_x.astype('float32') / 255

        train_y = tf.keras.utils.to_categorical(train_y, 10)
        test_y = tf.keras.utils.to_categorical(test_y, 10)

        if flat_features:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)

        return (train_x, train_y), (test_x, test_y)
