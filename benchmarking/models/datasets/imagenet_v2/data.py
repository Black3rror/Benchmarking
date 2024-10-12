import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, test_ratio, random_seed=None):
        super().__init__()

        self.test_ratio = test_ratio
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(self.test_ratio, self.random_seed)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = 1000
        self.output_activation = "softmax"
        self.loss_function = "sparse_categorical_crossentropy"
        self.metrics = ["sparse_categorical_accuracy"]


    @staticmethod
    def load_dataset(test_ratio, random_seed=None):
        """
        Loads the dataset.

        Args:
            flat_features (bool): If True, the features are flattened.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        train_percent = 100 - int(test_ratio * 100)
        trainset, testset = tfds.load('imagenet_v2', split = ['test[:{}%]'.format(train_percent), 'test[{}%:]'.format(train_percent)], as_supervised = True)

        trainset = trainset.map(lambda x, y: (tf.image.resize(x/255, (224, 224)), y)).shuffle(1024, seed=random_seed)
        testset = testset.map(lambda x, y: (tf.image.resize(x/255, (224, 224)), y))

        # get train_x, train_y, test_x, test_y as numpy arrays
        train_x, train_y = [], []
        test_x, test_y = [], []
        for x, y in trainset:
            train_x.append(x.numpy())
            train_y.append(y.numpy())
        for x, y in testset:
            test_x.append(x.numpy())
            test_y.append(y.numpy())
        train_x, train_y = np.array(train_x), np.array(train_y)
        test_x, test_y = np.array(test_x), np.array(test_y)

        return (train_x, train_y), (test_x, test_y)
