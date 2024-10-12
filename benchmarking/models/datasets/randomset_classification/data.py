import numpy as np

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, n_samples, test_ratio, feature_shape, num_labels, random_seed=None):
        super().__init__()

        self.n_samples = n_samples
        self.test_ratio = test_ratio
        self.feature_shape = feature_shape
        self.num_labels = num_labels
        self.output_activation = "softmax"
        self.loss_function = "categorical_crossentropy"
        self.metrics = ["accuracy"]
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(n_samples, test_ratio, feature_shape, num_labels, random_seed)


    @staticmethod
    def load_dataset(n_samples, test_ratio, feature_shape, num_labels, random_seed=None):
        """
        Loads the dataset.

        Args:
            n_samples (int): The number of samples that should be in the whole dataset (train + test).
            test_ratio (float): The ratio of the test set.
            feature_shape (tuple): The shape of the features.
            num_labels (int): The number of labels.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        dataset_x, dataset_y = DatasetSupervisor._create_dataset(n_samples, feature_shape, num_labels, random_seed)
        dataset_x = dataset_x.astype(np.float32)
        dataset_y = dataset_y.astype(np.float32)

        test_size = int(len(dataset_x)*test_ratio)
        train_x = dataset_x[test_size:]
        train_y = dataset_y[test_size:]
        test_x = dataset_x[:test_size]
        test_y = dataset_y[:test_size]

        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _create_dataset(n_samples, feature_shape, num_labels, random_seed=None):
        rng = np.random.RandomState(random_seed)
        x = rng.rand(n_samples, *feature_shape)
        y = rng.randint(0, num_labels, n_samples)
        y = np.eye(num_labels)[y]
        return x, y
