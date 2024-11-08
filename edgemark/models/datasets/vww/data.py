import os

import numpy as np
import tensorflow as tf

from edgemark.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, test_ratio, image_size=(96, 96), dataset_ratio=1, flat_features=False, random_seed=42):
        super().__init__()

        self.test_ratio = test_ratio
        self.image_size = image_size
        self.dataset_ratio = dataset_ratio
        self.flat_features = flat_features
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(test_ratio, image_size, dataset_ratio, flat_features, random_seed)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = 2
        self.output_activation = "softmax"
        self.loss_function = "categorical_crossentropy"
        self.metrics = ["accuracy"]


    @staticmethod
    def load_dataset(test_ratio, image_size=(96, 96), dataset_ratio=1, flat_features=False, random_seed=42):
        """
        Loads the dataset.

        Args:
            test_ratio (float): The ratio of the test set.
            image_size (tuple): The size of the images.
            dataset_ratio (float): The ratio of the dataset to be used. Can be used to reduce the dataset size and memory usage.
            flat_features (bool): If True, the features are flattened.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        dataset_dir = DatasetSupervisor._download_dataset()

        (train_x, train_y), (test_x, test_y) = DatasetSupervisor._load_data_from_dir(dataset_dir, test_ratio, image_size, dataset_ratio, random_seed)

        if flat_features:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)

        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _download_dataset():
        """
        Downloads the dataset.

        Returns:
            str: The path to the dataset directory.
        """
        dataset_url = 'https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz'
        dataset_file_name = 'vw_coco2014_96.tar.gz'
        extracted_dir_name = 'vw_coco2014_96'

        download_path = tf.keras.utils.get_file(fname=dataset_file_name, origin=dataset_url, extract=False)
        extracted_dir = os.path.join(os.path.dirname(download_path), extracted_dir_name)

        if not os.path.exists(extracted_dir):
            tf.keras.utils.get_file(fname=dataset_file_name, origin=dataset_url, extract=True)

        return extracted_dir


    @staticmethod
    def _load_data_from_dir(dataset_dir, test_ratio, image_size=(96, 96), dataset_ratio=1.0, random_seed=42):
        trainset, testset = tf.keras.preprocessing.image_dataset_from_directory(
            dataset_dir,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,      # Resize all images
            batch_size=None,
            shuffle=True,
            seed=random_seed,
            validation_split=test_ratio,
            subset='both'
        )

        trainset_size = tf.data.experimental.cardinality(trainset).numpy()
        testset_size = tf.data.experimental.cardinality(testset).numpy()

        trainset_size = dataset_ratio * trainset_size
        testset_size = dataset_ratio * testset_size

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        for train_x_sample, train_y_sample in trainset.as_numpy_iterator():
            train_x.append(train_x_sample/255.0)
            train_y.append(train_y_sample)
            if len(train_x) >= trainset_size:
                break

        for test_x_sample, test_y_sample in testset.as_numpy_iterator():
            test_x.append(test_x_sample/255.0)
            test_y.append(test_y_sample)
            if len(test_x) >= testset_size:
                break

        train_x = np.array(train_x).astype(np.float32)
        train_y = np.array(train_y).astype(np.float32)
        test_x = np.array(test_x).astype(np.float32)
        test_y = np.array(test_y).astype(np.float32)

        return (train_x, train_y), (test_x, test_y)
