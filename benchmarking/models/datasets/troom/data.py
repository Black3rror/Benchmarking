import os

import numpy as np
import pandas as pd

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, test_ratio, normalize=False, random_seed=None):
        super().__init__()

        self.random_seed = random_seed

        self.dataset_root_dir = os.path.join(os.path.dirname(__file__), "data")
        self.room_data_file_path = os.path.join(self.dataset_root_dir, "raw/data_ekkono_room1_reduced_dataset.xlsx")
        self.sample_interval = 2 * 60       # 2 minutes
        self.prediction_horizon = 60 * 60   # 60 minutes

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(self.room_data_file_path, test_ratio, normalize, random_seed)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = self.train_y.shape[1]
        self.output_activation = "linear"
        self.loss_function = "mse"
        self.metrics = ["mae"]


    @staticmethod
    def load_dataset(room_data_file_path, test_ratio, normalize=False, random_seed=None):
        """
        Loads the dataset.

        Args:
            room_data_file_path (str): The file path of the room data.
            test_ratio (float): The ratio of the test set.
            normalize (bool): If True, the features are normalized.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        room_data = pd.read_excel(room_data_file_path, header=0)

        input_headers = ["PumpRPM[1]", "solar_normal1", "solar_normal2", "solar_normal3", "solar_normal4", "Tamb", "Troom[1]",
                         "solar_normal1future", "solar_normal2future", "solar_normal3future", "solar_normal4future", "Tamb_future"]
        output_headers = ["Troom[1]future"]

        # drop rows with NaN values
        room_data = room_data.dropna(subset=input_headers + output_headers)

        data_x = room_data[input_headers]
        data_y = room_data[output_headers]

        # convert to numpy
        data_x = data_x.values
        data_y = data_y.values

        # separate the data
        test_size = int(len(data_x) * test_ratio)
        train_x = data_x[:-test_size]
        train_y = data_y[:-test_size]
        test_x = data_x[-test_size:]
        test_y = data_y[-test_size:]

        # normalize the data
        if normalize:
            train_x_mean = train_x.mean(axis=0)
            train_x_std = train_x.std(axis=0)
            train_y_mean = train_y.mean(axis=0)
            train_y_std = train_y.std(axis=0)

            train_x = (train_x - train_x_mean) / train_x_std
            train_y = (train_y - train_y_mean) / train_y_std
            test_x = (test_x - train_x_mean) / train_x_std
            test_y = (test_y - train_y_mean) / train_y_std

        # shuffle the data
        rng = np.random.RandomState(random_seed)
        shuffled_indices = rng.permutation(len(train_x))
        train_x = train_x[shuffled_indices]
        train_y = train_y[shuffled_indices]

        train_x = train_x.astype(np.float32)
        train_y = train_y.astype(np.float32)
        test_x = test_x.astype(np.float32)
        test_y = test_y.astype(np.float32)

        return (train_x, train_y), (test_x, test_y)
