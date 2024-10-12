import numpy as np

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, n_samples, test_ratio, input_size, output_size, sequence_length, sequential_output, using_embedding, random_seed=None):
        super().__init__()

        self.n_samples = n_samples
        self.test_ratio = test_ratio
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.sequential_output = sequential_output
        self.output_activation = "softmax" if using_embedding else "linear"
        self.loss_function = "sparse_categorical_crossentropy" if using_embedding else "mse"
        self.metrics = ["sparse_categorical_accuracy"] if using_embedding else ["mae"]
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(n_samples, test_ratio, input_size, output_size, sequence_length, sequential_output, using_embedding, random_seed)


    @staticmethod
    def load_dataset(n_samples, test_ratio, input_size, output_size, sequence_length, sequential_output, using_embedding, random_seed=None):
        """
        Loads the dataset.

        Args:
            n_samples (int): The number of samples that should be in the whole dataset (train + test).
            test_ratio (float): The ratio of the test set.
            input_size (int): The size of each element in the sequence.
            output_size (int): The size of the output.
            sequence_length (int): The length of the sequences.
            sequential_output (bool): If True, the output is sequential.
            using_embedding (bool): If True, inputs are fed to an embedding layer.
            random_seed (int): The random seed. If None, the random seed is not set.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        dataset_x, dataset_y = DatasetSupervisor._create_dataset(n_samples, input_size, output_size, sequence_length, sequential_output, using_embedding, random_seed)
        dataset_x = dataset_x.astype(np.float32)
        dataset_y = dataset_y.astype(np.float32)

        test_size = int(len(dataset_x)*test_ratio)
        train_x = dataset_x[test_size:]
        train_y = dataset_y[test_size:]
        test_x = dataset_x[:test_size]
        test_y = dataset_y[:test_size]

        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _create_dataset(n_samples, input_size, output_size, sequence_length, sequential_output, using_embedding, random_seed=None):
        rng = np.random.RandomState(random_seed)

        if using_embedding:
            data = rng.randint(0, input_size, (n_samples, sequence_length+1)).astype(np.float32)
        else:
            data = rng.rand(n_samples, sequence_length+1, input_size).astype(np.float32)

        x = data[:, :-1]

        if sequential_output:
            if output_size == input_size:
                y = data[:, 1:]
            else:
                if using_embedding:
                    y = rng.randint(0, output_size, (n_samples, sequence_length)).astype(np.float32)
                else:
                    y = rng.rand(n_samples, sequence_length, output_size).astype(np.float32)
        else:
            if output_size == input_size:
                y = data[:, -1]
            else:
                if using_embedding:
                    y = rng.randint(0, output_size, (n_samples)).astype(np.float32)
                else:
                    y = rng.rand(n_samples, output_size).astype(np.float32)

        return x, y
