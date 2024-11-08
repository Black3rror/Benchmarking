import numpy as np
import tensorflow as tf

from edgemark.models.datasets.data_template import DatasetSupervisorTemplate


class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, sequence_length, test_ratio, random_seed=None):
        super().__init__()

        self.sequence_length = sequence_length
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y), self.char2index, self.index2char = self._load_dataset(self.sequence_length, self.test_ratio, self.random_seed)

        self.input_size = len(self.char2index)
        self.output_size = len(self.char2index)
        self.sequence_length = self.sequence_length
        self.sequential_output = True
        self.output_activation = "softmax"
        self.loss_function = "sparse_categorical_crossentropy"
        self.metrics = ["sparse_categorical_accuracy"]


    @staticmethod
    def load_dataset(sequence_length, test_ratio, random_seed=None):
        """
        Loads the dataset.

        Args:
            sequence_length (int): The length of the sequences.
            test_ratio (float): The ratio of the test set.
            random_seed (int): The random seed.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        (train_x, train_y), (test_x, test_y), char2index, index2char = DatasetSupervisor._load_dataset(sequence_length, test_ratio, random_seed)
        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _load_dataset(sequence_length, test_ratio, random_seed=None):
        text = DatasetSupervisor._load_text()

        dataset, char2index, index2char = DatasetSupervisor._process_text(text, sequence_length, random_seed)

        train_size = int((1 - test_ratio) * len(dataset))
        trainset = dataset.take(train_size)
        testset = dataset.skip(train_size)

        # get train_x, train_y, test_x, test_y as numpy arrays
        train_x, train_y = [], []
        test_x, test_y = [], []
        for x, y in trainset:
            train_x.append(x.numpy())
            train_y.append(y.numpy())
        for x, y in testset:
            test_x.append(x.numpy())
            test_y.append(y.numpy())
        train_x, train_y = np.array(train_x).astype(np.float32), np.array(train_y).astype(np.float32)
        test_x, test_y = np.array(test_x).astype(np.float32), np.array(test_y).astype(np.float32)

        return (train_x, train_y), (test_x, test_y), char2index, index2char


    @staticmethod
    def _load_text():
        dataset_file_name = 'shakespeare.txt'
        dataset_file_origin = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

        dataset_file_path = tf.keras.utils.get_file(
            fname=dataset_file_name,
            origin=dataset_file_origin
        )

        text = open(dataset_file_path, mode='r').read()
        return text


    @staticmethod
    def _process_text(text, sequence_length, random_seed=None):

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        vocab = sorted(set(text))

        char2index = {char: index for index, char in enumerate(vocab)}
        index2char = np.array(vocab)

        text_as_int = np.array([char2index[char] for char in text])

        sequences = tf.data.Dataset.from_tensor_slices(text_as_int)             # each character in text is now a data in dataset
        sequences = sequences.batch(sequence_length + 1, drop_remainder=True)   # each sequence consists of 101 characters

        dataset = sequences.map(split_input_target)                             # each element is (input, target) tuple
        dataset = dataset.shuffle(10000, seed=random_seed)

        return dataset, char2index, index2char
