import glob
import os
import sys
import zipfile

import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from benchmarking.models.datasets.data_template import DatasetSupervisorTemplate


# Based on TinyMLPerf Anomaly Detection (https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection)
class DatasetSupervisor(DatasetSupervisorTemplate):
    def __init__(self, n_mels, frames, n_fft, hop_length, power, test_ratio, random_seed=42):
        super().__init__()

        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset(n_mels, frames, n_fft, hop_length, power, test_ratio, random_seed)

        self.feature_shape = self.train_x.shape[1:]
        self.num_labels = self.train_x.shape[1]
        self.output_activation = "linear"
        self.loss_function = "mse"
        self.metrics = ["mae"]


    @staticmethod
    def load_dataset(n_mels, frames, n_fft, hop_length, power, test_ratio, random_seed=42):
        """
        Loads the dataset.

        Args:
            n_mels (int): The number of mel bins.
            frames (int): The length of frames in time-domain to be included in one sample data.
            n_fft (int): The length of the FFT window.
            hop_length (int): The number of samples between successive FFT frames.
            power (float): The exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
            test_ratio (float): The ratio of the test set.
            random_seed (int): The random seed.

        Returns:
            tuple: ((trainX, trainY), (testX, testY))
        """
        dataset_dir = DatasetSupervisor._download_dataset()

        (train_x, train_y), (test_x, test_y) = DatasetSupervisor._load_data_from_dir(dataset_dir, n_mels, frames, n_fft, hop_length, power, test_ratio, random_seed)

        return (train_x, train_y), (test_x, test_y)


    @staticmethod
    def _download_dataset():
        """
        Downloads the dataset.

        Returns:
            str: The path to the dataset directory.
        """
        dataset_dir_name = 'ToyCar'

        dataset_url = 'https://zenodo.org/record/3678171/files/dev_data_ToyCar.zip?download=1'
        dataset_file_name = 'dev_data_ToyCar.zip'
        file_1_path = tf.keras.utils.get_file(fname=dataset_file_name, origin=dataset_url, extract=False)

        dataset_url = 'https://zenodo.org/record/3727685/files/eval_data_train_ToyCar.zip?download=1'
        dataset_file_name = 'eval_data_train_ToyCar.zip'
        file_2_path = tf.keras.utils.get_file(fname=dataset_file_name, origin=dataset_url, extract=False)

        extracted_dir = os.path.join(os.path.dirname(file_1_path), dataset_dir_name)
        if not os.path.exists(extracted_dir):
            with zipfile.ZipFile(file_1_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(extracted_dir))
            with zipfile.ZipFile(file_2_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(extracted_dir))

        return extracted_dir


    @staticmethod
    def _file_list_generator(dir, ext="wav"):
        training_list_path = os.path.abspath("{}/*.{}".format(dir, ext))
        files = sorted(glob.glob(training_list_path))
        if len(files) == 0:
            raise ValueError("No files found in {}".format(training_list_path))

        return files


    @staticmethod
    def _file_to_vector_array(file_name, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
        # 01 calculate the number of dimensions
        dims = n_mels * frames

        # 02 generate melspectrogramtry:
        try:
            y, sr = librosa.load(file_name, sr=None, mono=False)
        except Exception:
            raise ValueError("file reading error: ", file_name)

        # 02a generate melspectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)

        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

        # 3b take central part only
        log_mel_spectrogram = log_mel_spectrogram[:,50:250]

        # 04 calculate total vector size
        vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

        # 05 skip too short clips
        if vector_array_size < 1:
            return np.empty((0, dims))

        # 06 generate feature vectors by concatenating multiframes
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(frames):
            vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        return vector_array


    @staticmethod
    def _list_to_vector_array(file_list, n_mels=64, frames=5, n_fft=1024, hop_length=512, power=2.0):
        # calculate the number of dimensions
        dims = n_mels * frames

        # iterate file_to_vector_array()
        for i in tqdm(range(len(file_list)), desc="processing the dataset", leave=False):
            vector_array = DatasetSupervisor._file_to_vector_array(file_list[i], n_mels, frames, n_fft, hop_length, power)

            if i == 0:
                dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
            dataset[vector_array.shape[0] * i: vector_array.shape[0] * (i + 1), :] = vector_array

        return dataset


    @staticmethod
    def _load_data_from_dir(dataset_dir, n_mels, frames, n_fft, hop_length, power, test_ratio, random_seed=42):
        np.random.seed(random_seed)

        files = DatasetSupervisor._file_list_generator(os.path.join(dataset_dir, 'train'))
        dataset = DatasetSupervisor._list_to_vector_array(files, n_mels, frames, n_fft, hop_length, power)

        # shuffle dataset
        np.random.shuffle(dataset)

        # split into training and test sets
        train_size = int(len(dataset) * (1 - test_ratio))
        trainset, testset = np.split(dataset, [train_size])

        trainset = trainset.astype(np.float32)
        testset = testset.astype(np.float32)

        return (trainset, trainset), (testset, testset)
