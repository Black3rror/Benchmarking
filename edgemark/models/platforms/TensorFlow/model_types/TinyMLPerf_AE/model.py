import importlib

import tensorflow as tf

from edgemark.models.platforms.TensorFlow.model_template import ModelSupervisorTemplate


# Based on the TinyMLPerf Deep AutoEncoder model (https://github.com/mlcommons/tiny/blob/master/benchmark/training/anomaly_detection/keras_model.py)
class ModelSupervisor(ModelSupervisorTemplate):
    def __init__(self, cfg=None):
        super().__init__()

        # default configs
        self.load_pretrained_model = False
        self.denses_params = [128, 128, 128, 128, 8, 128, 128, 128]
        self.denses_dropout = 0.00
        self.activation = "relu"
        self.use_batchnorm = True
        self.epochs = 50
        self.batch_size = 512
        self.dataset_info = {
            "name": "toyADMOS",
            "path": "edgemark/models/datasets/toyADMOS/data.py",
            "args": {
                "n_mels": 64,
                "frames": 5,
                "n_fft": 1024,
                "hop_length": 512,
                "power": 2.0,
                "test_ratio": 0.1,
                "random_seed": 42
            }
        }
        self.random_seed = 42

        self.learning_rate = 1e-3
        self.fine_tuning_learning_rate = 1e-4
        self.fine_tuning_epochs = 5
        self.fine_tuning_batch_size = 128

        # update configs if provided
        if cfg is not None:
            self.set_configs(cfg)

        # load corresponding dataset
        spec = importlib.util.spec_from_file_location("imported_module", self.dataset_info["path"])
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        # load data and create model
        self.dataset = imported_module.DatasetSupervisor(**self.dataset_info["args"])

        if self.load_pretrained_model:
            self.model = self.get_pretrained_model()
            assert self.model.input_shape[1:] == self.dataset.feature_shape, "The input shape of the model must match the feature shape of the dataset. The model input shape is {} and the dataset feature shape is {}.".format(self.model.input_shape[1:], self.dataset.feature_shape)
            assert self.model.output_shape[1] == self.dataset.num_labels, "The output shape of the model must match the number of labels in the dataset."
            assert self.dataset.output_activation.lower() == "linear", "The output activation of the dataset must be linear."
        else:
            input_shape = self.dataset.feature_shape
            assert len(input_shape) == 1, "The input shape must be 1-dimensional."
            output_size = self.dataset.num_labels
            output_activation = self.dataset.output_activation
            self.model = self.create_model(input_shape, self.denses_params, output_size, self.activation, output_activation, self.use_batchnorm, self.denses_dropout, self.random_seed)


    def set_configs(self, cfg):
        if "load_pretrained_model" in cfg:
            self.load_pretrained_model = cfg["load_pretrained_model"]
        if "denses_params" in cfg:
            self.denses_params = cfg["denses_params"]
        if "denses_dropout" in cfg:
            self.denses_dropout = cfg["denses_dropout"]
        if "activation" in cfg:
            self.activation = cfg["activation"]
        if "use_batchnorm" in cfg:
            self.use_batchnorm = cfg["use_batchnorm"]
        if "epochs" in cfg:
            self.epochs = cfg["epochs"]
        if "batch_size" in cfg:
            self.batch_size = cfg["batch_size"]
        if "dataset" in cfg:
            self.dataset_info = cfg["dataset"]
        if "random_seed" in cfg:
            self.random_seed = cfg["random_seed"]
        if "learning_rate" in cfg:
            self.learning_rate = cfg["learning_rate"]
        if "fine_tuning_learning_rate" in cfg:
            self.fine_tuning_learning_rate = cfg["fine_tuning_learning_rate"]
        if "fine_tuning_epochs" in cfg:
            self.fine_tuning_epochs = cfg["fine_tuning_epochs"]
        if "fine_tuning_batch_size" in cfg:
            self.fine_tuning_batch_size = cfg["fine_tuning_batch_size"]


    @staticmethod
    def create_model(input_shape, denses_params, output_size, activation, output_activation, use_batchnorm=False, denses_dropout=0.00, random_seed=None):
        """
        Creates the model.

        Args:
            input_shape (tuple): The shape of the input.
            denses_params (list): Each element is the number of neurons of a dense layer (excluding the output layer).
            output_size (int): The size of the output.
            activation (str): The activation function of the hidden layers.
            output_activation (str): The activation function of the output layer.
            use_batchnorm (bool): If True, batch normalization is used.
            denses_dropout (float): The dropout rate of the hidden layers.
            random_seed (int): The random seed.
        """
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)

        model = tf.keras.Sequential()
        layers = tf.keras.layers

        model.add(layers.Input(shape=input_shape))

        for i in range(len(denses_params)):
            n = denses_params[i]
            model.add(layers.Dense(n))
            if use_batchnorm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation))
            if denses_dropout > 0:
                model.add(layers.Dropout(denses_dropout))

        model.add(layers.Dense(output_size, activation=output_activation))

        return model


    def get_pretrained_model(self):
        model_url = "https://github.com/mlcommons/tiny/raw/master/benchmark/training/anomaly_detection/trained_models/ad01.h5"
        model_path = tf.keras.utils.get_file("TinyMLPerf_AE.h5", model_url)
        model = tf.keras.models.load_model(model_path)
        return model


    def compile_model(self, fine_tuning=False):
        if not fine_tuning:
            learning_rate = self.learning_rate
        else:
            learning_rate = self.fine_tuning_learning_rate

        self._compile_model(learning_rate, self.dataset.loss_function, self.dataset.metrics)


    def _compile_model(self, learning_rate, loss_function, metrics):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=opt, loss=loss_function, metrics=metrics)


    def train_model(self, fine_tuning=False, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
        if not fine_tuning:
            epochs, batch_size, random_seed = self.epochs, self.batch_size, self.random_seed
        else:
            epochs, batch_size, random_seed = self.fine_tuning_epochs, self.fine_tuning_batch_size, self.random_seed

        return self._train_model(epochs, batch_size, tensorboard_log_dir, best_weights_dir, use_wandb, random_seed)


    def _train_model(self, epochs, batch_size, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False, random_seed=42):
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        if self.dataset.test_x is None or self.dataset.test_y is None:
            validation_data = None
        else:
            validation_data = (self.dataset.test_x, self.dataset.test_y)

        callbacks = self._get_training_callbacks(tensorboard_log_dir, best_weights_dir, use_wandb)
        return self.model.fit(self.dataset.train_x, self.dataset.train_y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)


    def evaluate_model(self):
        return self._evaluate_model()


    def _evaluate_model(self):
        eval_vals = self.model.evaluate(self.dataset.test_x, self.dataset.test_y, verbose=0)
        eval_dict = {self.dataset.loss_function: eval_vals[0]}
        for i in range(len(self.dataset.metrics)):
            eval_dict[self.dataset.metrics[i]] = eval_vals[i+1]
        return eval_dict


    def get_model_info(self):
        return self._get_model_info()


    def _get_model_info(self):
        model_info = {
            "load_pretrained_model": self.load_pretrained_model,
            "denses_params": self.denses_params,
            "denses_dropout": self.denses_dropout,
            "activation": self.activation,
            "use_batchnorm": self.use_batchnorm,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "fine_tuning_epochs": self.fine_tuning_epochs,
            "fine_tuning_batch_size": self.fine_tuning_batch_size,
            "fine_tuning_learning_rate": self.fine_tuning_learning_rate,
            "dataset": self.dataset_info,
            "random_seed": self.random_seed
        }
        return model_info
