import importlib

import numpy as np
import tensorflow as tf

from edgemark.models.platforms.TensorFlow.model_template import ModelSupervisorTemplate


# Based on the TinyMLPerf ResNet model (https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/keras_model.py)
class ModelSupervisor(ModelSupervisorTemplate):
    def __init__(self, cfg=None):
        super().__init__()

        # default configs
        self.load_pretrained_model = False
        self.num_filters = 16    # this should be 64 for an official resnet model
        self.activation = "relu"
        self.use_batchnorm = True
        self.epochs = 50
        self.batch_size = 32
        self.dataset_info = {
            "name": "cifar10",
            "path": "edgemark/models/datasets/cifar10/data.py",
            "args": {
                "flat_features": False
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
            assert self.model.input_shape[1:] == self.dataset.feature_shape, "The input shape of the model must match the feature shape of the dataset."
            assert self.model.output_shape[1] == self.dataset.num_labels, "The output shape of the model must match the number of labels in the dataset."
            assert self.dataset.output_activation.lower() == "softmax", "The output activation of the dataset must be softmax."
        else:
            input_shape = self.dataset.feature_shape
            assert len(input_shape) == 3, "The input shape must be 3-dimensional."
            output_size = self.dataset.num_labels
            output_activation = self.dataset.output_activation
            self.model = self.create_model(input_shape, output_size, self.num_filters, self.activation, output_activation, self.use_batchnorm, self.random_seed)


    def set_configs(self, cfg):
        if "load_pretrained_model" in cfg:
            self.load_pretrained_model = cfg["load_pretrained_model"]
        if "num_filters" in cfg:
            self.num_filters = cfg["num_filters"]
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
    def create_model(input_shape, output_size, num_filters, activation, output_activation, use_batchnorm, random_seed=None):
        """
        Creates the model.

        Args:
            input_shape (tuple): The shape of the input.
            output_size (int): The size of the output.
            num_filters (int): The number of filters for the convolutional layers in the first stack.
            activation (str): The activation function of the hidden layers.
            output_activation (str): The activation function of the output layer.
            use_batchnorm (bool): If True, batch normalization is used.
            random_seed (int): The random seed.
        """
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)

        layers = tf.keras.layers

        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        # x = layers.MaxPooling2D(pool_size=2)(x)     # uncomment this for official resnet model

        # First stack

        # Weight layers
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)

        # Overall residual, connect weight layer and identity paths
        x = layers.add([x, y])
        x = layers.Activation(activation)(x)

        # Second stack

        # Weight layers
        num_filters = 2 * num_filters   # Filters need to be double for each stack
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)

        # Adjust for change in dimension due to stride in identity
        x = layers.Conv2D(filters=num_filters, kernel_size=1, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        # Overall residual, connect weight layer and identity paths
        x = layers.add([x, y])
        x = layers.Activation(activation)(x)

        # Third stack

        # Weight layers
        num_filters = 2 * num_filters
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)
        y = layers.Activation(activation)(y)
        y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
        if use_batchnorm:
            y = layers.BatchNormalization()(y)

        # Adjust for change in dimension due to stride in identity
        x = layers.Conv2D(filters=num_filters, kernel_size=1, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        # Overall residual, connect weight layer and identity paths
        x = layers.add([x, y])
        x = layers.Activation(activation)(x)

        # Fourth stack.
        # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
        # Uncomments to use it

        # # Weight layers
        # num_filters = 2 * num_filters
        # y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        # if use_batchnorm:
        #     y = layers.BatchNormalization()(y)
        # y = layers.Activation(activation)(y)
        # y = layers.Conv2D(filters=num_filters, kernel_size=3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(y)
        # if use_batchnorm:
        #     y = layers.BatchNormalization()(y)

        # # Adjust for change in dimension due to stride in identity
        # x = layers.Conv2D(filters=num_filters, kernel_size=1, strides=2, padding="same", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        # # Overall residual, connect weight layer and identity paths
        # x = layers.add([x, y])
        # x = layers.Activation(activation)(x)

        # Final classification layer.
        pool_size = int(np.amin(x.shape[1:3]))
        x = layers.AveragePooling2D(pool_size=pool_size)(x)
        y = layers.Flatten()(x)
        outputs = layers.Dense(output_size, activation=output_activation, kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model


    def get_pretrained_model(self):
        # Bug: Loading the model will result in a very high categorical crossentropy loss, but accuracy is reasonable
        model_url = "https://github.com/mlcommons/tiny/raw/master/benchmark/training/image_classification/trained_models/pretrainedResnet.h5"
        model_path = tf.keras.utils.get_file("TinyMLPerf_ResNet.h5", model_url)
        model = tf.keras.models.load_model(model_path)

        # The model is designed for input values ranging from 0 to 255, while the dataset is normalized to 0 to 1
        # Layer 0 is the input layer, and layer 1 is the first convolutional layer that we'll multiply its weights by 255
        weights, biases = model.layers[1].get_weights()
        weights *= 255
        model.layers[1].set_weights([weights, biases])

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
            "num_filters": self.num_filters,
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


    def _estimate_arena_size(self):
        # the default setting would result in 132072, which is less than what is required
        # so, we'll set it to 190000 (200000 after adding the safety buffer) which is the amount required
        arena_size = 190000

        return arena_size
