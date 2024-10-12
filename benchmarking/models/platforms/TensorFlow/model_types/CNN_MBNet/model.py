import importlib

import tensorflow as tf

from benchmarking.models.platforms.TensorFlow.model_template import ModelSupervisorTemplate


class ModelSupervisor(ModelSupervisorTemplate):
    def __init__(self, cfg=None):
        super().__init__()

        # default configs
        self.epochs = 50
        self.batch_size = 32
        self.dataset_info = {
            "name": "imagenet_v2",
            "path": "benchmarking/models/datasets/imagenet_v2/data.py",
            "args": {
                "test_ratio": 0.2,
                "random_seed": 42
            }
        }
        self.random_seed = 42

        self.learning_rate = 1e-3
        self.fine_tuning_learning_rate = 1e-4
        self.fine_tuning_epochs = 10
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
        self.model = self.create_model()


    def set_configs(self, cfg):
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
    def create_model():
        """
        Creates the model.

        Returns:
            tf.keras.Model: MobileNetV2 model with pre-trained weights on ImageNet dataset.
        """
        return tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)


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
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'fine_tuning_epochs': self.fine_tuning_epochs,
            'fine_tuning_batch_size': self.fine_tuning_batch_size,
            'fine_tuning_learning_rate': self.fine_tuning_learning_rate,
            'dataset': self.dataset_info,
            'random_seed': self.random_seed
        }
        return model_info
