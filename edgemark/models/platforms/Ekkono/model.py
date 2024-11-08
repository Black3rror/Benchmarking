import importlib
import os
import time

import numpy as np
import wandb
import yaml
from ekkono import crystal, primer


class ModelSupervisor:
    def __init__(self, cfg=None):
        """
        Initializes the class.
        The following attributes should be set in the __init__ function:
            self.model (ekkono.primer.Model): The model.
            self.dataset (DatasetSupervisorTemplate): The dataset.

        Args:
            cfg (dict): The configuration of the model. Defaults to None.
        """

        # default configs
        self.ekkono_model_type = "pretrained"  # Ekkono model type. "pretrained" or "incremental". "pretrained" is not able to be trained on the end device, but "incremental" is.
        self.denses_params = [16]       # each element is the number of neurons of a dense layer
        self.activation = "sigmoid"     # the activation function of layers. It can be one of "sigmoid", "tanh", "relu" (which actually is leaky relu)
        self.epochs = 50
        self.batch_size = 32
        self.dataset_info = {
            "name": "cifar10",
            "path": "edgemark/models/datasets/cifar10/data.py",
            "args": {
                "flat_features": True
            }
        }
        self.random_seed = None

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

        # load data
        self.dataset = imported_module.DatasetSupervisor(**self.dataset_info["args"])
        assert len(self.dataset.feature_shape) == 1, "The input shape must be 1-dimensional."
        assert self.dataset.num_labels == 1, "The output size should be 1."
        assert len(self.dataset.train_x.shape) == 2, "The train_x should be 2-dimensional."
        assert len(self.dataset.train_y.shape) == 2, "The train_y should be 2-dimensional."
        assert self.dataset.train_y.shape[1] == 1, "The output size should be 1."
        assert self.dataset.output_activation == "linear", "The output activation should be linear."
        assert self.dataset.loss_function in ["mean_squared_error", "mse"], "The loss function should be mean_squared_error."
        for metric in self.dataset.metrics:
            if metric not in ["mean_squared_error", "mse", "rmse", "mean_absolute_error", "mae"]:
                print("Warning: The metric '{}' is not supported.".format(metric))

        (self.trainset, self.testset), target_name = self._create_ekkono_datasets(self.dataset)

        # create the model
        if self.ekkono_model_type == "pretrained":
            self.model = self.create_pretrained_model(self.denses_params, self.activation, self.learning_rate, self.trainset.pipeline_template, target_name)
        elif self.ekkono_model_type == "incremental":
            self.model = self.create_incremental_model(self.denses_params, self.activation, self.epochs, self.batch_size, self.learning_rate, self.trainset, target_name)
        else:
            raise ValueError("The specified model type is not supported.")


    @staticmethod
    def _create_ekkono_datasets(dataset):
        """
        Creates Ekkono datasets based on the given dataset.

        Args:
            dataset (DatasetSupervisorTemplate): The dataset.

        Returns:
            tuple: ((ekkono.primer.Dataset, ekkono.primer.Dataset), str) which is ((trainset, testset), target_name)
        """
        assert len(dataset.feature_shape) == 1, "The input shape must be 1-dimensional."
        assert dataset.num_labels == 1, "The output size should be 1."
        assert len(dataset.train_x.shape) == 2, "The train_x should be 2-dimensional."
        assert len(dataset.train_y.shape) == 2, "The train_y should be 2-dimensional."
        assert dataset.train_y.shape[1] == 1, "The output size should be 1."

        trainset_attributes = []
        for i in range(dataset.feature_shape[0]):
            trainset_attributes.append(primer.AttributeMeta('x_'+str(i)))
        trainset_attributes.append(primer.AttributeMeta('y'))
        trainset = primer.Dataset(trainset_attributes)

        testset_attributes = []
        for i in range(dataset.feature_shape[0]):
            testset_attributes.append(primer.AttributeMeta('x_'+str(i)))
        testset_attributes.append(primer.AttributeMeta('y'))
        testset = primer.Dataset(testset_attributes)

        for x, y in zip(dataset.train_x, dataset.train_y):
            instance = [x_i.tolist() for x_i in x]
            instance.append(y[0].tolist())
            trainset.create_instance(instance)

        for x, y in zip(dataset.test_x, dataset.test_y):
            instance = [x_i.tolist() for x_i in x]
            instance.append(y[0].tolist())
            testset.create_instance(instance)

        return (trainset, testset), "y"


    def set_configs(self, cfg):
        """
        Sets the configurations of the model.

        Note: The changed configs won't affect the data, model, or any other loaded attributes.
        In case you want to change them, you should call the corresponding functions.

        Args:
            cfg (dict): The configuration.
        """
        if "model_type" in cfg:
            if cfg["model_type"] is not None and cfg["model_type"] != "CNN":
                raise ValueError("The model type should be 'CNN'.")
        if "ekkono_model_type" in cfg:
            self.ekkono_model_type = cfg["ekkono_model_type"]
        if "convs_params" in cfg:
            if cfg["convs_params"] is not None and cfg["convs_params"] != []:
                raise ValueError("The model doesn't support convolutional layers.")
        if "denses_params" in cfg:
            self.denses_params = cfg["denses_params"]
        if "convs_dropout" in cfg:
            if cfg["convs_dropout"] is not None and cfg["convs_dropout"] != 0.00:
                raise ValueError("Dropout is not supported in the model.")
        if "denses_dropout" in cfg:
            if cfg["denses_dropout"] is not None and cfg["denses_dropout"] != 0.00:
                raise ValueError("Dropout is not supported in the model.")
        if "activation" in cfg:
            self.activation = cfg["activation"]
        if "use_batchnorm" in cfg:
            if cfg["use_batchnorm"] is not None and cfg["use_batchnorm"] is not False:
                raise ValueError("Batch normalization is not supported in the model.")
        if "epochs" in cfg:
            self.epochs = cfg["epochs"]
        if "batch_size" in cfg:
            self.batch_size = cfg["batch_size"]
        if "dataset" in cfg:
            self.dataset_info = cfg["dataset"]
        if "random_seed" in cfg:
            if cfg["random_seed"] is not None:
                print("Warning: Ekkono API doesn't let us to control its random operations. The random seed is ignored.")
        if "learning_rate" in cfg:
            self.learning_rate = cfg["learning_rate"]
        if "fine_tuning_learning_rate" in cfg:
            self.fine_tuning_learning_rate = cfg["fine_tuning_learning_rate"]
        if "fine_tuning_epochs" in cfg:
            self.fine_tuning_epochs = cfg["fine_tuning_epochs"]
        if "fine_tuning_batch_size" in cfg:
            self.fine_tuning_batch_size = cfg["fine_tuning_batch_size"]


    @staticmethod
    def create_pretrained_model(denses_params, activation, learning_rate, pipeline_template, target_name):
        """
        Creates the Ekkono pretrained model. The model is not able to be trained on the end device.

        Args:
            denses_params (list): Each element is the number of neurons of a dense layer (excluding the output layer which has one neuron).
            activation (str): The activation function of the hidden layers. Can be one of "sigmoid", "tanh", "relu" (which actually is leaky relu).
            pipeline_template (Ekkono.primer.PipelineTemplate): The pipeline template (usually taken from dataset).
            target_name (str): The name of the target attribute.
        """
        activation_function = None
        if activation == "sigmoid":
            activation_function = primer.MLPModelParams.ActivationFunction.SIGMOID
        elif activation == "tanh":
            activation_function = primer.MLPModelParams.ActivationFunction.TANH
        elif activation == "relu":
            activation_function = primer.MLPModelParams.ActivationFunction.LEAKYRELU
        else:
            raise ValueError("The specified activation function is not supported.")

        pipeline_template.add_target(target_name)

        edge_model = primer.ModelFactory.create_mlp_model(
            pipeline_template,
            hidden_layers=denses_params,
            optimizer= primer.MLPModelParams.Optimizer.ADAM,
            start_learning_rate=learning_rate,
            end_learning_rate=learning_rate,
            activation_function=activation_function
        )

        return edge_model


    @staticmethod
    def create_incremental_model(denses_params, activation, epochs, batch_size, learning_rate, trainset, target_name):
        """
        Creates the Ekkono incremental model. The model is able to be trained on the end device.

        Args:
            denses_params (list): Each element is the number of neurons of a dense layer (excluding the output layer which has one neuron).
            activation (str): The activation function of the hidden layers. Can be one of "sigmoid", "tanh", "relu" (which actually is leaky relu).
            epochs (int): The number of epochs for training.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate for training.
            trainset (DatasetSupervisorTemplate): The training dataset.
            target_name (str): The name of the target attribute.
        """
        activation_function = None
        if activation == "sigmoid":
            activation_function = primer.MLPModelParams.ActivationFunction.SIGMOID
        elif activation == "tanh":
            activation_function = primer.MLPModelParams.ActivationFunction.TANH
        elif activation == "relu":
            activation_function = primer.MLPModelParams.ActivationFunction.LEAKYRELU
        else:
            raise ValueError("The specified activation function is not supported.")

        pipeline_template = trainset.pipeline_template
        pipeline_template.add_target(target_name)

        edge_model = primer.ModelFactory.create_incremental_mlp_model(
            pipeline_template,
            hidden_layers=denses_params,                        # Crystal Tunable
            batch_size=batch_size,                              # Crystal Tunable
            iterations_per_batch=epochs,                        # Crystal Fixed - Edge can have an epoch greater than 1 but for Crystal it is fixed to 1
            optimizer= primer.MLPModelParams.Optimizer.ADAM,    # Crystal Fixed - Edge can use different optimizers but Crystal uses GRADIENTDESCENT
            start_learning_rate=learning_rate,                  # Crystal Tunable
            end_learning_rate=learning_rate,                    # Crystal Tunable
            decay_instances=5000,                               # Crystal Tunable
            activation_function=activation_function,            # Crystal Tunable
            attribute_stats=pipeline_template.instantiate(trainset).get_attribute_stats()
        )

        return edge_model


    # Optional function: Whether you implement this function or not depends on your application.
    def train_model(self):
        """
        Trains the model.
        """
        primer.ModelTrainer.train(self.model, self.trainset, batch_size=self.batch_size, epochs=self.epochs)


    # Optional function: Whether you implement this function or not depends on your application.
    def evaluate_model(self):
        """
        Evaluates the model.

        Returns:
            dict: The evaluation metrics.
        """
        train_pred_tuples = primer.ModelTester.get_prediction_tuples(self.model, self.model.target_attributes[0], self.trainset)
        train_mae = primer.ModelTester.calculate_mae(train_pred_tuples)
        train_rmse = primer.ModelTester.calculate_rmse(train_pred_tuples)

        test_pred_tuples = primer.ModelTester.get_prediction_tuples(self.model, self.model.target_attributes[0], self.testset)
        test_mae = primer.ModelTester.calculate_mae(test_pred_tuples)
        test_rmse = primer.ModelTester.calculate_rmse(test_pred_tuples)

        output = {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        }
        return output


    def get_model_info(self):
        """
        Returns the model info.

        Returns:
            dict: The model info.
        """
        model_info = {
            "ekkono_model_type": self.ekkono_model_type,
            "denses_params": self.denses_params,
            "activation": self.activation,
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


    def _estimate_arena_size(self, verbose=False):
        """
        Estimates the size of the arena for the TFLM model.

        Returns:
            int: The size of the arena in bytes.
        """
        arena_size = primer.ModelInspector.crystal_dynamic_memory_usage(self.model)

        if verbose:
            print("Edge memory usage: {:.2f} kB".format(self.model.memory_usage / 1024))
            print("Crystal file size: {:.2f} kB".format(primer.ModelInspector.crystal_model_size(self.model) / 1024))
            print("Crystal RAM requirement: {:.2f} kB".format(arena_size / 1024))

        return arena_size


    def measure_execution_time(self):
        """
        Measures the execution time of the model.

        Returns:
            float: The execution time in ms.
        """
        rng = np.random.RandomState(42)
        sample_idx = rng.randint(0, self.dataset.train_x.shape[0])
        x = np.array(self.dataset.train_x[sample_idx])

        crystal_model_data = primer.Converter.convert(self.model, primer.Converter.ConversionTarget.CRYSTAL)
        if self.ekkono_model_type == "pretrained":
            crystal_model = crystal.load_predictive_model(crystal_model_data)
        elif self.ekkono_model_type == "incremental":
            crystal_model = crystal.load_incremental_predictive_model(crystal_model_data)

        # warm up
        tic = time.time()
        for i in range(10000):
            crystal_model.predict(x)
        toc = time.time()
        itr = int(10 * 10000 / (toc - tic))

        # run the test
        tic = time.time()
        for i in range(itr):
            crystal_model.predict(x)
        toc = time.time()
        execution_time = (toc-tic)/itr*1000     # in ms

        return execution_time


    def save_eqcheck_data(self, n_samples, save_dir):
        """
        Saves the eqcheck data as {"data_x", "data_y_pred"}.

        The data_x has shape (samples, *input_shape) and data_y_pred has shape (samples, *output_shape).

        Args:
            n_samples (int): The number of samples to be saved
            save_dir (str): The directory where the data should be saved
        """
        data_x = self.dataset.train_x[:n_samples]
        data_y_pred = []
        for i in range(len(data_x)):
            instance = data_x[i].tolist() + [0]  # the last element is a placeholder for the output
            data_y_pred.append(self.model.predict(instance))
        data_y_pred = np.array(data_y_pred)

        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, 'eqcheck_data.npz'), data_x=data_x, data_y_pred=data_y_pred)


    # Optional function
    @staticmethod
    def load_eqcheck_data(load_dir):
        """
        Loads the eqcheck data.

        Args:
            load_dir (str): The directory where the eqcheck data is stored.

        Returns:
            tuple: (data_x, data_y_pred)
        """
        eqcheck_data = np.load(os.path.join(load_dir, 'eqcheck_data.npz'))
        data_x = eqcheck_data['data_x']
        data_y_pred = eqcheck_data['data_y_pred']
        return data_x, data_y_pred


    def save_model(self, save_dir):
        """
        Saves the model.

        Args:
            save_dir (str): The directory where the model should be saved in.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save(os.path.join(save_dir, "model.edge"))


    def load_model(self, load_dir):
        """
        Loads the model.

        Args:
            load_dir (str): The parent directory where the SavedModel format is stored in.
        """
        if self.ekkono_model_type == "pretrained":
            self.model = primer.PredictiveModel.load(os.path.join(load_dir, "model.edge"))
        elif self.ekkono_model_type == "incremental":
            self.model = primer.IncrementalPredictiveModel.load(os.path.join(load_dir, "model.edge"))


    @staticmethod
    def log_model_to_wandb(model_dir, model_save_name):
        """
        Logs the model to wandb.

        Args:
            model_dir (str): The directory where the model is stored.
            model_save_name (str): The name that will be assigned to the model artifact.
        """
        model_path = os.path.join(model_dir, "model.edge")
        model_artifact = wandb.Artifact(model_save_name, type="model")
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)


    @staticmethod
    def save_model_info(model_info, save_dir):
        """
        Saves the model info.

        Args:
            model_info (dict): The model info.
            save_dir (str): The directory where the model info should be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        yaml.Dumper.ignore_aliases = lambda *args : True
        with open(os.path.join(save_dir, "model_info.yaml"), 'w') as f:
            yaml.dump(model_info, f, indent=4, sort_keys=False)


    def fill_crystal_templates(self, save_dir, eqcheck_data_dir, templates_dir):
        """
        Taken from the main.h/c and data.h/c available in the templates_dir, fills their
        placeholders with the appropriate data and saves them to the save_dir.

        Args:
            save_dir (str): The directory where the filled files should be saved.
            eqcheck_data_dir (str): The directory where the eqcheck data is stored.
            templates_dir (str): The directory where the templates are stored.
        """
        def _np_to_c(array):
            if array.ndim == 0:
                return str(array.item())
            c_array = "{" + ", ".join(_np_to_c(subarray) for subarray in array) + "}"
            return c_array

        # create model files
        with open(os.path.join(templates_dir, 'model.h'), 'r') as f:
            h_file = f.read()
        with open(os.path.join(templates_dir, 'model.c'), 'r') as f:
            c_file = f.read()

        h_file = h_file.replace("{model_type}", self.ekkono_model_type.upper())
        h_file = h_file.replace("{arena_size}", str(self._estimate_arena_size()))
        h_file = h_file.replace("{input_size}", str(self.dataset.feature_shape[0]))

        crystal_model_data = primer.Converter.convert(self.model, primer.Converter.ConversionTarget.CRYSTAL)

        c_file = c_file.replace("{model_data_size}", str(len(crystal_model_data)))

        model_data_str = ""
        for i, byte in enumerate(crystal_model_data):
            if i % 16 == 0:
                model_data_str += "\n\t"
            model_data_str += "0x{:02x}, ".format(byte)
        model_data_str = model_data_str[:-2] + "\n"

        c_file = c_file.replace("{model_data}", model_data_str)

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'model.h'), 'w') as f:
            f.write(h_file)
        with open(os.path.join(save_dir, 'model.c'), 'w') as f:
            f.write(c_file)

        # create data files
        data = np.load(os.path.join(eqcheck_data_dir, 'eqcheck_data.npz'))
        data_x = data['data_x']
        data_y = data['data_y_pred']
        data.close()
        assert len(data_x.shape) == 2, "The data_x should be 2-dimensional."
        assert len(data_y.shape) == 2, "The data_y should be 2-dimensional."
        assert len(data_x) == len(data_y), "The number of samples in data_x and data_y should be the same."
        assert data_y.shape[1] == 1, "The output size should be 1."

        with open(os.path.join(templates_dir, 'data.h'), 'r') as f:
            h_file = f.read()
        with open(os.path.join(templates_dir, 'data.c'), 'r') as f:
            c_file = f.read()

        h_file = h_file.replace("{n_samples}", str(data_x.shape[0]))
        h_file = h_file.replace("{n_features}", str(data_x.shape[1]))
        c_file = c_file.replace("{n_features}", str(data_x.shape[1]))

        data_x_str = "\n"
        for i, sample_x in enumerate(data_x):
            data_x_str += "\t" + _np_to_c(sample_x)
            if i < len(data_x) - 1:
                data_x_str += ","
            data_x_str += "\n"

        data_y_str = "\n"
        for i, sample_y in enumerate(data_y):
            data_y_str += "\t" + _np_to_c(sample_y)
            if i < len(data_y) - 1:
                data_y_str += ","
            data_y_str += "\n"

        c_file = c_file.replace("{samples_x}", data_x_str)
        c_file = c_file.replace("{samples_y}", data_y_str)

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'data.h'), 'w') as f:
            f.write(h_file)
        with open(os.path.join(save_dir, 'data.c'), 'w') as f:
            f.write(c_file)
