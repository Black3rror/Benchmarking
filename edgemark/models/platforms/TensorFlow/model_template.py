"""
This module contains a template class that other models should inherit from.
"""

import os
import time

import numpy as np
import tensorflow as tf
import wandb
import yaml
from tensorflow.python.profiler import model_analyzer, option_builder
from wandb.keras import WandbCallback


class ModelSupervisorTemplate:
    """
    This class is a template for TensorFlow models. In order to create a new model,
    you should inherit from this class and implement its abstract functions.
    """

    def __init__(self, cfg=None):
        """
        Initializes the class.
        The following attributes should be set in the __init__ function:
            self.model (tf.keras.Model): The model.
            self.dataset (DatasetSupervisorTemplate): The dataset.

        Args:
            cfg (dict): The configurations of the model. Defaults to None.
        """
        self.model = None
        self.dataset = None


    def set_configs(self, cfg):
        """
        Sets the configs from the given dictionary.

        Note: The changed configs won't affect the data, model, or any other loaded attributes.
        In case you want to change them, you should call the corresponding functions.

        Args:
            cfg (dict): The configurations.
        """
        raise NotImplementedError


    # Optional function: Whether you implement this function or not depends on your application.
    def compile_model(self, fine_tuning=False):
        """
        Compiles the model.

        Args:
            fine_tuning (bool, optional): If True, the model will be compiled for fine-tuning. Defaults to False.
        """
        raise NotImplementedError


    # Optional function: Whether you implement this function or not depends on your application.
    def train_model(self, fine_tuning=False, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
        """
        Trains the model.

        Args:
            fine_tuning (bool, optional): If True, the model will be trained for fine-tuning. Defaults to False.
            tensorboard_log_dir (str, optional): The directory where the logs should be saved. If None, the logs won't be saved. Defaults to None.
            best_weights_dir (str, optional): The directory where the best weights should be saved. If None, the best weights won't be saved. Defaults to None.
            use_wandb (bool, optional): If True, the training progress will be logged to W&B. Defaults to False.

        Returns:
            Optional[tf.keras.callbacks.History]: The training history or None.
        """
        raise NotImplementedError


    # Optional function: Whether you implement this function or not depends on your application.
    def evaluate_model(self):
        """
        Evaluates the model.

        Returns:
            dict: The evaluation metrics.
        """
        raise NotImplementedError


    def get_model_info(self):
        """
        Returns the model info that can be anything important, including its configuration.

        Returns:
            dict: The model info.
        """
        raise NotImplementedError


    def get_params_count(self):
        """
        Returns the number of parameters in the model.

        Returns:
            list[int]: The total number of parameters, the number of trainable parameters, and the number of non-trainable parameters.
        """
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        for layer in self.model.variables:
            total_params += np.prod(layer.shape)

        for layer in self.model.trainable_variables:
            trainable_params += np.prod(layer.shape)

        non_trainable_params = total_params - trainable_params

        return int(total_params), int(trainable_params), int(non_trainable_params)


    def get_FLOPs(self):
        """
        Returns the number of FLOPs of the model.

        Returns:
            int: The number of FLOPs.
        """
        input_signature = [
            tf.TensorSpec(
                shape=(1, *params.shape[1:]),
                dtype=params.dtype,
                name=params.name
            ) for params in self.model.inputs
        ]
        forward_graph = tf.function(self.model, input_signature).get_concrete_function().graph
        options = option_builder.ProfileOptionBuilder.float_operation()
        options['output'] = 'none'
        graph_info = model_analyzer.profile(forward_graph, options=options)

        FLOPs = graph_info.total_float_ops

        return FLOPs


    def measure_execution_time(self):
        """
        Measures the execution time of the model.
        The process starts by a warm-up phase for 100 iterations, then the execution time is measured for ~10 seconds.

        Returns:
            float: The execution time in ms.
        """
        rng = np.random.RandomState(42)
        sample_idx = rng.randint(0, self.dataset.train_x.shape[0])
        x = np.array([self.dataset.train_x[sample_idx]])

        # warm up
        tic = time.time()
        for i in range(100):
            self.model(x, training=False)
        toc = time.time()
        itr = int(10 * 100 / (toc - tic))

        # run the test
        tic = time.time()
        for i in range(itr):
            self.model(x, training=False)
        toc = time.time()
        execution_time = (toc-tic)/itr*1000     # in ms

        return execution_time


    def save_representative_data(self, n_samples, save_dir):
        """
        Saves the representative data with shape (samples, *input_shape).

        Args:
            n_samples (int): The number of samples to be saved.
            save_dir (str): The directory where the data should be saved.
        """
        data_x = self.dataset.train_x[:n_samples]
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'representative_data.npy'), data_x)


    # Optional function
    @staticmethod
    def load_representative_data(load_dir):
        """
        Loads the representative data.

        Args:
            load_dir (str): The directory where the representative data is stored.

        Returns:
            numpy.ndarray: The representative data.
        """
        representative_data = np.load(os.path.join(load_dir, 'representative_data.npy'))
        return representative_data


    def save_eqcheck_data(self, n_samples, save_dir):
        """
        Saves the eqcheck data as {"data_x", "data_y_pred"}.

        The data_x has shape (samples, *input_shape) and data_y_pred has shape (samples, *output_shape).

        Args:
            n_samples (int): The number of samples to be saved
            save_dir (str): The directory where the data should be saved
        """
        # sanity check (useful for when deploying the model using TFLM)
        for data_dim_size, model_dim_size in zip(self.dataset.train_x.shape[1:], self.model.inputs[0].shape[1:]):
            if data_dim_size != model_dim_size and model_dim_size is not None:
                raise ValueError("The shape of the train_x doesn't match the input shape of the model.")

        data_x = self.dataset.train_x[:n_samples]
        data_y_pred = self.model.predict(data_x, verbose=0)

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
        Saves the model in two formats: Keras and SavedModel.

        Args:
            save_dir (str): The directory where the model should be saved in.
        """
        # save the model as a Keras format
        os.makedirs(os.path.join(save_dir, "keras_format"), exist_ok=True)
        self.model.save(os.path.join(save_dir, "keras_format/model.keras"))

        # save the model as a SavedModel format
        os.makedirs(os.path.join(save_dir, "saved_model_format"), exist_ok=True)
        self.model.save(os.path.join(save_dir, "saved_model_format"))


    def load_model(self, load_dir):
        """
        Loads the model in the SavedModel format.

        Args:
            load_dir (str): The parent directory where the SavedModel format is stored in.
        """
        self.model = tf.keras.models.load_model(os.path.join(load_dir, "saved_model_format"))


    @staticmethod
    def log_model_to_wandb(model_dir, model_save_name):
        """
        Logs the model to W&B.

        Args:
            model_dir (str): The directory where the model is stored.
            model_save_name (str): The name that will be assigned to the model artifact.
        """
        model_path = os.path.join(model_dir, "keras_format/model.keras")
        model_artifact = wandb.Artifact(model_save_name, type="model")
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)


    def save_weights(self, save_dir):
        """
        Saves the model weights.

        Args:
            save_dir (str): The directory where the model weights should be saved.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_weights(os.path.join(save_dir, "weights"))


    def load_weights(self, load_dir):
        """
        Loads the model weights.

        Args:
            load_dir (str): The directory where the model weights are stored.
        """
        self.model.load_weights(os.path.join(load_dir, "weights"))


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


    def save_TFLM_info(self, save_path):
        """
        Saves the information required by the TFLM converter as a YAML file.
        This is to help TFLM converter in a later stage.

        Args:
            save_path (str): The YAML file path where the TFLM info should be saved.
        """
        TFLM_info = {}

        arena_size_base = self._estimate_arena_size()
        TFLM_info["arena_size"] = {
            "32bit": arena_size_base,
            "16bit": arena_size_base//2,
            "8bit": arena_size_base//4
        }

        TFLM_info["input_dims"] = [int(dim) for dim in self.model.inputs[0].shape[1:]]
        TFLM_info["output_dims"] = [int(dim) for dim in self.model.outputs[0].shape[1:]]

        TFLM_info["op_resolver_funcs"] = None
        try:
            TFLM_info["op_resolver_funcs"] = self._get_op_resolver_funcs()
        except NotImplementedError:
            print("The model doesn't have an implementation for the _get_op_resolver_funcs function. The placeholders should be filled manually.")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(TFLM_info, f, indent=4, sort_keys=False)


    def _get_training_callbacks(self, tensorboard_log_dir=None, best_weights_dir=None, use_wandb=False):
        """
        Returns the training callbacks.

        Args:
            tensorboard_log_dir (str, optional): The directory where the logs should be saved. If None, the logs won't be saved. Defaults to None.
            best_weights_dir (str, optional): The directory where the best weights should be saved. If None, the best weights won't be saved. Defaults to None.
            use_wandb (bool, optional): If True, the training progress will be logged to wandb. Defaults to False.
        """
        callbacks = []

        if tensorboard_log_dir is not None:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1))

        if best_weights_dir is not None:
            best_weights_path = os.path.join(best_weights_dir, "weights")
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_weights_path, save_best_only=True, save_weights_only=True, verbose=0))

        if use_wandb:
            callbacks.append(WandbCallback(save_model=False, log_weights=True, compute_flops=True))

        return callbacks


    def _estimate_arena_size(self):
        """
        Estimates the size of the arena for the TFLM model.
        This is to help the TFLM converter in a later stage.
        Note: Depending on your model architecture, you may need to override this function.

        Returns:
            int: The size of the arena in bytes.
        """

        # Note: Assuming a Sequential model. Also, assuming that TFLM is wise to do in-place operations if possible.

        def _mul_dims(dims):
            output = 1
            for dim in dims:
                output *= dim
            return output
        arena_size = 0
        layer_1_size = _mul_dims(self.model.layers[0].input_shape[1:])

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            elif isinstance(layer, tf.keras.layers.Dense):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
                layer_2_size = _mul_dims(layer.output_shape[1:])

            elif isinstance(layer, tf.keras.layers.Flatten):
                continue

            elif isinstance(layer, tf.keras.layers.Add):
                continue

            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                continue

            elif isinstance(layer, tf.keras.layers.Activation):
                continue

            elif isinstance(layer, tf.keras.layers.Dropout):
                continue

            elif isinstance(layer, tf.keras.layers.ReLU):
                continue

            elif isinstance(layer, tf.keras.layers.Softmax):
                continue

            else:
                raise ValueError("Unknown layer type: {}".format(layer))

            if layer_1_size + layer_2_size > arena_size:
                arena_size = layer_1_size + layer_2_size
            layer_1_size = layer_2_size

        arena_size = arena_size * 4     # 4 bytes for each float32
        return arena_size


    def _get_op_resolver_funcs(self):
        """
        Returns the operators needed to run the TFLM model.
        This is to help the TFLM converter in a later stage.
        Possible strings in the output can be found here:
            https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_mutable_op_resolver.h

        Returns:
            list[str]: The op resolvers to be called by the C++ code.

        Example:
            >>> get_op_resolver_funcs()
            ["AddFullyConnected()", "AddRelu()", "AddSoftmax()"]
        """
        output = []
        for layer in self.model.layers:
            # TODO: check if registration is needed for non-float32 operations (e.g., AddConv2D(tflite::Register_CONV_2D_INT8()))
            if isinstance(layer, tf.keras.layers.InputLayer):
                pass

            elif isinstance(layer, tf.keras.layers.Dense):
                if "AddFullyConnected()" not in output:
                    output.append("AddFullyConnected()")

            elif isinstance(layer, tf.keras.layers.Conv2D):
                if "AddConv2D()" not in output:
                    output.append("AddConv2D()")

            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                if "AddDepthwiseConv2D()" not in output:
                    output.append("AddDepthwiseConv2D()")

            elif isinstance(layer, tf.keras.layers.MaxPooling2D):
                if "AddMaxPool2D()" not in output:
                    output.append("AddMaxPool2D()")

            elif isinstance(layer, tf.keras.layers.AveragePooling2D):
                if "AddAveragePool2D()" not in output:
                    output.append("AddAveragePool2D()")

            elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
                if "AddMean()" not in output:
                    output.append("AddMean()")

            elif isinstance(layer, tf.keras.layers.ZeroPadding2D):
                if "AddPad()" not in output:
                    output.append("AddPad()")

            elif isinstance(layer, tf.keras.layers.Flatten):
                if "AddReshape()" not in output:
                    output.append("AddReshape()")

            elif isinstance(layer, tf.keras.layers.Add):
                if "AddAdd()" not in output:
                    output.append("AddAdd()")

            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                pass

            elif isinstance(layer, tf.keras.layers.Embedding):
                if "AddCast()" not in output:
                    output.append("AddCast()")
                if "AddGather()" not in output:
                    output.append("AddGather()")

            elif isinstance(layer, tf.keras.layers.SimpleRNN):
                if "AddReshape()" not in output:
                    output.append("AddReshape()")
                if "AddFullyConnected()" not in output:
                    output.append("AddFullyConnected()")
                if "AddAdd()" not in output:
                    output.append("AddAdd()")
                if "AddTanh()" not in output:
                    output.append("AddTanh()")
                if "AddPack()" not in output:
                    output.append("AddPack()")
                if "AddUnpack()" not in output:
                    output.append("AddUnpack()")
                if "AddQuantize()" not in output:       # needed for int8_only quantization
                    output.append("AddQuantize()")
                if "AddDequantize()" not in output:     # needed for int8_only quantization
                    output.append("AddDequantize()")

            elif isinstance(layer, tf.keras.layers.LSTM):
                if "AddReshape()" not in output:
                    output.append("AddReshape()")
                if "AddFullyConnected()" not in output:
                    output.append("AddFullyConnected()")
                if "AddAdd()" not in output:
                    output.append("AddAdd()")
                if "AddTanh()" not in output:
                    output.append("AddTanh()")
                if "AddPack()" not in output:
                    output.append("AddPack()")
                if "AddUnpack()" not in output:
                    output.append("AddUnpack()")
                if "AddSplit()" not in output:
                    output.append("AddSplit()")
                if "AddLogistic()" not in output:
                    output.append("AddLogistic()")
                if "AddMul()" not in output:
                    output.append("AddMul()")
                if "AddQuantize()" not in output:       # needed for int8_only quantization
                    output.append("AddQuantize()")
                if "AddDequantize()" not in output:     # needed for int8_only quantization
                    output.append("AddDequantize()")

            elif isinstance(layer, tf.keras.layers.GRU):
                if "AddReshape()" not in output:
                    output.append("AddReshape()")
                if "AddFullyConnected()" not in output:
                    output.append("AddFullyConnected()")
                if "AddAdd()" not in output:
                    output.append("AddAdd()")
                if "AddTanh()" not in output:
                    output.append("AddTanh()")
                if "AddPack()" not in output:
                    output.append("AddPack()")
                if "AddUnpack()" not in output:
                    output.append("AddUnpack()")
                if "AddSplit()" not in output:
                    output.append("AddSplit()")
                if "AddLogistic()" not in output:
                    output.append("AddLogistic()")
                if "AddMul()" not in output:
                    output.append("AddMul()")
                if "AddSub()" not in output:
                    output.append("AddSub()")
                if "AddSplitV()" not in output:
                    output.append("AddSplitV()")


            elif isinstance(layer, tf.keras.layers.Activation):
                pass

            elif isinstance(layer, tf.keras.layers.Dropout):
                pass

            elif isinstance(layer, tf.keras.layers.ReLU):
                if "AddRelu()" not in output:
                    output.append("AddRelu()")

            elif isinstance(layer, tf.keras.layers.Softmax):
                if "AddSoftmax()" not in output:
                    output.append("AddSoftmax()")

            else:
                raise ValueError("Unknown layer type: {}".format(layer))

            try:
                if layer.activation is tf.keras.activations.sigmoid:
                    if "AddLogistic()" not in output:
                        output.append("AddLogistic()")

                elif layer.activation is tf.keras.activations.tanh:
                    if "AddTanh()" not in output:
                        output.append("AddTanh()")

                elif layer.activation is tf.keras.activations.relu:
                    if "AddRelu()" not in output:
                        output.append("AddRelu()")

                elif layer.activation is tf.keras.activations.softmax:
                    if "AddSoftmax()" not in output:
                        output.append("AddSoftmax()")

            except Exception as _:
                pass

        return output
