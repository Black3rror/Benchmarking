import importlib

import tensorflow as tf

from benchmarking.models.platforms.TensorFlow.model_template import ModelSupervisorTemplate


class ModelSupervisor(ModelSupervisorTemplate):
    def __init__(self, cfg=None):
        super().__init__()

        # default configs
        self.rnn_type = "SimpleRNN"
        self.embedding_dim = 32
        self.rnn_units = 64
        self.epochs = 50
        self.batch_size = 64
        self.dataset_info = {
            "name": "shakespeare",
            "path": "benchmarking/models/datasets/shakespeare/data.py",
            "args": {
                "sequence_length": 100,
                "test_ratio": 0.2,
                "random_seed": 42
            }
        }
        self.random_seed = 42

        self.learning_rate = 1e-3
        self.fine_tuning_learning_rate = 1e-4
        self.fine_tuning_epochs = 10
        self.fine_tuning_batch_size = 256

        # update configs if provided
        if cfg is not None:
            self.set_configs(cfg)

        # load corresponding dataset
        spec = importlib.util.spec_from_file_location("imported_module", self.dataset_info["path"])
        imported_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imported_module)

        self.dataset = imported_module.DatasetSupervisor(**self.dataset_info["args"])

        # The following arguments are required in oder to be able to convert the model to TFLite and TFLM
        stateful = False
        batch_size = 1
        sequence_length = self.dataset.sequence_length
        unroll = True
        self.model = self.create_model(self.rnn_type, self.dataset.input_size, self.dataset.output_size, self.dataset.output_activation, self.embedding_dim, self.rnn_units, stateful, batch_size, sequence_length, self.dataset.sequential_output, unroll, self.random_seed)


    def set_configs(self, cfg):
        if "rnn_type" in cfg:
            self.rnn_type = cfg["rnn_type"]
        if "embedding_dim" in cfg:
            self.embedding_dim = cfg["embedding_dim"]
        if "rnn_units" in cfg:
            self.rnn_units = cfg["rnn_units"]
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
    def create_model(rnn_type, input_size, output_size, output_activation, embedding_dim, rnn_units, stateful, batch_size, sequence_length, sequential_output, unroll, random_seed=None):
        """
        Creates the model.

        Args:
            rnn_type (str): The type of the RNN layer. It can be 'SimpleRNN', 'LSTM' or 'GRU'.
            input_size (int): The size of each element in the sequence.
            output_size (int): The size of the output.
            output_activation (str): The name of the activation function of the output layer.
            embedding_dim (int): The dimension of the embedding layer. If None, the embedding layer is not used.
            rnn_units (int): The number of units of the RNN layer.
            stateful (bool): If True, the RNN layer is stateful.
            batch_size (int): The batch size.
            sequence_length (int): The length of the sequences.
            sequential_output (bool): If True, the output is sequential.
            unroll (bool): If True, the RNN layer is unrolled.
            random_seed (int): The random seed.
        """
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
        model = tf.keras.models.Sequential()

        if embedding_dim is not None:
            model.add(tf.keras.layers.Embedding(
                input_dim=input_size,
                output_dim=embedding_dim,
                batch_input_shape=[batch_size, sequence_length]
            ))
        else:
            model.add(tf.keras.layers.InputLayer(input_shape=[sequence_length, input_size], batch_size=batch_size))

        if rnn_type == 'SimpleRNN':
            model.add(tf.keras.layers.SimpleRNN(
                units=rnn_units,
                return_sequences=sequential_output,
                stateful=stateful,
                unroll=unroll
            ))
        elif rnn_type == 'LSTM':
            model.add(tf.keras.layers.LSTM(
                units=rnn_units,
                return_sequences=sequential_output,
                stateful=stateful,
                unroll=unroll
            ))
        elif rnn_type == 'GRU':
            model.add(tf.keras.layers.GRU(
                units=rnn_units,
                return_sequences=sequential_output,
                stateful=stateful,
                unroll=unroll
            ))

        model.add(tf.keras.layers.Dense(output_size, activation=output_activation))

        return model


    def compile_model(self, fine_tuning=False):
        if not fine_tuning:
            learning_rate = self.learning_rate
        else:
            learning_rate = self.fine_tuning_learning_rate

        return self._compile_model(self.model, learning_rate, self.dataset.loss_function, self.dataset.metrics)


    @staticmethod
    def _compile_model(model, learning_rate, loss_function, metrics):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss=loss_function, metrics=metrics)


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

        # To do the training with the requested batch size, we'll create the same model but having the requested batch size
        stateful = False
        batch_size = self.batch_size
        sequence_length = self.dataset.sequence_length
        unroll = True
        training_model = self.create_model(self.rnn_type, self.dataset.input_size, self.dataset.output_size, self.dataset.output_activation, self.embedding_dim, self.rnn_units, stateful, batch_size, sequence_length, self.dataset.sequential_output, unroll, self.random_seed)
        training_model.set_weights(self.model.get_weights())
        self._compile_model(training_model, learning_rate=self.learning_rate, loss_function=self.dataset.loss_function, metrics=self.dataset.metrics)

        history = training_model.fit(self.dataset.train_x, self.dataset.train_y, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)

        self.model.set_weights(training_model.get_weights())

        return history


    @staticmethod
    def _generate_text(model, start_string, n_gen_char, char2index, index2char, temperature, random_seed=None):
        # temperature: the higher the temperature, the more random the generated text
        if char2index is None or index2char is None:
            raise ValueError('char2index or index2char are not set.')

        input_indices = [char2index[s] for s in start_string]
        input_indices = tf.expand_dims(input_indices, 0)

        model.reset_states()
        text_generated = []
        for _ in range(n_gen_char):
            predictions = model(input_indices)[0]
            logits = tf.math.log(predictions)

            # Using a categorical distribution to predict the character returned by the model.
            logits = logits / temperature
            predicted_id = tf.random.categorical(logits, num_samples=1, seed=random_seed)[-1,0].numpy()
            text_generated.append(index2char[predicted_id])

            input_indices = tf.expand_dims([predicted_id], 0)

        return (start_string + ''.join(text_generated))


    def evaluate_model(self):
        # Bonus: generate text
        # Create the same model, but this time with stateful RNN to more efficiently generate text
        try:
            if self.dataset.char2index is not None and self.dataset.index2char is not None:     # To avoid printing unnecessary exception message if the model was not supposed to generate text
                gen_model = self.create_model(self.rnn_type, self.dataset.input_size, self.dataset.output_size, self.dataset.output_activation, self.embedding_dim, self.rnn_units, stateful=True, batch_size=1, sequence_length=None, sequential_output=self.dataset.sequential_output, unroll=False, random_seed=self.random_seed)
                gen_model.set_weights(self.model.get_weights())
                print("Generated text:")
                print(self._generate_text(gen_model, start_string="ROMEO:\n", n_gen_char=1000, char2index=self.dataset.char2index, index2char=self.dataset.index2char, temperature=1.0, random_seed=self.random_seed))
                print()
        except Exception as e:
            print("Error in generating text: {}".format(e))
            print("Continuing without generating text")

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
            "rnn_type": self.rnn_type,
            "embedding_dim": self.embedding_dim,
            "rnn_units": self.rnn_units,
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
        return 300000
