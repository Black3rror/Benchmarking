Each *YAML* file in this directory or its subdirectories represents a target model. In order to exclude a target model from the benchmarking process, simply add *dot (.)* to the beginning of the file name. For example, to exclude the target model *target_model.yaml*, rename it to *.target_model.yaml*. All targets in a directory can be excluded by adding *dot (.)* to the beginning of the directory name.

Each target can be one of the following types:

- *CNN*: A combination of CNN layers followed by a number of fully connected layers. CNN layers can be left empty in order to create a fully connected network.
- *CNN_MBNet*: The MobileNetV2 network.
- *RNN*: Recurrent neural network, including simple RNN, LSTM, and GRU.
- *TinyMLPerf_AE*: The model used in the TinyMLPerf benchmarking suite for anomaly detection.
- *TinyMLPerf_DS_CNN*: The model used in the TinyMLPerf benchmarking suite for keyword spotting.
- *TinyMLPerf_MBNet*: The model used in the TinyMLPerf benchmarking suite for image classification.
- *TinyMLPerf_ResNet*: The model used in the TinyMLPerf benchmarking suite for image classification.

## CNN
The CNN target model should be defined like this:
```yaml
model_type: "CNN"
convs_params: [
    # each element creates a convolutional layer
    # [c, k, s] where c is the number of channels, k is the kernel size, and s is the stride. For example: [32, 3, 1]
    # k and s can be a tuple. For example: [32, [3, 1], [2, 1]]
    # if c is zero, but k and s have values, it means a max pooling layer. [0, 2, 2]
    # if c and k are zero, it means a global average pooling layer. [0, 0, 0]
]
denses_params: []       # each element creates a dense layer. For example: [128, 64] creates two dense layers with 128 and 64 units.
convs_dropout: 0.00     # should be a float between 0 and 1
denses_dropout: 0.00    # should be a float between 0 and 1
activation: "relu"      # activation function for all layers (except the output layer)
use_batch_norm: True    # whether to use batch normalization
epochs: 10              # number of epochs for training
batch_size: 32          # batch size for training
dataset:                # dataset configuration
  name: "sinus"         # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    n_samples: 100000
    test_ratio: 0.2
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```

## CNN_MBNet
The MobileNetV2 target model should be defined like this:
```yaml
model_type: "CNN_MBNet"
epochs: 10              # number of epochs for training
batch_size: 32          # batch size for training
dataset:                # dataset configuration
  name: "imagenet_v2"   # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    test_ratio: 0.2
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```

## RNN
The RNN target model should be defined like this:
```yaml
model_type: "RNN"
rnn_type: "LSTM"        # RNN type. Can be "SimpleRNN", "LSTM", or "GRU"
embedding_dim: null     # embedding dimension. If null, no embedding layer will be used
rnn_units: 64           # number of units in the RNN layer
epochs: 50              # number of epochs for training
batch_size: 64          # batch size for training
dataset:                # dataset configuration
  name: "randomset_seq" # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    n_samples: 1000
    test_ratio: 0.2
    input_size: 32
    output_size: 32
    sequence_length: 100
    sequential_output: true
    using_embedding: false
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```

## TinyMLPerf_AE
The TinyMLPerf AutoEncoder target model should be defined like this:
```yaml
model_type: "TinyMLPerf_AE"
load_pretrained_model: True   # whether to load the pretrained model or create a new one
# in case a new model is created, the following commented parameters will be used
# denses_params: [128, 128, 128, 128, 8, 128, 128, 128]   # each element creates a dense layer
# denses_dropout: 0.00  # should be a float between 0 and 1
# activation: "relu"    # activation function for all layers (except the output layer)
# use_batch_norm: True  # whether to use batch normalization
epochs: 0               # number of epochs for training
batch_size: 512         # batch size for training
dataset:                # dataset configuration
  name: "toyADMOS"      # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    n_mels: 128
    frames: 5
    n_fft: 1024
    hop_length: 512
    power: 2.0
    test_ratio: 0.1
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```

## TinyMLPerf_DS_CNN
The TinyMLPerf Depthwise Separable Convolutional Neural Network (DS-CNN) target model should be defined like this:
```yaml
model_type: "TinyMLPerf_DS_CNN"
load_pretrained_model: True   # whether to load the pretrained model or create a new one
# in case a new model is created, the following commented parameters will be used
# num_filters: 64       # number of filters in the convolutional layers
# activation: "relu"    # activation function for all layers (except the output layer)
# use_batch_norm: True  # whether to use batch normalization
epochs: 0               # number of epochs for training
batch_size: 100         # batch size for training
dataset:                # dataset configuration
  name: "randomset_classification"  # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    n_samples: 10000
    test_ratio: 0.2
    feature_shape: [49, 10, 1]
    num_labels: 12
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```

## TinyMLPerf_MBNet
The TinyMLPerf MobileNetV1 target model should be defined like this:
```yaml
model_type: "TinyMLPerf_MBNet"
load_pretrained_model: True   # whether to load the pretrained model or create a new one
# in case a new model is created, the following commented parameters will be used
# num_filters: 8        # number of filters in the first convolutional layer
# activation: "relu"    # activation function for all layers (except the output layer)
# use_batch_norm: True  # whether to use batch normalization
epochs: 0               # number of epochs for training
batch_size: 32          # batch size for training
dataset:                # dataset configuration
  name: "vww"           # dataset name. shouold match the directory name of the dataset in the datasets directory
  args:
    # arguments to be passed to the dataset class
    # for example, it can be as follows:
    image_size: [96, 96]
    dataset_ratio: 0.1
    test_ratio: 0.1
    flat_features: False
    random_seed: 42
random_seed: 42         # random seed for reproducibility. If null, a random seed will be used
```
