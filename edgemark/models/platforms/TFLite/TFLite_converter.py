import argparse
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import traceback
import zipfile

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from edgemark.models.utils.utils import get_abs_path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
config_file_path = "edgemark/models/platforms/TFLite/configs/TFLite_converter_config.yaml"
config_file_path = get_abs_path(config_file_path)
progress_bar_format = '{l_bar}{bar:30}| [{n_fmt}/{total_fmt} - {elapsed}]{postfix}'


class _ProcessResult:
    def __init__(self):
        self.result = None
        self.exception = None
        self.timed_out = False


def _put_output_to_buffer(result_buf, function, *args, **kwargs):
    process_result = _ProcessResult()
    try:
        process_result.result = function(*args, **kwargs)
    except Exception:
        process_result.exception = traceback.format_exc()
    result_buf.put(process_result)


def _run_as_process(timeout, function, *args, **kwargs):
    """
    Run the provided callable in a separate process and capture the output.
    Note: The callable should return something other than None to indicate success.

    Args:
        timeout (float): The timeout for the process in seconds.
        function (callable): The callable to run.
        args (tuple): The positional arguments to pass to the callable.
        kwargs (dict): The keyword arguments to pass to the callable.

    Returns:
        tuple: A tuple of the success flag and the process result.
    """
    result_buf = multiprocessing.Queue()
    args = (result_buf, function) + args

    p = multiprocessing.Process(target=_put_output_to_buffer, args=args, kwargs=kwargs)

    p.start()
    p.join(timeout=timeout)

    timed_out = False
    if p.is_alive():
        timed_out = True
        p.terminate()

    if not result_buf.empty():
        process_result = result_buf.get()
    else:
        process_result = _ProcessResult()
    process_result.timed_out = timed_out

    # We assume that function returns something
    success = process_result.result is not None and process_result.exception is None and process_result.timed_out is False

    return success, process_result


def _run_in_silence(function, *args, **kwargs):
    """
    Run the provided callable in silence and capture the output.

    Args:
        function (callable): The callable to run.
        args (tuple): The positional arguments to pass to the callable.
        kwargs (dict): The keyword arguments to pass to the callable.

    Returns:
        tuple: A tuple of the result and the captured output.
    """
    # Save the original file descriptors
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Create temporary files to capture the outputs
    with tempfile.TemporaryFile(mode='w+b') as temp_stdout, \
         tempfile.TemporaryFile(mode='w+b') as temp_stderr:
        # Redirect stdout and stderr to the temporary files
        os.dup2(temp_stdout.fileno(), original_stdout_fd)
        os.dup2(temp_stderr.fileno(), original_stderr_fd)

        # Call the provided callable and capture the output
        result = function(*args, **kwargs)

        # Flush any buffered output
        sys.stdout.flush()
        sys.stderr.flush()

        # Restore the original file descriptors
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)

        # Read the captured output from the temporary files
        temp_stdout.seek(0)
        captured_stdout = temp_stdout.read().decode()
        temp_stderr.seek(0)
        captured_stderr = temp_stderr.read().decode()

    message = captured_stdout + captured_stderr
    return result, message


def _convert_and_save(save_dir, model_path, conversion_funcs):
    """
    Convert the model using the provided conversion functions and save it.
    In case of an exception, the exception message is saved in the 'exception.txt' file.

    Args:
        save_dir (str): The directory to save the model.
        model_path (str): The path to the TensorFlow model.
        conversion_funcs (list): The list of conversion functions and their arguments. Each element should be a dictionary with the following keys:
            - func (callable): The conversion function. The first argument is always populated with the model path.
            - args (tuple): The positional arguments to pass to the conversion function (beginning from the second argument).
            - kwargs (dict): The keyword arguments to pass to the conversion function.
            Note: The last function should return a tflite model, and the others should return a TensorFlow model.

    Returns:
        bool: True if the conversion and saving were successful, False otherwise.
    """
    try:
        for i, element in enumerate(conversion_funcs):
            func = element['func']
            args = element['args'] if 'args' in element else ()
            kwargs = element['kwargs'] if 'kwargs' in element else {}

            model = func(model_path, *args, **kwargs)

            if i != len(conversion_funcs) - 1:      # if not the last function, the model should be saved
                model_path = os.path.join(save_dir, "tmp", "model")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                model.save(model_path)

        if os.path.exists(os.path.join(save_dir, "tmp")):
            shutil.rmtree(os.path.join(save_dir, "tmp"))

        tflite_model = model
        _save_tflite_model(save_dir, tflite_model, delete_exception_file=True)

    except Exception:
        e = traceback.format_exc()
        _save_tflite_exception(str(e), save_dir, delete_model=True)

    success = os.path.exists(os.path.join(save_dir, 'model.tflite'))
    return success


def basic_convert(model_path):
    """
    Convert the model without any optimizations.

    Args:
        model_path (str): The path to the TensorFlow model.

    Returns:
        bytes: The TFLite model.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def quantize(model_path, representative_data=None, int_only=False, a16_w8=False, float16=False):
    """
    Quantize the model.

    Args:
        model_path (str): The path to the TensorFlow model.
        representative_data (numpy.ndarray): The representative data for quantization.
        int_only (bool): Whether to force the model to use integer quantization only.
        a16_w8 (bool): Whether to quantize the model to use 16-bit activations and 8-bit weights.
        float16 (bool): Whether to quantize the model to use 16-bit floating point numbers.

    Returns:
        bytes: The TFLite model.
    """
    # sanity check
    if float16 and ((representative_data is not None) or int_only or a16_w8):
        print("Warning: float16 came with int_only, a16_w8, or representative_data. Are you sure you want to do this?")
    if int_only and (representative_data is None):
        print("Warning: int_only requires representative_data. You will receive an error.")
    if a16_w8 and representative_data is None and int_only is False and float16 is False:
        print("Warning: Just setting a16_w8 is possible but not common. Are you sure you want to do this?")

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_data is not None:
        def representative_dataset():
            for sample in representative_data:
                    sample = np.expand_dims(sample, axis=0)     # batch_size = 1
                    yield [sample]      # set sample as the first (and only) input of the model
        converter.representative_dataset = representative_dataset

    if int_only:
        if a16_w8:
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
        else:
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    if a16_w8:
        if int_only:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        else:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    else:
        if int_only:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        else:
            pass

    if float16:
        converter.target_spec.supported_types = [tf.float16]

    tflite_quant_model = converter.convert()
    return tflite_quant_model


def prune(model_path, target_sparsity):
    """
    Prune the model.

    Args:
        model_path (str): The path to the TensorFlow model.
        target_sparsity (float): The target sparsity.

    Returns:
        tf.keras.Model: The pruned model.
    """
    model = tf.keras.models.load_model(model_path)

    pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(target_sparsity=target_sparsity, begin_step=0, frequency=1)
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    train_x = np.random.rand(1, *model.input.shape[1:])
    train_y = np.random.rand(1, *model.output.shape[1:])

    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    opt = tf.keras.optimizers.SGD(learning_rate=0)

    model_for_pruning.compile(optimizer=opt, loss='mse')
    model_for_pruning.fit(train_x, train_y, epochs=1, callbacks=callbacks, verbose=0)

    stripped_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    return stripped_model


def cluster(model_path, n_clusters):
    """
    Cluster the model.

    Args:
        model_path (str): The path to the TensorFlow model.
        n_clusters (int): The number of clusters.

    Returns:
        tf.keras.Model: The clustered model.
    """
    model = tf.keras.models.load_model(model_path)

    clustered_model = tfmot.clustering.keras.cluster_weights(model, number_of_clusters=n_clusters)

    stripped_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    return stripped_model


def _save_tflite_model(save_dir, tflite_model, delete_exception_file=False):
    """
    Save the TFLite model and its zipped format.

    Args:
        save_dir (str): The directory to save the model.
        tflite_model (bytes): The TFLite model.
        delete_exception_file (bool): Whether to delete the exception file if it exists.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save tflite model
    with open(os.path.join(save_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)

    # save zip model
    with zipfile.ZipFile(os.path.join(save_dir, 'model.zip'), 'w', compression = zipfile.ZIP_DEFLATED) as f:
        f.write(os.path.join(save_dir, 'model.tflite'))

    # delete exception file if it exists
    if delete_exception_file:
        if os.path.exists(os.path.join(save_dir, 'exception.txt')):
            os.remove(os.path.join(save_dir, 'exception.txt'))


def _save_tflite_exception(message, save_dir, delete_model=False):
    """
    Save the exception message and delete the model if requested.

    Args:
        message (str): The exception message.
        save_dir (str): The directory to save the exception message.
        delete_model (bool): Whether to delete the model if it exists.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save the exception message
    with open(os.path.join(save_dir, 'exception.txt'), 'w') as f:
        f.write(message)

    # delete tflite related files if they exist
    if delete_model:
        if os.path.exists(os.path.join(save_dir, 'model.tflite')):
            os.remove(os.path.join(save_dir, 'model.tflite'))
        if os.path.exists(os.path.join(save_dir, 'model.zip')):
            os.remove(os.path.join(save_dir, 'model.zip'))
        if os.path.exists(os.path.join(save_dir, 'tmp')):
            shutil.rmtree(os.path.join(save_dir, 'tmp'))


def main(cfg_path=config_file_path, **kwargs):
    """
    Convert the TensorFlow models to TFLite models with the specified optimizations.

    Args:
        cfg_path (str): The path to the configuration file.
            The configuration file that this path points to should contain the following keys:
                - model_base_dir (str): A placeholder for the model base directory. This will be populated by the target directory.
                - linkers_dir (str): Path to the directory where the generated models list is loaded from and the converted models list will be saved.
                - tf_model_path (str): Path to the TensorFlow model.
                - representative_data_path (str): Path to the representative data for quantization. The file should be a numpy file.
                - tflite_save_dir (str): Path to the directory to save the converted models and their assets.
                - conversion_timeout (float): The timeout for each conversion in seconds.
                - optimizations (list): The optimizations to apply during conversion. In each element/group, the optimizations should be separated by '+'.
        **kwargs (dict): Keyword arguments to be passed to the configuration file.

    Returns:
        list: A list of dictionaries containing the following keys for each target model:
            - dir (str): The directory of the target model.
            - flavors (list): A list of dictionaries containing the following keys for each optimization:
                - flavor (str): The optimization flavor.
                - result (str): The result of the conversion. It can be either "success" or "failed".
                - error (str): The error message in case of failure.
                - traceback (str): The traceback in case of failure. Either this or 'exception_file' will be present.
                - exception_file (str): The path to the exception file in case of failure. Either this or 'traceback' will be present.
    """
    cfg = OmegaConf.load(cfg_path, **kwargs)
    cfg.update(OmegaConf.create(kwargs))

    targets = OmegaConf.load(os.path.join(cfg.linkers_dir, "tf_generated_models_list.yaml"))
    converted_models_list = []
    output = [{"dir": target_dir, "flavors": []} for target_dir in targets]

    for i, target_dir in enumerate(targets):
        cfg.model_base_dir = target_dir
        print("Converting the model in: {} ({}/{})".format(cfg.tf_model_path, i+1, len(targets)))
        print("Saving to: {}".format(cfg.tflite_save_dir))

        conversions_list = []
        model_path = cfg.tf_model_path
        saving_root = cfg.tflite_save_dir
        opt_groups = cfg.optimizations
        representative_data_path = cfg.representative_data_path
        representative_data = None

        finishers = ["basic", "q_dynamic", "q_full_int", "q_full_int_only", "q_16x8", "q_16x8_int_only", "q_float16"]

        # sanity checks
        for opt_group in opt_groups:
            try:
                opts = [opt.strip() for opt in opt_group.split('+')]

                if "q_full_int" in opts or "q_full_int_only" in opts or "q_16x8" in opts or "q_16x8_int_only" in opts:
                    if not os.path.exists(representative_data_path):
                        raise FileNotFoundError("representative_dataset_path is required for full_int, full_int_only, 16x8, and 16x8_int_only optimizations, but {} does not exist.".format(representative_data_path))
                    representative_data = np.load(representative_data_path)

                for opt in opts[:-1]:
                    if opt in finishers:
                        raise ValueError("The {} optimization should be the last one in its group ({}).".format(opt, opt_group))

            except Exception as e:
                output[i]["flavors"].append({
                    "flavor": opt_group,
                    "result": "failed",
                    "error": type(e).__name__,
                    "traceback": traceback.format_exc()
                })
                print("Error:")
                print(traceback.format_exc())
                opt_groups.remove(opt_group)

        progress_bar = tqdm(total=len(opt_groups), desc="TFLite conversion", colour='green', bar_format=progress_bar_format, leave=True)
        errors = []
        convertion_messages = ""

        for opt_group in opt_groups:
            try:
                opts = [opt.strip() for opt in opt_group.split('+')]

                if opts[-1] not in finishers:
                    opts.append("basic")

                conversion_funcs = []
                for i, opt in enumerate(opts):
                    if opt == "basic":
                        conversion_funcs.append({"func": basic_convert, "args": ()})
                    elif opt == "q_dynamic":
                        conversion_funcs.append({"func": quantize, "args": ()})
                    elif opt == "q_full_int":
                        conversion_funcs.append({"func": quantize, "args": (representative_data,)})
                    elif opt == "q_full_int_only":
                        conversion_funcs.append({"func": quantize, "args": (representative_data,), "kwargs": {"int_only": True}})
                    elif opt == "q_16x8":
                        conversion_funcs.append({"func": quantize, "args": (representative_data,), "kwargs": {"a16_w8": True}})
                    elif opt == "q_16x8_int_only":
                        conversion_funcs.append({"func": quantize, "args": (representative_data,), "kwargs": {"a16_w8": True, "int_only": True}})
                    elif opt == "q_float16":
                        conversion_funcs.append({"func": quantize, "args": (), "kwargs": {"float16": True}})
                    elif opt.startswith("p_") and opt[2:].isdigit():
                        target_sparsity = float(opt[2:]) / 100
                        assert 0 <= target_sparsity <= 1, "Sparsity should be between 0 and 1."
                        conversion_funcs.append({"func": prune, "args": (target_sparsity,)})
                    elif opt.startswith("c_") and opt[2:].isdigit():
                        n_clusters = int(opt[2:])
                        assert n_clusters > 0, "Number of clusters should be greater than 0."
                        conversion_funcs.append({"func": cluster, "args": (n_clusters,)})
                    else:
                        raise ValueError("Unknown optimization: {}".format(opt))

                progress_bar.set_postfix_str("{}".format(opt_group))
                save_dir = os.path.join(saving_root, opt_group)
                convert_and_save_args = (save_dir, model_path, conversion_funcs)

                process_success, process_result = _run_as_process(cfg.conversion_timeout, _run_in_silence, _convert_and_save, *convert_and_save_args)

                if process_success:
                    convertion_success, _ = process_result.result
                else:
                    convertion_success = False
                    failure_report = "Process failed.\nResult: {}\nException: {}\nTimed out: {}\n".format(process_result.result, process_result.exception, process_result.timed_out)
                    if process_result.result is None and process_result.exception is None and process_result.timed_out is False:
                        failure_report += "Possible crash of the process\n"
                    _save_tflite_exception(failure_report, os.path.join(saving_root, opt_group), delete_model=True)

                if convertion_success:
                    conversions_list.append(opt_group)
                    _, message = process_result.result
                    convertion_messages += "="*80 + "\n" + "{}:\n".format(opt_group) + message + "\n\n"

                    output[i]["flavors"].append({
                        "flavor": opt_group,
                        "result": "success"
                    })

                else:
                    convertion_messages += "="*80 + "\n" + "{}:\n".format(opt_group) + "Conversion failed\n\n"
                    errors.append(opt_group)

                    output[i]["flavors"].append({
                        "flavor": opt_group,
                        "result": "failed",
                        "error": "Conversion failed",
                        "exception_file": os.path.join(saving_root, opt_group, 'exception.txt')
                    })

                progress_bar.update(1)

            except Exception as e:
                errors.append(opt_group)
                convertion_messages += "="*80 + "\n" + "{}:\n".format(opt_group) + traceback.format_exc() + "\n\n"

                output[i]["flavors"].append({
                    "flavor": opt_group,
                    "result": "failed",
                    "error": type(e).__name__,
                    "traceback": traceback.format_exc()
                })

                progress_bar.update(1)

        progress_bar.set_postfix_str("Done")
        progress_bar.close()

        converted_models_element = {
            "model_base_dir": cfg.model_base_dir,
            "conversions": conversions_list
        }
        converted_models_list.append(converted_models_element)

        # Save the conversion messages
        with open(os.path.join(saving_root, 'conversion_messages.txt'), 'w') as f:
            f.write(convertion_messages)

        if len(errors) > 0:
            print("Errors:")
            for error in errors:
                print("    " + error)
            print("Please check the exception files for details.")

        print()

    # Save the list of converted models
    os.makedirs(cfg.linkers_dir, exist_ok=True)
    with open(os.path.join(cfg.linkers_dir, 'tflite_converted_models_list.yaml'), 'w') as f:
        yaml.dump(converted_models_list, f, indent=4, sort_keys=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the TensorFlow models to TFLite models with the specified optimizations.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="The path to the configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
