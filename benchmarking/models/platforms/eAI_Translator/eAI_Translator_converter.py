import argparse
import logging
import os
import traceback

import numpy as np
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

from benchmarking.models.utils.utils import get_abs_path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
config_file_path = "benchmarking/models/platforms/eAI_Translator/configs/eAI_Translator_converter_config.yaml"
config_file_path = get_abs_path(config_file_path)


def _save_translator_exception(message, save_dir, delete_assets=False):
    """
    Save the exception message and delete the assets if requested.

    Args:
        message (str): The exception message.
        save_dir (str): The directory to save the exception message.
        delete_assets (bool): Whether to delete the eAI Translator related assets.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save the exception message
    with open(os.path.join(save_dir, 'exception.txt'), 'w') as f:
        f.write(message)

    # delete translator related files if they exist
    if delete_assets:
        if os.path.exists(os.path.join(save_dir, 'data.h')):
            os.remove(os.path.join(save_dir, 'data.h'))
        if os.path.exists(os.path.join(save_dir, 'data.c')):
            os.remove(os.path.join(save_dir, 'data.c'))


def create_data_source_files(tflite_model_path, eqcheck_data_path, templates_dir, save_dir):
    """
    Create the data source files for the eAI Translator model.

    Args:
        tflite_model_path (str): The path to the TensorFlow Lite model.
        eqcheck_data_path (str): The path to the equality check data.
        templates_dir (str): The directory containing the templates.
        save_dir (str): The directory to save the data source files.
    """
    def _np_to_c(array):
        array = np.array(array)
        array = array.reshape(-1)
        c_array = "{" + ", ".join([str(x) for x in array]) + "}"
        return c_array

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_scale, in_zero_point = input_details[0]['quantization']
    out_scale, out_zero_point = output_details[0]['quantization']
    if input_details[0]['dtype'] is np.float32 and output_details[0]['dtype'] is np.float32:
        in_out_dtype = "float"
    elif input_details[0]['dtype'] is np.int16 and output_details[0]['dtype'] is np.int16:
        in_out_dtype = "int16"
    elif input_details[0]['dtype'] is np.int8 and output_details[0]['dtype'] is np.int8:
        in_out_dtype = "int8"
    else:
        raise ValueError("Unknown input and output types: {} and {}".format(input_details[0]['dtype'], output_details[0]['dtype']))

    data = np.load(eqcheck_data_path)
    data_x = data['data_x']
    data_y = data['data_y_pred']
    data.close()

    with open(os.path.join(templates_dir, 'data.h'), 'r') as f:
        h_file = f.read()

    with open(os.path.join(templates_dir, 'data.c'), 'r') as f:
        c_file = f.read()

    h_file = h_file.replace("{n_samples}", str(data_x.shape[0]))
    c_file = c_file.replace("{n_samples}", str(data_x.shape[0]))

    data_x_size = np.prod(data_x.shape[1:])
    data_y_size = np.prod(data_y.shape[1:])
    h_file = h_file.replace("{samples_x_size}", str(data_x_size))
    h_file = h_file.replace("{samples_y_size}", str(data_y_size))
    c_file = c_file.replace("{samples_x_size}", str(data_x_size))
    c_file = c_file.replace("{samples_y_size}", str(data_y_size))

    if in_out_dtype == "float":
        h_file = h_file.replace("{samples_x_dtype}", "float")
        h_file = h_file.replace("{samples_y_dtype}", "float")
        c_file = c_file.replace("{samples_x_dtype}", "float")
        c_file = c_file.replace("{samples_y_dtype}", "float")
    elif in_out_dtype == "int8":
        h_file = h_file.replace("{samples_x_dtype}", "int8_t")
        h_file = h_file.replace("{samples_y_dtype}", "int8_t")
        c_file = c_file.replace("{samples_x_dtype}", "int8_t")
        c_file = c_file.replace("{samples_y_dtype}", "int8_t")
        data_x = (data_x / in_scale) + in_zero_point
        data_y = (data_y / out_scale) + out_zero_point
        data_x = np.clip(data_x, -128, 127)
        data_y = np.clip(data_y, -128, 127)
        data_x = data_x.astype(np.int8)
        data_y = data_y.astype(np.int8)
    elif in_out_dtype == "int16":
        h_file = h_file.replace("{samples_x_dtype}", "int16_t")
        h_file = h_file.replace("{samples_y_dtype}", "int16_t")
        c_file = c_file.replace("{samples_x_dtype}", "int16_t")
        c_file = c_file.replace("{samples_y_dtype}", "int16_t")
        data_x = (data_x / in_scale) + in_zero_point
        data_y = (data_y / out_scale) + out_zero_point
        data_x = np.clip(data_x, -32768, 32767)
        data_y = np.clip(data_y, -32768, 32767)
        data_x = data_x.astype(np.int16)
        data_y = data_y.astype(np.int16)
    else:
        raise ValueError("Unknown in_out_dtype: {}".format(in_out_dtype))

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


def main(cfg_path=config_file_path, **kwargs):
    """
    Create the data source files for the eAI Translator model.

    Args:
        cfg_path (str): The path to the configuration file.
            The configuration file that this path points to should contain the following keys:
                - model_base_dir (str): A placeholder for the model base directory. This will be populated by the target directory.
                - tflite_conversion_type (str): A placeholder for the TFLite conversion type. This will be populated by the conversion type.
                - linkers_dir (str): Path to the directory where the tflite converted models list is loaded from and the list of converted models is saved.
                - tflite_model_path (str): The path to the TensorFlow Lite model.
                - eqcheck_data_path (str): The path to the equality check data.
                - data_templates_dir (str): Path to the directory containing the data templates.
                - translator_save_dir (str): The directory to save the eAI Translator files.
        **kwargs (dict): Keyword arguments to be passed to the configuration file.

    Returns:
        list: A list of dictionaries containing the following keys for each target model:
            - dir (str): The directory of the target model.
            - flavors (list): A list of dictionaries containing the following keys for each optimization:
                - flavor (str): The optimization flavor.
                - result (str): The result of the conversion. It can be either "success" or "failed".
                - error (str): The error message in case of failure.
                - traceback (str): The traceback in case of failure.
    """
    cfg = OmegaConf.load(cfg_path, **kwargs)
    cfg.update(OmegaConf.create(kwargs))

    targets = OmegaConf.load(os.path.join(cfg.linkers_dir, "tflite_converted_models_list.yaml"))

    output = [{"dir": base_model["model_base_dir"], "flavors": []} for base_model in targets]

    converted_models_list = []
    for i, base_model in enumerate(targets):
        print("\Creating data files for the model in: {} ({}/{})".format(base_model["model_base_dir"], i+1, len(targets)))

        for model_flavor in base_model["conversions"]:
            if model_flavor in ["basic", "q_full_int_only"]:
                txt = "Model type: {} ...".format(model_flavor)
                txt += " " * (32 - len(txt))
                print(txt, end=" ", flush=True)
                cfg.model_base_dir = base_model["model_base_dir"]
                cfg.tflite_conversion_type = model_flavor

                try:
                    create_data_source_files(cfg.tflite_model_path, cfg.eqcheck_data_path, cfg.data_templates_dir, cfg.translator_save_dir)
                    if os.path.exists(os.path.join(cfg.translator_save_dir, 'exception.txt')):
                        os.remove(os.path.join(cfg.translator_save_dir, 'exception.txt'))
                    converted_models_list.append(cfg.translator_save_dir)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "success"
                    })
                    print("Done")
                except Exception as e:
                    error_message = traceback.format_exc()
                    _save_translator_exception(error_message, cfg.translator_save_dir, delete_assets=True)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "failed",
                        "error": type(e).__name__,
                        "traceback": traceback.format_exc()
                    })
                    print("Failed")

    os.makedirs(cfg.linkers_dir, exist_ok=True)
    with open(os.path.join(cfg.linkers_dir, 'translator_converted_models_list.yaml'), 'w') as f:
        yaml.dump(converted_models_list, f, indent=4, sort_keys=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create the data source files for the eAI Translator model.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
    print("\nThe data files are ready. Please use the Renesas eAI Translator to convert the tflite models to C++ code.")
