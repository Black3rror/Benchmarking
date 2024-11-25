import argparse
import logging
import os
import traceback

import numpy as np
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

from edgemark.models.platforms.TFLite.TFLite_converter import save_random_eqcheck_data
from edgemark.models.utils.utils import get_abs_path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
config_file_path = "edgemark/models/platforms/TFLM/configs/TFLM_converter_config.yaml"
config_file_path = get_abs_path(config_file_path)


def create_tflm_source_files(tflite_model_path, eqcheck_data_path, tflm_info_path, templates_dir, save_dir):
    """
    Create the TFLM model and data source files.

    Args:
        tflite_model_path (str): The path to the TensorFlow Lite model.
        eqcheck_data_path (str): The path to the data file for the equivalence check. The file should be a numpy file with 'data_x' and 'data_y_pred' arrays.
        tflm_info_path (str): The path to the TFLM info file. This should be a YAML file containing the following keys:
            - input_dims (list): The input dimensions of the model.
            - output_dims (list): The output dimensions of the model.
            - arena_size (dict): The arena sizes for different data types. The keys should be '8bit', '16bit', and '32bit'.
            - op_resolver_funcs (list): The list of strings. These should be the operator resolver functions to be added.
        templates_dir (str): The directory containing the template files for the model and data source files.
        save_dir (str): The directory to save the source files.
    """
    def _np_to_c(array):
        if array.ndim == 0:
            return str(array.item())
        c_array = "{" + ", ".join(_np_to_c(subarray) for subarray in array) + "}"
        return c_array

    tflm_info = OmegaConf.load(tflm_info_path)
    tflite_model = open(tflite_model_path, "rb").read()

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
    elif input_details[0]['dtype'] is np.uint8 and output_details[0]['dtype'] is np.uint8:
        in_out_dtype = "uint8"
    else:
        raise ValueError("Unknown input and output types: {} and {}".format(input_details[0]['dtype'], output_details[0]['dtype']))

    if in_out_dtype == "float" and tflm_info.op_resolver_funcs is not None:
        for tensor in interpreter.get_tensor_details():
            if tensor['dtype'] not in [np.float32, np.int32]:       # not the basic quantization either
                tflm_info.op_resolver_funcs += ["AddQuantize()", "AddDequantize()"]
                break

    # create model files
    with open(os.path.join(templates_dir, 'model.h'), 'r') as f:
        h_file = f.read()

    with open(os.path.join(templates_dir, 'model.cpp'), 'r') as f:
        cpp_file = f.read()

    if in_out_dtype == "float":
        h_file = h_file.replace("{input_dtype}", "float")
        h_file = h_file.replace("{output_dtype}", "float")
        h_file = h_file.replace("{arena_size}", str(tflm_info.arena_size["32bit"] + 10240))     # adding 10kB for safety
    elif in_out_dtype == "int8":
        h_file = h_file.replace("{input_dtype}", "int8_t")
        h_file = h_file.replace("{output_dtype}", "int8_t")
        h_file = h_file.replace("{arena_size}", str(tflm_info.arena_size["8bit"] + 10240))
    elif in_out_dtype == "uint8":
        h_file = h_file.replace("{input_dtype}", "uint8_t")
        h_file = h_file.replace("{output_dtype}", "uint8_t")
        h_file = h_file.replace("{arena_size}", str(tflm_info.arena_size["8bit"] + 10240))
    elif in_out_dtype == "int16":
        h_file = h_file.replace("{input_dtype}", "int16_t")
        h_file = h_file.replace("{output_dtype}", "int16_t")
        h_file = h_file.replace("{arena_size}", str(tflm_info.arena_size["16bit"] + 10240))
    else:
        raise ValueError("Unknown in_out_dtype: {}".format(in_out_dtype))

    h_file = h_file.replace("{input_n_dims}", str(len(tflm_info.input_dims)))
    h_file = h_file.replace("{output_n_dims}", str(len(tflm_info.output_dims)))

    input_dims_size_str = ""
    for i, dim in enumerate(tflm_info.input_dims):
        if i > 0:
            input_dims_size_str += "\n"
        input_dims_size_str += "#define INPUT_DIM_{i}_SIZE {dim}".format(i=i, dim=dim)
    h_file = h_file.replace("{input_dims_size}", input_dims_size_str)

    output_dims_size_str = ""
    for i, dim in enumerate(tflm_info.output_dims):
        if i > 0:
            output_dims_size_str += "\n"
        output_dims_size_str += "#define OUTPUT_DIM_{i}_SIZE {dim}".format(i=i, dim=dim)
    h_file = h_file.replace("{output_dims_size}", output_dims_size_str)

    if tflm_info.op_resolver_funcs is not None:
        h_file = h_file.replace("{n_operators}", str(len(tflm_info.op_resolver_funcs)))

    cpp_file = cpp_file.replace("{model_data_size}", str(len(tflite_model)))

    model_data_str = ""
    for i, byte in enumerate(tflite_model):
        if i % 16 == 0:
            model_data_str += "\n\t"
        model_data_str += "0x{:02x}, ".format(byte)
    model_data_str = model_data_str[:-2] + "\n"

    cpp_file = cpp_file.replace("{model_data}", model_data_str)

    if tflm_info.op_resolver_funcs is not None:
        op_resolver_content_str = ""
        op_resolver_content_str += "\n\tTfLiteStatus status;\n"
        for i, op_resolver_func in enumerate(tflm_info.op_resolver_funcs):
            op_resolver_content_str += "\n"
            op_resolver_content_str += "\tstatus = resolver_ptr->{func};\n".format(func=op_resolver_func)
            op_resolver_content_str += "\tif (status != kTfLiteOk) {\n"
            op_resolver_content_str += "\t\tprintf(\"Failed to add {func}\");\n".format(func=op_resolver_func)
            op_resolver_content_str += "\t\treturn status;\n"
            op_resolver_content_str += "\t}\n"
        op_resolver_content_str += "\n"
        op_resolver_content_str += "\treturn kTfLiteOk;\n"
        cpp_file = cpp_file.replace("{op_resolver_content}", op_resolver_content_str)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'model.h'), 'w') as f:
        f.write(h_file)
    with open(os.path.join(save_dir, 'model.cpp'), 'w') as f:
        f.write(cpp_file)

    # create data files
    data = np.load(eqcheck_data_path)
    data_x = data['data_x']
    data_y = data['data_y_pred']
    data.close()

    with open(os.path.join(templates_dir, 'data.h'), 'r') as f:
        h_file = f.read()

    with open(os.path.join(templates_dir, 'data.cpp'), 'r') as f:
        cpp_file = f.read()

    h_file = h_file.replace("{n_samples}", str(data_x.shape[0]))

    data_x_shape_str = ""
    for dim in data_x.shape:
        data_x_shape_str += "[{}]".format(dim)
    data_y_shape_str = ""
    for dim in data_y.shape:
        data_y_shape_str += "[{}]".format(dim)
    h_file = h_file.replace("{samples_x_shape}", data_x_shape_str)
    h_file = h_file.replace("{samples_y_shape}", data_y_shape_str)
    cpp_file = cpp_file.replace("{samples_x_shape}", data_x_shape_str)
    cpp_file = cpp_file.replace("{samples_y_shape}", data_y_shape_str)

    if in_out_dtype == "float":
        h_file = h_file.replace("{input_dtype}", "float")
        h_file = h_file.replace("{output_dtype}", "float")
        cpp_file = cpp_file.replace("{input_dtype}", "float")
        cpp_file = cpp_file.replace("{output_dtype}", "float")
    elif in_out_dtype == "int8":
        h_file = h_file.replace("{input_dtype}", "int8_t")
        h_file = h_file.replace("{output_dtype}", "int8_t")
        cpp_file = cpp_file.replace("{input_dtype}", "int8_t")
        cpp_file = cpp_file.replace("{output_dtype}", "int8_t")
        data_x = (data_x / in_scale) + in_zero_point
        data_y = (data_y / out_scale) + out_zero_point
        data_x = np.clip(data_x, -128, 127)
        data_y = np.clip(data_y, -128, 127)
        data_x = data_x.astype(np.int8)
        data_y = data_y.astype(np.int8)
    elif in_out_dtype == "uint8":
        h_file = h_file.replace("{input_dtype}", "uint8_t")
        h_file = h_file.replace("{output_dtype}", "uint8_t")
        cpp_file = cpp_file.replace("{input_dtype}", "uint8_t")
        cpp_file = cpp_file.replace("{output_dtype}", "uint8_t")
        data_x = (data_x / in_scale) + in_zero_point
        data_y = (data_y / out_scale) + out_zero_point
        data_x = np.clip(data_x, 0, 255)
        data_y = np.clip(data_y, 0, 255)
        data_x = data_x.astype(np.uint8)
        data_y = data_y.astype(np.uint8)
    elif in_out_dtype == "int16":
        h_file = h_file.replace("{input_dtype}", "int16_t")
        h_file = h_file.replace("{output_dtype}", "int16_t")
        cpp_file = cpp_file.replace("{input_dtype}", "int16_t")
        cpp_file = cpp_file.replace("{output_dtype}", "int16_t")
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

    cpp_file = cpp_file.replace("{samples_x}", data_x_str)
    cpp_file = cpp_file.replace("{samples_y}", data_y_str)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'data.h'), 'w') as f:
        f.write(h_file)
    with open(os.path.join(save_dir, 'data.cpp'), 'w') as f:
        f.write(cpp_file)


def _save_tflm_exception(message, save_dir, delete_assets=False):
    """
    Save the exception message and delete the assets if requested.

    Args:
        message (str): The exception message.
        save_dir (str): The directory to save the exception message.
        delete_assets (bool): Whether to delete the TFLM related assets.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save the exception message
    with open(os.path.join(save_dir, 'exception.txt'), 'w') as f:
        f.write(message)

    # delete TFLM related files if they exist
    if delete_assets:
        if os.path.exists(os.path.join(save_dir, 'model.h')):
            os.remove(os.path.join(save_dir, 'model.h'))
        if os.path.exists(os.path.join(save_dir, 'model.cpp')):
            os.remove(os.path.join(save_dir, 'model.cpp'))
        if os.path.exists(os.path.join(save_dir, 'data.h')):
            os.remove(os.path.join(save_dir, 'data.h'))
        if os.path.exists(os.path.join(save_dir, 'data.cpp')):
            os.remove(os.path.join(save_dir, 'data.cpp'))


def main(cfg_path=config_file_path, **kwargs):
    """
    Convert TensorFlow Lite models to TFLM models.

    Args:
        cfg_path (str): The path to the configuration file.
            The configuration file that this path points to should contain the following keys:
                - model_base_dir (str): A placeholder for the model base directory. This will be populated by the target directory.
                - tflite_conversion_type (str): A placeholder for the TFLite conversion type. This will be populated by the conversion type.
                - linkers_dir (str): Path to the directory where the tflite converted models list is loaded from and the TFLM converted models list will be saved.
                - tflite_model_path (str): The path to the TensorFlow Lite model.
                - eqcheck_data_path (str): The path to the equality check data.
                - tflm_info_path (str): The path to the TFLM info file.
                - tflm_templates_dir (str): Path to the directory containing the TFLM model and data source templates.
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
        print("\nConverting the model in: {} ({}/{})".format(base_model["model_base_dir"], i+1, len(targets)))

        for model_flavor in base_model["conversions"]:
            if model_flavor != "q_float_16":
                txt = "Model type: {} ...".format(model_flavor)
                txt += " " * (32 - len(txt))
                print(txt, end=" ", flush=True)
                cfg.model_base_dir = base_model["model_base_dir"]
                cfg.tflite_conversion_type = model_flavor

                try:
                    if not os.path.exists(cfg.eqcheck_data_path):
                        save_random_eqcheck_data(cfg.tflite_model_path, cfg.n_random_eqcheck_data, cfg.eqcheck_data_path)
                    create_tflm_source_files(cfg.tflite_model_path, cfg.eqcheck_data_path, cfg.tflm_info_path, cfg.tflm_templates_dir, cfg.tflm_save_dir)
                    if os.path.exists(os.path.join(cfg.tflm_save_dir, 'exception.txt')):
                        os.remove(os.path.join(cfg.tflm_save_dir, 'exception.txt'))
                    converted_models_list.append(cfg.tflm_save_dir)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "success"
                    })
                    print("Done")
                except Exception as e:
                    error_message = traceback.format_exc()
                    _save_tflm_exception(error_message, cfg.tflm_save_dir, delete_assets=True)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "failed",
                        "error": type(e).__name__,
                        "traceback": traceback.format_exc()
                    })
                    print("Failed")

    os.makedirs(cfg.linkers_dir, exist_ok=True)
    with open(os.path.join(cfg.linkers_dir, 'tflm_converted_models_list.yaml'), 'w') as f:
        yaml.dump(converted_models_list, f, indent=4, sort_keys=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow Lite models to TFLM models.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
