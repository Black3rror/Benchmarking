import argparse
import base64
import logging
import os
import time
import traceback
import zipfile

import numpy as np
import requests
import tensorflow as tf
import yaml
from omegaconf import OmegaConf

from edgemark.models.platforms.TFLite.TFLite_converter import save_random_eqcheck_data
from edgemark.models.utils.utils import get_abs_path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
config_file_path = "edgemark/models/platforms/EI/configs/EI_converter_config.yaml"
config_file_path = get_abs_path(config_file_path)


def _save_ei_exception(message, save_dir, delete_assets=False):
    """
    Save the exception message and delete the assets if requested.

    Args:
        message (str): The exception message.
        save_dir (str): The directory to save the exception message.
        delete_assets (bool): Whether to delete the EI related assets.
    """
    os.makedirs(save_dir, exist_ok=True)

    # save the exception message
    with open(os.path.join(save_dir, 'exception.txt'), 'w') as f:
        f.write(message)

    # delete ei related files if they exist
    if delete_assets:
        if os.path.exists(os.path.join(save_dir, 'model.zip')):
            os.remove(os.path.join(save_dir, 'model.zip'))
        if os.path.exists(os.path.join(save_dir, 'data.h')):
            os.remove(os.path.join(save_dir, 'data.h'))
        if os.path.exists(os.path.join(save_dir, 'data.cpp')):
            os.remove(os.path.join(save_dir, 'data.cpp'))
        model_dir = os.path.join(save_dir, 'model')
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                os.remove(os.path.join(model_dir, file))
            os.rmdir(model_dir)


def convert_model(tflite_model_path, model_type, engine, api_key, project_id, save_dir):
    """
    Convert a TensorFlow Lite model to an Edge Impulse model.

    Args:
        tflite_model_path (str): The path to the TensorFlow Lite model.
        model_type (str): The model type. Can be 'int8' or 'float32'.
        engine (str): The engine to use. Can be 'tflite' or 'tflite-eon'.
        api_key (str): The API key to use.
        project_id (str): The project ID to use.
        save_dir (str): The directory to save the Edge Impulse zipped C++ code.
    """
    api_endpoint = "https://studio.edgeimpulse.com/v1/api/{}".format(project_id)

    # get the output n_dims of the model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    assert len(output_details) == 1, "The model should have only one output tensor"
    output_shape = output_details[0]["shape"]
    assert len(output_shape) == 2, "The output tensor should have 2 dimensions"
    n_dims = output_shape[1]
    if n_dims == 1:
        model_output_info = {"modelType": "regression"}
    else:
        model_output_info = {
            "modelType": "classification",
            "labels": ["class {}".format(i+1) for i in range(n_dims)]
        }

    # step 1: upload the model to Edge Impulse
    with open(tflite_model_path, "rb") as f:
        model_data = f.read()
    model_data_base64 = base64.b64encode(model_data).decode("utf-8")

    url = api_endpoint + "/jobs/deploy-pretrained-model"
    payload = {
        "modelFileType": "tflite",
        "modelInfo": {
            "input": {"inputType": "other"},      # even if image, we don't want to have scaling or normalization, so we pretend it's 'other'
            "model": model_output_info
        },
        "modelFileBase64": model_data_base64,
        "deploymentType": "zip",
        "engine": engine,
        "deployModelType": model_type
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key
    }

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()
    response = response.json()
    if response["success"] is not True:
        raise Exception("Failed to upload the model to Edge Impulse. The response is: {}".format(response))
    job_id = response["id"]

    # step 2: wait for the job to finish
    tic = time.time()
    while time.time() - tic < 120:
        url = api_endpoint + "/jobs/{}/status".format(job_id)
        headers = {
            "accept": "application/json",
            "x-api-key": api_key
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response = response.json()
        if response["success"] is not True:
            raise Exception("Failed to get the job status. The response is: {}".format(response))
        if "finished" in response["job"]:
            break
        time.sleep(5)
    if response["job"]["finishedSuccessful"] is not True:
        raise Exception("The job failed. The response is: {}".format(response))

    # step 3: download the Edge Impulse zipped C++ code
    url = api_endpoint + "/deployment/download?type=zip&modelType={}&engine={}".format(model_type, engine)
    headers = {
        "accept": "application/zip",
        "x-api-key": api_key
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.zip"), "wb") as f:
        f.write(response.content)

    with zipfile.ZipFile(os.path.join(save_dir, "model.zip"), 'r') as zip_ref:
        zip_ref.extractall(os.path.join(save_dir, "model"))


def create_data_source_files(eqcheck_data_path, templates_dir, save_dir):
    """
    Create the data source files for the Edge Impulse model.

    Args:
        eqcheck_data_path (str): The path to the equality check data.
        templates_dir (str): The directory containing the templates.
        save_dir (str): The directory to save the data source files.
    """
    def _np_to_c(array):
        array = np.array(array)
        array = array.reshape(-1)
        c_array = "{" + ", ".join([str(x) for x in array]) + "}"
        return c_array

    data = np.load(eqcheck_data_path)
    data_x = data['data_x']
    data_y = data['data_y_pred']
    data.close()

    with open(os.path.join(templates_dir, 'data.h'), 'r') as f:
        h_file = f.read()

    with open(os.path.join(templates_dir, 'data.cpp'), 'r') as f:
        cpp_file = f.read()

    h_file = h_file.replace("{n_samples}", str(data_x.shape[0]))
    cpp_file = cpp_file.replace("{n_samples}", str(data_x.shape[0]))

    data_x_size = np.prod(data_x.shape[1:])
    data_y_size = np.prod(data_y.shape[1:])
    h_file = h_file.replace("{samples_x_size}", str(data_x_size))
    h_file = h_file.replace("{samples_y_size}", str(data_y_size))
    cpp_file = cpp_file.replace("{samples_x_size}", str(data_x_size))
    cpp_file = cpp_file.replace("{samples_y_size}", str(data_y_size))

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


def main(cfg_path=config_file_path, **kwargs):
    """
    Convert TensorFlow Lite models to Edge Impulse models.

    Args:
        cfg_path (str): The path to the configuration file.
            The configuration file that this path points to should contain the following keys:
                - model_base_dir (str): A placeholder for the model base directory. This will be populated by the target directory.
                - tflite_conversion_type (str): A placeholder for the TFLite conversion type. This will be populated by the conversion type.
                - linkers_dir (str): Path to the directory where the tflite converted models list is loaded from and the EI converted models list will be saved.
                - user_config (str): Path to the user configuration file. This file should contain the following keys:
                    - ei_api_key (str): The Edge Impulse API key.
                    - ei_project_id (str): The Edge Impulse project ID.
                - tflite_model_path (str): The path to the TensorFlow Lite model.
                - eqcheck_data_path (str): The path to the equality check data.
                - data_templates_dir (str): Path to the directory containing the data templates.
                - ei_save_dir (str): The directory to save the Edge Impulse files.
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
    if os.path.exists(cfg.user_config):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

    targets = OmegaConf.load(os.path.join(cfg.linkers_dir, "tflite_converted_models_list.yaml"))

    output = [{"dir": base_model["model_base_dir"], "flavors": []} for base_model in targets]

    converted_models_list = []
    for i, base_model in enumerate(targets):
        print("\nConverting the model in: {} ({}/{})".format(base_model["model_base_dir"], i+1, len(targets)))

        for model_flavor in base_model["conversions"]:
            if model_flavor in ["basic", "q_full_int_only"]:
                txt = "Model type: {} ...".format(model_flavor)
                txt += " " * (32 - len(txt))
                print(txt, end=" ", flush=True)
                cfg.model_base_dir = base_model["model_base_dir"]
                cfg.tflite_conversion_type = model_flavor
                model_type = "int8" if model_flavor == "q_full_int_only" else "float32"

                try:
                    convert_model(cfg.tflite_model_path, model_type, "tflite-eon", cfg.ei_api_key, cfg.ei_project_id, cfg.ei_save_dir)
                    if not os.path.exists(cfg.eqcheck_data_path):
                        save_random_eqcheck_data(cfg.tflite_model_path, cfg.n_random_eqcheck_data, cfg.eqcheck_data_path)
                    create_data_source_files(cfg.eqcheck_data_path, cfg.data_templates_dir, cfg.ei_save_dir)
                    if os.path.exists(os.path.join(cfg.ei_save_dir, 'exception.txt')):
                        os.remove(os.path.join(cfg.ei_save_dir, 'exception.txt'))
                    converted_models_list.append(cfg.ei_save_dir)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "success"
                    })
                    print("Done")
                except Exception as e:
                    error_message = traceback.format_exc()
                    _save_ei_exception(error_message, cfg.ei_save_dir, delete_assets=True)
                    output[i]["flavors"].append({
                        "flavor": model_flavor,
                        "result": "failed",
                        "error": type(e).__name__,
                        "traceback": traceback.format_exc()
                    })
                    print("Failed")

    os.makedirs(cfg.linkers_dir, exist_ok=True)
    with open(os.path.join(cfg.linkers_dir, 'ei_converted_models_list.yaml'), 'w') as f:
        yaml.dump(converted_models_list, f, indent=4, sort_keys=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow Lite models to Edge Impulse models.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
