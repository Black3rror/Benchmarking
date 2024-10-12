import argparse
import datetime
import importlib
import logging
import os
import traceback

import wandb
from omegaconf import OmegaConf

from benchmarking.models.utils.utils import get_abs_path, find_target_files


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["WANDB_SILENT"] = "true"
config_file_path = "benchmarking/models/platforms/TensorFlow/configs/model_generator_config.yaml"
config_file_path = get_abs_path(config_file_path)


def main(cfg_path=config_file_path, **kwargs):
    """
    Generate, train, evaluate, and save models based on the given configuration file.

    Args:
        cfg_path (str): The path to the configuration file containing the model generation parameters.
            The configuration file that this path points to should contain the following keys:
                - model_type (str): A placeholder for the model type. This will be populated by the target model configuration.
                - time_tag (str): A placeholder for the time tag. This will be populated by the current time.
                - target_models_dir (str): Path to the directory containing the target models configurations.
                - datasets_dir (str): Path to the directory containing the datasets.
                - linkers_dir (str): Path to the directory where the generated models list will be saved.
                - model_path (str): Path to the model file.
                - model_save_dir (str): Path to the directory where the generated model will be saved.
                - data_save_dir (str): Path to the directory where the representative and equality check data will be saved.
                - TFLM_info_save_path (str): Path to the file where the TFLM info will be saved.
                - wandb_online (bool): Flag to enable or disable the W&B online mode.
                - wandb_project_name (str): Name of the W&B project.
                - train_models (bool): Flag to enable or disable model training.
                - evaluate_models (bool): Flag to enable or disable model evaluation.
                - measure_execution_time (bool): Flag to enable or disable the measurement of execution time.
                - epochs (int): Number of epochs for training the model. If specified, it will override the number of epochs in the model configuration.
                - n_representative_data (int): Number of samples to be saved for TFLite conversion.
                - n_eqcheck_data (int): Number of samples to be saved for equivalence check of the model on PC and MCU.
        **kwargs (dict): Keyword arguments to be passed to the configuration file.

    Returns:
        list: A list of dictionaries containing the following keys for each target model:
            - name (str): Name of the target model configuration file.
            - result (str): Result of the model generation. It can be either "success" or "failed".
            - error (str): Error message in case of failure.
            - traceback (str): Traceback in case of failure.
    """
    cfg = OmegaConf.load(cfg_path)
    cfg.update(OmegaConf.create(kwargs))

    if not cfg.wandb_online:
        os.environ['WANDB_MODE'] = 'offline'
    models_list = []

    target_files = find_target_files(cfg.target_models_dir)

    output = [{"name": os.path.splitext(target_file)[0]} for target_file in target_files]

    for i, target_file in enumerate(target_files):
        try:
            model_cfg_path = os.path.join(cfg.target_models_dir, target_file)
            model_cfg = OmegaConf.load(model_cfg_path)
            cfg.model_type = model_cfg.model_type
            cfg.time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if "epochs" in cfg:
                model_cfg.epochs = cfg.epochs
            if "dataset" in model_cfg:
                model_cfg.dataset.path = os.path.join(cfg.datasets_dir, model_cfg.dataset.name, "data.py")

            wandb_name = datetime.datetime.strptime(cfg.time_tag, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
            wandb_dir = get_abs_path(os.path.join(cfg.model_save_dir, 'tf'))
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(project=cfg.wandb_project_name, group=model_cfg.model_type, tags=[model_cfg.model_type, model_cfg.dataset.name, os.path.splitext(target_file)[0]], name=wandb_name, dir=wandb_dir)

            spec = importlib.util.spec_from_file_location("imported_module", cfg.model_path)
            imported_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(imported_module)

            title = "Creating the {} model described in {} ({}/{})".format(model_cfg.model_type, target_file, i+1, len(target_files))
            print("\n")
            print("="*80)
            print("-"*((80-len(title)-2)//2), end=" ")
            print(title, end=" ")
            print("-"*((80-len(title)-2)//2))
            print("="*80)

            supervisor = imported_module.ModelSupervisor(OmegaConf.to_container(model_cfg, resolve=True))

            print("Saving representative data to the directory: {} ...".format(cfg.data_save_dir), end=" ", flush=True)
            supervisor.save_representative_data(cfg.n_representative_data, cfg.data_save_dir)
            print("Done\n")

            try:
                supervisor.compile_model(fine_tuning=False)
            except Exception as e:
                print("Error in compiling the model: {}".format(e))
                print("Continuing without compilation")

            # print("Model summary:")
            # supervisor.model.summary()
            # print("")
            total_params, trainable_params, non_trainable_params = supervisor.get_params_count()
            MACs = supervisor.get_FLOPs() // 2

            if cfg.train_models:
                print("Training the model ...")
                tensorboard_log_dir = os.path.join(cfg.model_save_dir, 'tf/logs')
                best_weights_dir = os.path.join(cfg.model_save_dir, 'tf/weights/weights_best')
                supervisor.train_model(fine_tuning=False, tensorboard_log_dir=tensorboard_log_dir, best_weights_dir=best_weights_dir, use_wandb=True)
                print("")

            evaluation_result = None
            if cfg.evaluate_models:
                try:
                    evaluation_result = supervisor.evaluate_model()
                    for metric, value in evaluation_result.items():
                        print(metric, ":", value)
                    print("")
                except Exception as e:
                    print("Error in evaluating the model: {}".format(e))
                    print("Continuing without evaluation")

            print("Saving model and weights to the directory: {} ...".format(cfg.model_save_dir), end=" ", flush=True)
            supervisor.save_model(os.path.join(cfg.model_save_dir, "tf/model"))
            supervisor.save_weights(os.path.join(cfg.model_save_dir, 'tf/weights/weights_last'))
            print("Done\n")
            supervisor.log_model_to_wandb(os.path.join(cfg.model_save_dir, "tf/model"), os.path.splitext(target_file)[0].replace("/", "_"))

            print("Saving equality check data to the directory: {} ...".format(cfg.data_save_dir), end=" ", flush=True)
            supervisor.save_eqcheck_data(cfg.n_eqcheck_data, cfg.data_save_dir)
            print("Done\n")

            if cfg.measure_execution_time:
                print("Measuring execution time ...")
                execution_time = supervisor.measure_execution_time()
                print("Average run time: {} ms\n".format(execution_time))

            model_info = {"Description": ""}
            model_info["setting_file"] = target_file
            model_info["model_type"] = model_cfg.model_type
            model_info["trained"] = cfg.train_models
            model_info.update(supervisor.get_model_info())
            model_info["total_params"] = total_params
            model_info["trainable_params"] = trainable_params
            model_info["non_trainable_params"] = non_trainable_params
            model_info["MACs"] = MACs
            if evaluation_result is not None:
                for metric, value in evaluation_result.items():
                    if not isinstance(metric, str):
                        metric = str(metric)
                    model_info[metric] = value
            if cfg.measure_execution_time:
                model_info["execution_time"] = execution_time
            model_info["wandb_name"] = wandb_name

            print("Saving the model info in the directory: {} ...".format(cfg.model_save_dir), end=" ", flush=True)
            supervisor.save_model_info(model_info, cfg.model_save_dir)
            print("Done\n")
            wandb.config.update(model_info)

            try:
                print("Saving the TFLM info in: {} ...".format(cfg.TFLM_info_save_path), end=" ", flush=True)
                supervisor.save_TFLM_info(cfg.TFLM_info_save_path)
                print("Done\n")
            except Exception as e:
                print("Error in saving the TFLM info: {}".format(e))
                print("TFLM info will not be saved. Please fix this issue if you want to use the TFLM converter later.")

            models_list.append(cfg.model_save_dir)

            wandb.finish()

            output[i]["result"] = "success"

        except Exception as e:
            output[i]["result"] = "failed"
            output[i]["error"] = type(e).__name__
            output[i]["traceback"] = traceback.format_exc()
            print("Error in generating the model:")
            print(traceback.format_exc())

    print("Saving the generated models list in the directory: {} ...".format(cfg.linkers_dir), end=" ", flush=True)
    supervisor.save_models_list(models_list, cfg.linkers_dir)
    print("Done\n")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, train, evaluate, and save models based on the given configuration file.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="The path to the configuration file containing the model generation parameters.")
    args = parser.parse_args()

    main(args.cfg_path)
