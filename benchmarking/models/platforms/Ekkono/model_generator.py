import argparse
import datetime
import os
import traceback

import wandb
import yaml
from omegaconf import OmegaConf

from benchmarking.models.platforms.Ekkono.model import ModelSupervisor
from benchmarking.models.utils.utils import get_abs_path, find_target_files


os.environ["WANDB_SILENT"] = "true"
config_file_path = "benchmarking/models/platforms/Ekkono/configs/model_generator_config.yaml"
config_file_path = get_abs_path(config_file_path)


def main(cfg_path=config_file_path, **kwargs):
    """
    Generate, train, evaluate, and save models based on the given configuration file.

    Args:
        cfg_path (str): The path to the configuration file containing the model generation parameters.
            The configuration file that this path points to should contain the following keys:
                - time_tag (str): A placeholder for the time tag. This will be populated by the current time.
                - linkers_dir (str): Path to the directory where the generated models list will be saved.
                - target_models_dir (str): Path to the directory containing the target models configurations.
                - datasets_dir (str): Path to the directory containing the datasets.
                - model_save_dir (str): Path to the directory where the generated model will be saved.
                - crystal_templates_dir (str): Path to the directory containing the crystal templates.
                - wandb_online (bool): Flag to enable or disable the W&B online mode.
                - wandb_project_name (str): Name of the W&B project.
                - train_models (bool): Flag to enable or disable model training.
                - evaluate_models (bool): Flag to enable or disable model evaluation.
                - measure_execution_time (bool): Flag to enable or disable the measurement of execution time.
                - n_eqcheck_data (int): Number of samples to be saved for equivalence check of the model on PC and MCU.
        **kwargs (dict): Keyword arguments to be passed to the configuration file.

    Returns:
        list: A list of dictionaries containing the following keys for each target model:
            - name (str): Name of the target model configuration file.
            - result (str): Result of the model generation. It can be either "success" or "failed".
            - error (str): Error message in case of failure.
            - traceback (str): Traceback in case of failure.
    """
    cfg = OmegaConf.load(cfg_path, **kwargs)
    cfg.update(OmegaConf.create(kwargs))

    if not cfg.wandb_online:
        os.environ['WANDB_MODE'] = 'offline'

    target_files = find_target_files(cfg.target_models_dir)

    output = [{"name": os.path.splitext(target_file)[0]} for target_file in target_files]

    converted_models_list = []
    for i, target_file in enumerate(target_files):
        try:
            model_cfg_path = os.path.join(cfg.target_models_dir, target_file)
            model_cfg = OmegaConf.load(model_cfg_path)
            cfg.time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if "epochs" in cfg:
                model_cfg.epochs = cfg.epochs
            if "dataset" in model_cfg:
                model_cfg.dataset.path = os.path.join(cfg.datasets_dir, model_cfg.dataset.name, "data.py")

            wandb_name = datetime.datetime.strptime(cfg.time_tag, "%Y-%m-%d_%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
            wandb_dir = get_abs_path(cfg.model_save_dir)
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(project=cfg.wandb_project_name, tags=[model_cfg.dataset.name, os.path.splitext(target_file)[0]], name=wandb_name, dir=wandb_dir)

            title = "Creating the model described in {} ({}/{})".format(target_file, i+1, len(target_files))
            print("\n")
            print("="*80)
            print("-"*((80-len(title)-2)//2), end=" ")
            print(title, end=" ")
            print("-"*((80-len(title)-2)//2))
            print("="*80)

            supervisor = ModelSupervisor(OmegaConf.to_container(model_cfg, resolve=True))

            if cfg.train_models:
                print("Training the model ...", end=" ", flush=True)
                supervisor.train_model()
                print("Done\n")

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
                    continue

            print("Saving model to the directory: {} ...".format(os.path.join(cfg.model_save_dir, "edge")), end=" ", flush=True)
            supervisor.save_model(os.path.join(cfg.model_save_dir, "edge"))
            print("Done\n")
            supervisor.log_model_to_wandb(os.path.join(cfg.model_save_dir, "edge"), os.path.splitext(target_file)[0].replace("/", "_"))

            print("Saving equality check data to the directory: {} ...".format(os.path.join(cfg.model_save_dir, "data")), end=" ", flush=True)
            supervisor.save_eqcheck_data(cfg.n_eqcheck_data, os.path.join(cfg.model_save_dir, "data"))
            print("Done\n")

            if cfg.measure_execution_time:
                print("Measuring execution time ...")
                execution_time = supervisor.measure_execution_time()
                print("Average run time: {} ms\n".format(execution_time))

            model_info = {"Description": ""}
            model_info["setting_file"] = target_file
            model_info["trained"] = cfg.train_models
            model_info.update(supervisor.get_model_info())
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

            print("Saving the crystal files in the directory: {} ...".format(os.path.join(cfg.model_save_dir, "crystal")), end=" ", flush=True)
            supervisor.fill_crystal_templates(os.path.join(cfg.model_save_dir, "crystal"), os.path.join(cfg.model_save_dir, "data"), cfg.crystal_templates_dir)
            print("Done\n")

            converted_models_list.append(os.path.join(cfg.model_save_dir, "crystal"))

            wandb.finish()

            output[i]["result"] = "success"

        except Exception as e:
            output[i]["result"] = "failed"
            output[i]["error"] = type(e).__name__
            output[i]["traceback"] = traceback.format_exc()
            print("Error in generating the model:")
            print(traceback.format_exc())

    os.makedirs(cfg.linkers_dir, exist_ok=True)
    with open(os.path.join(cfg.linkers_dir, 'ekkono_converted_models_list.yaml'), 'w') as f:
        yaml.dump(converted_models_list, f, indent=4, sort_keys=False)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate, train, evaluate, and save models based on the given configuration file.")
    parser.add_argument("--cfg_path", type=str, default=config_file_path, help="Path to the configuration file.")
    args = parser.parse_args()

    main(args.cfg_path)
