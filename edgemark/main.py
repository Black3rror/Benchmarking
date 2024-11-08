import datetime
import os
import subprocess
import sys
import traceback
import warnings

try:
    import colorama
    with warnings.catch_warnings():     # suppress the deprecation warning
        warnings.simplefilter("ignore", DeprecationWarning)
        import pkg_resources
    from omegaconf import OmegaConf
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'colorama==0.4.6', 'setuptools==69.0.3', 'hydra-core==1.3.2'])
    print("\n----- New packages have been installed. Please run the script again -----")
    sys.exit(0)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # suppress TensorFlow info and warnings
import_error = False
try:    # in case requirements are not installed
    import edgemark.models.automate.automate as automate
    import edgemark.models.platforms.eAI_Translator.eAI_Translator_converter as eai_translator_converter
    import edgemark.models.platforms.EI.EI_converter as ei_converter
    import edgemark.models.platforms.TensorFlow.model_generator as tf_model_generator
    import edgemark.models.platforms.TFLite.TFLite_converter as tflite_converter
    import edgemark.models.platforms.TFLM.TFLM_converter as tflm_converter
    import edgemark.models.platforms.Ekkono.model_generator as ekkono_model_generator   # keep it last, so it won't it won't be a backdoor for other errors
except ImportError as e:
    if "ekkono" not in str(e):  # we check for ekkono's installation later
        import_error = True
        import_error_traceback = traceback.format_exc()


COLOR_SECONDARY = colorama.Fore.LIGHTCYAN_EX
COLOR_TERTIARY = colorama.Fore.LIGHTMAGENTA_EX
COLOR_RESET = colorama.Style.RESET_ALL
colorama.init(autoreset=True)


class _Question:
    """
    A class to represent a question in the console.
    """
    def __init__(self, question, description=None, options=None, one_line_options=None, default=None):
        """
        Initialize the question with the given parameters.

        Args:
            question (str): The question to be asked.
            description (str): The description of the question. It will be displayed dimmed and won't appear in the summary.
            options (list): The options to be displayed. It should be a list of strings.
            one_line_options (bool): If the options are to be displayed in one line. If None, it will be determined automatically.
            default (str): The default option.
        """
        # sanity check
        assert isinstance(question, str), "question must be a string"
        assert description is None or isinstance(description, str), "description must be a string"
        assert options is None or isinstance(options, list), "options must be a list"
        if options is not None:
            for option in options:
                assert isinstance(option, str), "options must be a list of strings"
        assert one_line_options is None or isinstance(one_line_options, bool), "one_line_options must be a boolean"
        assert default is None or isinstance(default, str), "default must be a string"
        if default is not None and options is not None:
            assert default in options, "default must be in options"

        self.question = question
        self.description = description
        self.options = options
        self.one_line_options = one_line_options
        self.default = default
        self.response = None

        if one_line_options is None:
            if options is None:
                self.one_line_options = False

            else:
                len_options = 0
                for option in options:
                    len_options += len(option)

                self.one_line_options = True if len_options <= 8 else False


    def check_response(self, response):
        """
        Check if the response is valid. Override this method to customize the validation.

        Args:
            response (str): The response to be checked.

        Returns:
            tuple: A tuple containing a boolean indicating if the response is valid and the response itself.
                - valid (bool): True if the response is valid, False otherwise.
                - response (str): The string representation of the response (even if the user chose a number).
        """
        if self.options is None:
            return True, response

        else:
            if response.isdigit():
                response = int(response)
                if response >= 1 and response <= len(self.options):
                    return True, self.options[response - 1]
                else:
                    return False, None

            else:
                if response in self.options:
                    return True, response
                else:
                    return False, None


    def ask(self):
        """
        Ask the question and get the response from the user.

        Returns:
            str: The response given by the user.
        """
        # print the question
        print(self.question, end="")
        if self.description is not None:
            print(colorama.Style.DIM + "\n" + self.description, end="")
            if self.options is None or self.one_line_options:
                print("\nPlease answer", end="")

        # print the options -> [a/b/c/d]
        if self.options is not None:
            if self.one_line_options:
                options_txt = " ["
                options_txt += "/".join(self.options)
                options_txt += "]"
                print(COLOR_TERTIARY + options_txt, end="")

            else:
                print()
                for i, option in enumerate(self.options):
                    print(COLOR_TERTIARY + str(i + 1), end="")
                    print(". " + option)

                print("Choose from ", end="")
                options_txt = "["
                options_txt += "/".join([str(i + 1) for i in range(len(self.options))])
                options_txt += "]"
                print(COLOR_TERTIARY + options_txt, end="")

        # print the default option -> (a)
        if self.default is not None:
            if self.one_line_options or self.options is None:
                print(COLOR_SECONDARY + " (" + self.default + ")", end="")
            else:
                default_idx = self.options.index(self.default)
                print(COLOR_SECONDARY + " (" + str(default_idx + 1) + ")", end="")

        print(": ", end="")

        # get the user input
        while True:
            response = input()
            if response == "" and self.default is not None:
                response = self.default

            valid_response, response_str = self.check_response(response)
            if valid_response:
                break
            else:
                print("Invalid response. Please try again: ", end="")

        self.response = response_str
        return response_str


    def summarize(self):
        """
        Summarize the question and the response.

        Returns:
            str: The summarized text
        """
        txt = self.question + ": " + COLOR_SECONDARY + self.response + COLOR_RESET
        return txt


def _clear_console():
    """
    Clear the console screen.
    """
    os.system("cls" if os.name == "nt" else "clear")


def check_package_requirements(requirements_file):
    """
    Check if the required packages are installed.

    Args:
        requirements_file (str): The path to the requirements file.

    Returns:
        bool: True if all the required packages are installed, False otherwise.
    """
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()

    try:
        pkg_resources.require(requirements)
    except pkg_resources.UnknownExtra:  # Bug: we'll get this error even if everything is fine, so we'll ignore it
        pass
    except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
        return False

    return True


def install_requirements(requirements_file):
    """
    Install the required packages.

    Args:
        requirements_file (str): The path to the requirements file.
    """
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', requirements_file, '--no-warn-script-location'])


def _check_ekkono_installation():
    """
    Check if Ekkono is installed.

    Returns:
        bool: True if Ekkono is installed, False otherwise.
    """
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'ekkono.primer'], capture_output=True, text=True)
        result.check_returncode()
    except subprocess.CalledProcessError:
        return False

    return True


def main():
    """
    The main function to give a user-friendly interface to run the project.
    """
    # configs
    reqirements_file = "requirements.txt"
    ei_converter_config_path = "edgemark/models/platforms/EI/configs/EI_converter_config.yaml"
    stm32_automate_config_path = "edgemark/models/automate/hardware_types/NUCLEO-L4R5ZI/configs/hardware_config.yaml"
    renesas_automate_config_path = "edgemark/models/automate/hardware_types/RenesasRX65N/configs/hardware_config.yaml"
    target_dir = "target_models"

    time_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("benchmarking_results", time_tag)

    _clear_console()
    page = ""

    # check the all the required packages are installed
    installed_requirements = False
    while not check_package_requirements(reqirements_file):
        q = _Question("Packages are missing. What do you want to do",
                      description="Please make sure that you are in the correct environment.\nIt's best to create a new Conda environment with 'python 3.11.7'. The command would be 'conda create -n edgemark python=3.11.7'. Then activate the environment with 'conda activate edgemark' and run the project again.",
                      options=["Install packages for me", "I will install them myself"],
                      one_line_options=False,
                      default="Install packages for me")
        response = q.ask()
        _clear_console()
        page = q.summarize() + "\n"

        if response == "Install packages for me":
            install_requirements(reqirements_file)
            installed_requirements = True

        else:
            page += "You can find the requirements in the file '{}'. Please install them manually.".format(reqirements_file)
            print(page)
            return

    if installed_requirements:
        page += "Requirements are installed successfully.\n"
        page += "Please run the script again."
        _clear_console()
        print(page)
        return

    if import_error:
        page += "There was an " + COLOR_TERTIARY +  "error" + COLOR_RESET + " while importing the required modules.\n"
        page += "Please check the error below and fix the issue.\n"
        page += import_error_traceback
        _clear_console()
        print(page)
        return

    page = ""
    _clear_console()

    page += "Hello " + COLOR_SECONDARY + ":)" + COLOR_RESET + "\n"
    print(page)

    # get the modules to run

    available_modules = [
        "Generate TF models",
        "Generate Ekkono models",
        "Convert to TFLite",
        "Convert to TFLM",
        "Convert to Edge Impulse",
        "Convert to eAI Translator",
        "Test on NUCLEO-L4R5ZI (TFLM)",
        "Test on NUCLEO-L4R5ZI (Edge Impulse)",
        "Test on NUCLEO-L4R5ZI (Ekkono)",
        "Test on RenesasRX65N (TFLM)",
        "Test on RenesasRX65N (Edge Impulse)",
        "Test on RenesasRX65N (Ekkono)",
        "Test on RenesasRX65N (eAI Translator)"
    ]

    q = _Question("What do you want to do",
                  description="Here you can see the options for running the full pipeline (generating models, converting, testing). Choose 'Others' if you want to run specific modules.",
                  options=[
                      "TFLM + NUCLEO-L4R5ZI",
                      "TFLM + RenesasRX65N",
                      "Edge Impulse + NUCLEO-L4R5ZI",
                      "Edge Impulse + RenesasRX65N",
                      "Ekkono + NUCLEO-L4R5ZI",
                      "Ekkono + RenesasRX65N",
                      "eAI Translator + RenesasRX65N",
                      "Others"
                  ])
    response = q.ask()
    _clear_console()
    page += q.summarize() + "\n"
    print(page)

    if response == "TFLM + NUCLEO-L4R5ZI":
        modules = [
            "Generate TF models",
            "Convert to TFLite",
            "Convert to TFLM",
            "Test on NUCLEO-L4R5ZI (TFLM)"
        ]

    elif response == "TFLM + RenesasRX65N":
        modules = [
            "Generate TF models",
            "Convert to TFLite",
            "Convert to TFLM",
            "Test on RenesasRX65N (TFLM)"
        ]

    elif response == "Edge Impulse + NUCLEO-L4R5ZI":
        modules = [
            "Generate TF models",
            "Convert to TFLite",
            "Convert to Edge Impulse",
            "Test on NUCLEO-L4R5ZI (Edge Impulse)"
        ]

    elif response == "Edge Impulse + RenesasRX65N":
        modules = [
            "Generate TF models",
            "Convert to TFLite",
            "Convert to Edge Impulse",
            "Test on RenesasRX65N (Edge Impulse)"
        ]

    elif response == "Ekkono + NUCLEO-L4R5ZI":
        modules = [
            "Generate Ekkono models",
            "Test on NUCLEO-L4R5ZI (Ekkono)"
        ]

    elif response == "Ekkono + RenesasRX65N":
        modules = [
            "Generate Ekkono models",
            "Test on RenesasRX65N (Ekkono)"
        ]

    elif response == "eAI Translator + RenesasRX65N":
        modules = [
            "Generate TF models",
            "Convert to TFLite",
            "Convert to eAI Translator",
            "Test on RenesasRX65N (eAI Translator)"
        ]

    elif response == "Others":
        q = _Question("Which module(s) do you want to run",
                      description="You can choose multiple options by separating them with a plus sign (+). Example: 1+2\nThe rationality of the sequence of the modules is user's responsibility.",
                      options=available_modules)
        def _check_response(response):
            valid = True
            for m in response.split("+"):
                if not m.isdigit() or int(m) < 1 or int(m) > len(available_modules):
                    valid = False
                    break
            if valid:
                response = [available_modules[int(m) - 1] for m in response.split("+")]
                response = " + ".join(response)
            return valid, response
        q.check_response = _check_response
        response = q.ask()
        _clear_console()
        page += q.summarize() + "\n"
        print(page)

        modules = [module.strip() for module in response.split("+")]

    # check requirements of the modules
    modules_requiremets = []

    if "Generate Ekkono models" in modules:
        if "ekkono_sdk" not in modules_requiremets:
            modules_requiremets.append("ekkono_sdk")

    if "Convert to Edge Impulse" in modules:
        if "edge_impulse_secrets" not in modules_requiremets:
            modules_requiremets.append("edge_impulse_secrets")

    if ("Test on NUCLEO-L4R5ZI (TFLM)" in modules or
        "Test on NUCLEO-L4R5ZI (Edge Impulse)" in modules or
        "Test on NUCLEO-L4R5ZI (Ekkono)" in modules):
        if "stm32cubeide" not in modules_requiremets:
            modules_requiremets.append("stm32cubeide")
        if "stm32_programmer_cli" not in modules_requiremets:
            modules_requiremets.append("stm32_programmer_cli")

    if ("Test on RenesasRX65N (TFLM)" in modules or
        "Test on RenesasRX65N (Edge Impulse)" in modules or
        "Test on RenesasRX65N (Ekkono)" in modules or
        "Test on RenesasRX65N (eAI Translator)" in modules):
        if "e2studio" not in modules_requiremets:
            modules_requiremets.append("e2studio")
        if "rfp" not in modules_requiremets:
            modules_requiremets.append("rfp")

    # check if the requirements are satisfied
    if "ekkono_sdk" in modules_requiremets:
        while not _check_ekkono_installation():
            _clear_console()
            print(page)
            q = _Question("Ekkono is not installed. Please provide the path to the wheel file",
                          description="Ekkono is not free. If you want to use it, you need to buy this product from https://www.ekkono.ai, download the its files and provide the wheel path here.\nFor example, put the files in 'edgemark/models/platforms/Ekkono' and provide the path 'edgemark/models/platforms/Ekkono/ekkono-sdk/primer/python/{distribution}/{python-version}/ekkono.primer-{name-suffix}.whl'")
            q.check_response = lambda response: (True, response) if (os.path.exists(response) and response.endswith(".whl")) else (False, None)
            response = q.ask()
            subprocess.run([sys.executable, '-m', 'pip', 'install', response])
            print("Ekkono has been installed. Please run the script again.")
            return
        _clear_console()
        print(page)

    if "edge_impulse_secrets" in modules_requiremets:
        ei_converter_config = OmegaConf.load(ei_converter_config_path)
        if os.path.exists(ei_converter_config.user_config):
            ei_converter_user_config = OmegaConf.load(ei_converter_config.user_config)
        else:
            ei_converter_user_config = OmegaConf.create()

        valid_ei_api_key = False
        if "ei_api_key" in ei_converter_user_config:
            if ei_converter_user_config.ei_api_key.startswith("ei_"):
                valid_ei_api_key = True

        if not valid_ei_api_key:
            q = _Question("Edge Impulse information is missing. Please provide the API key",
                          description="If you don't have an account, please create one at https://www.edgeimpulse.com. The API key can be found in the 'Keys' section of the project. You can also access this page by its address: https://studio.edgeimpulse.com/studio/{project_id}/keys",)
            q.check_response = lambda response: (True, response) if response.startswith("ei_") else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            ei_converter_user_config.ei_api_key = response
            OmegaConf.save(ei_converter_user_config, ei_converter_config.user_config)

        valid_ei_project_id = False
        if "ei_project_id" in ei_converter_user_config:
            valid_ei_project_id = True

        if not valid_ei_project_id:
            q = _Question("Edge Impulse information is missing. Please provide the project ID",
                          description="The project ID can be found in the main page of the project ('project info' section). You can also find it by looking at the URL of the page: https://studio.edgeimpulse.com/studio/{project_id}",)
            q.check_response = lambda response: (True, response) if response.isdigit() else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            ei_converter_user_config.ei_project_id = response
            OmegaConf.save(ei_converter_user_config, ei_converter_config.user_config)

    if "stm32cubeide" in modules_requiremets:
        stm32_automate_config = OmegaConf.load(stm32_automate_config_path)
        if os.path.exists(stm32_automate_config.user_config):
            stm32_automate_user_config = OmegaConf.load(stm32_automate_config.user_config)
        else:
            stm32_automate_user_config = OmegaConf.create()

        valid_stm32cubeide_path = False
        if "stm32cubeide_path" in stm32_automate_user_config:
            if os.path.exists(stm32_automate_user_config.stm32cubeide_path) or os.path.exists(stm32_automate_user_config.stm32cubeide_path + ".exe"):
                valid_stm32cubeide_path = True

        if not valid_stm32cubeide_path:
            q = _Question("Could not find a valid STM32CubeIDE path. Please provide the path to the STM32CubeIDE executable",
                          description="If you don't have STM32CubeIDE, you can download it from https://www.st.com/en/development-tools/stm32cubeide.html\nIf you have already installed it, the executable path will be {installation_dir}/STM32CubeIDE_{version}/STM32CubeIDE/stm32cubeide.exe. For example, C:/ST/STM32CubeIDE_1.14.1/STM32CubeIDE/stm32cubeide.exe\nThe project was tested against STM32CubeIDE version 1.14.1")
            q.check_response = lambda response: (True, response) if os.path.exists(response) or os.path.exists(response + ".exe") else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            stm32_automate_user_config.stm32cubeide_path = response
            OmegaConf.save(stm32_automate_user_config, stm32_automate_config.user_config)

        valid_workspace_dir = False
        if "workspace_dir" in stm32_automate_user_config:
            if os.path.exists(stm32_automate_user_config.workspace_dir):
                valid_workspace_dir = True

        if not valid_workspace_dir:
            q = _Question("STM32CubeIDE workspace directory does not exist. Please provide the workspace directory",
                          description="When STM32CubeIDE is open, you can find the workspace directory in File > Switch Workspace > Other...")
            q.check_response = lambda response: (True, response) if os.path.exists(response) else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            stm32_automate_user_config.workspace_dir = response
            OmegaConf.save(stm32_automate_user_config, stm32_automate_config.user_config)

        if "Test on NUCLEO-L4R5ZI (TFLM)" in modules:
            stm32_automate_config.project_name = "NUCLEO-L4R5ZI_TFLM"
            q = _Question("Can you confirm that the NUCLEO-L4R5ZI_TFLM project exists in your STM32CubeIDE's projects and also in this location: {}".format(stm32_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in STM32CubeIDE.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

        if "Test on NUCLEO-L4R5ZI (Edge Impulse)" in modules:
            stm32_automate_config.project_name = "NUCLEO-L4R5ZI_EI"
            q = _Question("Can you confirm that the NUCLEO-L4R5ZI_EI project exists in your STM32CubeIDE's projects and also in this location: {}".format(stm32_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in STM32CubeIDE.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

        if "Test on NUCLEO-L4R5ZI (Ekkono)" in modules:
            stm32_automate_config.project_name = "NUCLEO-L4R5ZI_Ekkono"
            q = _Question("Can you confirm that the NUCLEO-L4R5ZI_Ekkono project exists in your STM32CubeIDE's projects and also in this location: {}".format(stm32_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in STM32CubeIDE.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

    if "stm32_programmer_cli" in modules_requiremets:
        stm32_automate_config = OmegaConf.load(stm32_automate_config_path)
        if os.path.exists(stm32_automate_config.user_config):
            stm32_automate_user_config = OmegaConf.load(stm32_automate_config.user_config)
        else:
            stm32_automate_user_config = OmegaConf.create()

        valid_stm32_programmer_path = False
        if "stm32_programmer_path" in stm32_automate_user_config:
            if os.path.exists(stm32_automate_user_config.stm32_programmer_path) or os.path.exists(stm32_automate_user_config.stm32_programmer_path + ".exe"):
                valid_stm32_programmer_path = True

        if not valid_stm32_programmer_path:
            q = _Question("Could not find a valid STM32 Programmer CLI path. Please provide the path to the STM32 Programmer CLI executable",
                          description="STM32 Programmer CLI is a part of STM32CubeCLT. So, if you don't have STM32 Programmer CLI, you can download the STM32CubeCLT from https://www.st.com/en/development-tools/stm32cubeclt.html\nOnce you have installed it, the executable should be in {installation_dir}/STM32CubeCLT_{version}/STM32CubeProgrammer/bin/STM32_programmer_CLI.exe. For example, C:/ST/STM32CubeCLT_1.15.1/STM32CubeProgrammer/bin/STM32_programmer_CLI.exe\nThe project was tested against STM32CubeProgrammer version 2.16.0",)
            q.check_response = lambda response: (True, response) if os.path.exists(response) or os.path.exists(response + ".exe") else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            stm32_automate_user_config.stm32_programmer_path = response
            OmegaConf.save(stm32_automate_user_config, stm32_automate_config.user_config)

    if "e2studio" in modules_requiremets:
        renesas_automate_config = OmegaConf.load(renesas_automate_config_path)
        if os.path.exists(renesas_automate_config.user_config):
            renesas_automate_user_config = OmegaConf.load(renesas_automate_config.user_config)
        else:
            renesas_automate_user_config = OmegaConf.create()

        valid_e2studio_path = False
        if "e2studio_path" in renesas_automate_user_config:
            if os.path.exists(renesas_automate_user_config.e2studio_path) or os.path.exists(renesas_automate_user_config.e2studio_path + ".exe"):
                valid_e2studio_path = True

        if not valid_e2studio_path:
            q = _Question("Could not find a valid e2 studio path. Please provide the path to the e2 studio executable",
                          description="If you don't have e2 studio, you can download it from https://www.renesas.com/us/en/software-tool/e-studio\nOnce you have installed it, the executable should be {installation_dir}/Renesas/e2_studio/eclipse/e2studioc.exe. For example, C:/Renesas/e2_studio/eclipse/e2studioc.exe\nThe project was tested against e2 studio version 24.1.1",)
            q.check_response = lambda response: (True, response) if os.path.exists(response) or os.path.exists(response + ".exe") else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            renesas_automate_user_config.e2studio_path = response
            OmegaConf.save(renesas_automate_user_config, renesas_automate_config.user_config)

        valid_workspace_dir = False
        if "workspace_dir" in renesas_automate_user_config:
            if os.path.exists(renesas_automate_user_config.workspace_dir):
                valid_workspace_dir = True

        if not valid_workspace_dir:
            q = _Question("e2 studio workspace directory does not exist. Please provide the workspace directory",
                          description="When e2 studio is open, you can find the workspace directory in File > Switch Workspace > Other...")
            q.check_response = lambda response: (True, response) if os.path.exists(response) else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            renesas_automate_user_config.workspace_dir = response
            OmegaConf.save(renesas_automate_user_config, renesas_automate_config.user_config)

        if "Test on RenesasRX65N (TFLM)" in modules:
            renesas_automate_config.project_name = "RenesasRX_TFLM"
            q = _Question("Can you confirm that the RenesasRX_TFLM project exists in your e2 studio's projects and also in this location: {}".format(renesas_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in e2 studio.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

        if "Test on RenesasRX65N (Edge Impulse)" in modules:
            renesas_automate_config.project_name = "RenesasRX_EI"
            q = _Question("Can you confirm that the RenesasRX_EI project exists in your e2 studio's projects and also in this location: {}".format(renesas_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in e2 studio.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

        if "Test on RenesasRX65N (Ekkono)" in modules:
            renesas_automate_config.project_name = "RenesasRX_Ekkono"
            q = _Question("Can you confirm that the RenesasRX_Ekkono project exists in your e2 studio's projects and also in this location: {}".format(renesas_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in e2 studio.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

        if "Test on RenesasRX65N (eAI Translator)" in modules:
            renesas_automate_config.project_name = "RenesasRX_eAI_Translator"
            q = _Question("Can you confirm that the RenesasRX_eAI_Translator project exists in your e2 studio's projects and also in this location: {}".format(renesas_automate_config.project_dir),
                          description="If you don't have the project in the specified location, probably you can find the zipped project in that directory. You can extract it and import it in e2 studio.",
                          options=["y", "n"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            if response == "n":
                print("Please create the project and run the script again.")
                return
            _clear_console()
            print(page)

    if "rfp" in modules_requiremets:
        renesas_automate_config = OmegaConf.load(renesas_automate_config_path)
        if os.path.exists(renesas_automate_config.user_config):
            renesas_automate_user_config = OmegaConf.load(renesas_automate_config.user_config)
        else:
            renesas_automate_user_config = OmegaConf.create()

        valid_rfp_path = False
        if "rfp_path" in renesas_automate_user_config:
            if os.path.exists(renesas_automate_user_config.rfp_path) or os.path.exists(renesas_automate_user_config.rfp_path + ".exe"):
                valid_rfp_path = True

        if not valid_rfp_path:
            q = _Question("Could not find a valid Renesas Flash Programmer path. Please provide the path to the Renesas Flash Programmer executable",
                          description="If you don't have Renesas Flash Programmer, you can download it from https://www.renesas.com/us/en/software-tool/renesas-flash-programmer-programming-gui#downloads\nOnce you have installed it, the executable should be {installation_dir}/Renesas Electronics/Programming Tools/Renesas Flash Programmer V{version}/RFPV{version}.exe. For example, C:/Program Files (x86)/Renesas Electronics/Programming Tools/Renesas Flash Programmer V3.15/RFPV3.exe\nThe project was tested against Renesas Flash Programmer version 3.15.00",)
            q.check_response = lambda response: (True, response) if os.path.exists(response) or os.path.exists(response + ".exe") else (False, None)
            response = q.ask()
            _clear_console()
            print(page)

            renesas_automate_user_config.rfp_path = response
            OmegaConf.save(renesas_automate_user_config, renesas_automate_config.user_config)

        valid_rfp_project_path = False
        if "rfp_project_path" in renesas_automate_user_config:
            if os.path.exists(renesas_automate_user_config.rfp_project_path):
                valid_rfp_project_path = True

        if not valid_rfp_project_path:
            q = _Question("Could not find a valid Renesas Flash Programmer project path. Please provide the path to the Renesas Flash Programmer project file",
                          description="The project file is a '.rpj' file that you can create in Renesas Flash Programmer.\nTo create a project file, open Renesas Flash Programmer and do the following steps:\n- File > New Project...\n- Microcontroller: RX65x\n- Tool: E2 emulator Lite\n- Interface: FINE\n- Tool Details... > Reset Settings > Reset signal at Disconnect: Reset Pin as Hi-Z\nOnce you have created the project, enter the path to the project file here",
                          check_response=lambda response: (True, response) if response.endswith(".rpj") and os.path.exists(response) else (False, None))
            response = q.ask()
            _clear_console()
            print(page)

            renesas_automate_user_config.rfp_project_path = response
            OmegaConf.save(renesas_automate_user_config, renesas_automate_config.user_config)

    if ("Generate TF models" in modules or
        "Generate Ekkono models" in modules):
        while True:
            q = _Question("Please put all your model files in the {} directory. Can you confirm that this is done".format(target_dir),
                          description="You can follow the instructions in the {}. In short, the files that do not have dot (.) in their path will be generated".format(target_dir + "/README.md"),
                          options=["y"],
                          one_line_options=True,
                          default="y")
            response = q.ask()
            _clear_console()
            print(page)
            if response == "y":
                break

    if ("Test on NUCLEO-L4R5ZI (TFLM)" in modules or
        "Test on NUCLEO-L4R5ZI (Edge Impulse)" in modules or
        "Test on NUCLEO-L4R5ZI (Ekkono)" in modules):
        while True:
            q = _Question("Please\n- Connect the NUCLEO-L4R5ZI board to the computer\n- Close STM32CubeIDE\n- Close any serial monitor applications (e.g. PuTTY)\nCan you confirm that these items are addressed",
                        options=["y"],
                        one_line_options=True,
                        default="y")
            response = q.ask()
            _clear_console()
            print(page)
            if response == "y":
                break

    if ("Test on RenesasRX65N (TFLM)" in modules or
        "Test on RenesasRX65N (Edge Impulse)" in modules or
        "Test on RenesasRX65N (Ekkono)" in modules or
        "Test on RenesasRX65N (eAI Translator)" in modules):
        while True:
            q = _Question("Please\n- Connect the Renesas RX65N board to the computer\n- Connect a USB to TTL cable between your computer and the board\n- Close e2 studio\n- Close Renesas Flash Programmer\n- Close any serial monitor applications (e.g. PuTTY)\nCan you confirm that these items are addressed",
                        options=["y"],
                        one_line_options=True,
                        default="y")
            response = q.ask()
            _clear_console()
            print(page)
            if response == "y":
                break

    # run the modules
    page += "\n" + COLOR_TERTIARY + "Running the modules" + COLOR_RESET + "\n"
    _clear_console()
    print(page)

    flawless = True
    for i, module in enumerate(modules):
        page += colorama.Style.DIM + "[{}/{}]".format(i + 1, len(modules)) + COLOR_RESET + " "

        if module == "Generate TF models":
            page += "Generating TensorFlow models"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = tf_model_generator.main()

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["name"])
                    report += "Traceback:\n{}".format(target["traceback"])

                    report_name = target["name"].replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Generate TF models", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "name": target["name"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["name"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Generate Ekkono models":
            page += "Generating Ekkono models"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = ekkono_model_generator.main()

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["name"])
                    report += "Traceback:\n{}".format(target["traceback"])

                    report_name = target["name"].replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Generate Ekkono models", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "name": target["name"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["name"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Convert to TFLite":
            page += "Converting TensorFlow models to TFLite"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = tflite_converter.main()

            n_targets = 0
            n_success = 0
            failures = []
            for target in output:
                for flavor in target["flavors"]:
                    n_targets += 1
                    if flavor["result"] == "success":
                        n_success += 1
                    else:
                        report = "Model: {}\nFlavor: {}\n\n".format(target["dir"], flavor["flavor"])
                        if "traceback" in flavor:
                            report += "Traceback:\n{}".format(flavor["traceback"])
                        else:
                            report += "Exception file path: {}\n".format(flavor["exception_file"])

                        report_name = os.path.basename(target["dir"])
                        report_name += "_" + flavor["flavor"]
                        report_name = report_name.replace("/", " - ").replace("\\", " - ")
                        report_path = os.path.join(save_dir, "errors/Convert to TFLite", "{}.txt".format(report_name))
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w") as file:
                            file.write(report)

                        failures.append({
                            "dir": target["dir"],
                            "flavor": flavor["flavor"],
                            "error": flavor["error"],
                            "report_path": report_path.replace("\\", "/")
                        })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {} >> {}\n".format(failure["dir"], failure["flavor"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Convert to TFLM":
            page += "Converting TFLite models to TFLM"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = tflm_converter.main()

            n_targets = 0
            n_success = 0
            failures = []
            for target in output:
                for flavor in target["flavors"]:
                    n_targets += 1
                    if flavor["result"] == "success":
                        n_success += 1
                    else:
                        report = "Model: {}\nFlavor: {}\n\n".format(target["dir"], flavor["flavor"])
                        report += "Traceback:\n{}".format(flavor["traceback"])

                        report_name = os.path.basename(target["dir"])
                        report_name += "_" + flavor["flavor"]
                        report_name = report_name.replace("/", " - ").replace("\\", " - ")
                        report_path = os.path.join(save_dir, "errors/Convert to TFLM", "{}.txt".format(report_name))
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w") as file:
                            file.write(report)

                        failures.append({
                            "dir": target["dir"],
                            "flavor": flavor["flavor"],
                            "error": flavor["error"],
                            "report_path": report_path.replace("\\", "/")
                        })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {} >> {}\n".format(failure["dir"], failure["flavor"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Convert to Edge Impulse":
            page += "Converting TFLite models to Edge Impulse"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = ei_converter.main()

            n_targets = 0
            n_success = 0
            failures = []
            for target in output:
                for flavor in target["flavors"]:
                    n_targets += 1
                    if flavor["result"] == "success":
                        n_success += 1
                    else:
                        report = "Model: {}\nFlavor: {}\n\n".format(target["dir"], flavor["flavor"])
                        report += "Traceback:\n{}".format(flavor["traceback"])

                        report_name = os.path.basename(target["dir"])
                        report_name += "_" + flavor["flavor"]
                        report_name = report_name.replace("/", " - ").replace("\\", " - ")
                        report_path = os.path.join(save_dir, "errors/Convert to Edge Impulse", "{}.txt".format(report_name))
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w") as file:
                            file.write(report)

                        failures.append({
                            "dir": target["dir"],
                            "flavor": flavor["flavor"],
                            "error": flavor["error"],
                            "report_path": report_path.replace("\\", "/")
                        })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {} >> {}\n".format(failure["dir"], failure["flavor"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Convert to eAI Translator":
            page += "Converting TFLite models to eAI Translator"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = eai_translator_converter.main()

            n_targets = 0
            n_success = 0
            failures = []
            for target in output:
                for flavor in target["flavors"]:
                    n_targets += 1
                    if flavor["result"] == "success":
                        n_success += 1
                    else:
                        report = "Model: {}\nFlavor: {}\n\n".format(target["dir"], flavor["flavor"])
                        report += "Traceback:\n{}".format(flavor["traceback"])

                        report_name = os.path.basename(target["dir"])
                        report_name += "_" + flavor["flavor"]
                        report_name = report_name.replace("/", " - ").replace("\\", " - ")
                        report_path = os.path.join(save_dir, "errors/Convert to eAI Translator", "{}.txt".format(report_name))
                        os.makedirs(os.path.dirname(report_path), exist_ok=True)
                        with open(report_path, "w") as file:
                            file.write(report)

                        failures.append({
                            "dir": target["dir"],
                            "flavor": flavor["flavor"],
                            "error": flavor["error"],
                            "report_path": report_path.replace("\\", "/")
                        })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {} >> {}\n".format(failure["dir"], failure["flavor"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

            if n_success == 0 and i < len(modules) - 1:
                q = _Question("No model has survived. Do you want to continue",
                              description="If the following modules are dependent on the output of this module, this will probably lead to errors.",
                              options=["y", "n"],
                              one_line_options=True,
                              default="n")
                response = q.ask()
                if response == "n":
                    return

            _clear_console()
            print(page)

        elif module == "Test on NUCLEO-L4R5ZI (TFLM)":
            page += "Testing TFLM models on NUCLEO-L4R5ZI"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="TFLM", hardware_platform="NUCLEO-L4R5ZI", save_path=os.path.join(save_dir, "TFLM + NUCLEO-L4R5ZI.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on NUCLEO-L4R5ZI (TFLM)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on NUCLEO-L4R5ZI (Edge Impulse)":
            page += "Testing Edge Impulse models on NUCLEO-L4R5ZI"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="EI", hardware_platform="NUCLEO-L4R5ZI", save_path=os.path.join(save_dir, "Edge Impulse + NUCLEO-L4R5ZI.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on NUCLEO-L4R5ZI (Edge Impulse)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on NUCLEO-L4R5ZI (Ekkono)":
            page += "Testing Ekkono models on NUCLEO-L4R5ZI"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="Ekkono", hardware_platform="NUCLEO-L4R5ZI", save_path=os.path.join(save_dir, "Ekkono + NUCLEO-L4R5ZI.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on NUCLEO-L4R5ZI (Ekkono)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on RenesasRX65N (TFLM)":
            page += "Testing TFLM models on RenesasRX65N"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="TFLM", hardware_platform="RenesasRX65N", save_path=os.path.join(save_dir, "TFLM + RenesasRX65N.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on RenesasRX65N (TFLM)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on RenesasRX65N (Edge Impulse)":
            page += "Testing Edge Impulse models on RenesasRX65N"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="EI", hardware_platform="RenesasRX65N", save_path=os.path.join(save_dir, "Edge Impulse + RenesasRX65N.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on RenesasRX65N (Edge Impulse)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on RenesasRX65N (Ekkono)":
            page += "Testing Ekkono models on RenesasRX65N"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="Ekkono", hardware_platform="RenesasRX65N", save_path=os.path.join(save_dir, "Ekkono + RenesasRX65N.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on RenesasRX65N (Ekkono)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

        elif module == "Test on RenesasRX65N (eAI Translator)":
            page += "Testing eAI Translator models on RenesasRX65N"
            _clear_console()
            print(page)

            print(colorama.Style.DIM + "\n\n\n" + "="*80 + "\n" + "Module output:")
            output = automate.main(software_platform="eAI_Translator", hardware_platform="RenesasRX65N", save_path=os.path.join(save_dir, "eAI_Translator + RenesasRX65N.xlsx"))

            n_targets = len(output)
            n_success = 0
            failures = []
            for target in output:
                if target["result"] == "success":
                    n_success += 1
                else:
                    report = "Model: {}\n\n".format(target["dir"])
                    if "traceback" in target:
                        report += "Traceback:\n{}".format(target["traceback"])
                    else:
                        report += "Error file path: {}\n".format(target["error_file"])

                    report_name = target["dir"]
                    report_name = report_name.replace("/", " - ").replace("\\", " - ")
                    report_path = os.path.join(save_dir, "errors/Test on RenesasRX65N (eAI Translator)", "{}.txt".format(report_name))
                    os.makedirs(os.path.dirname(report_path), exist_ok=True)
                    with open(report_path, "w") as file:
                        file.write(report)

                    failures.append({
                        "dir": target["dir"],
                        "error": target["error"],
                        "report_path": report_path.replace("\\", "/")
                    })

            if n_success == n_targets:
                page += COLOR_SECONDARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
            else:
                flawless = False
                page += COLOR_TERTIARY + " ({}/{})\n".format(n_success, n_targets) + COLOR_RESET
                page += "Failed models:\n"
                for failure in failures:
                    page += "- Model: {}\n".format(failure["dir"])
                    page += "  Error: {}\n".format(failure["error"])
                    page += "  Details: {}\n".format(failure["report_path"])

            _clear_console()
            print(page)

    if flawless:
        page += "\nAll the modules were " + COLOR_SECONDARY + "successfully" + COLOR_RESET + " executed\n"
    else:
        page += "\nAll the modules are executed. Some of them " + COLOR_TERTIARY + "failed" + COLOR_RESET + " in this process\n"

    if ("Test on NUCLEO-L4R5ZI (TFLM)" in modules or
        "Test on NUCLEO-L4R5ZI (Edge Impulse)" in modules or
        "Test on NUCLEO-L4R5ZI (Ekkono)" in modules or
        "Test on RenesasRX65N (TFLM)" in modules or
        "Test on RenesasRX65N (Edge Impulse)" in modules or
        "Test on RenesasRX65N (Ekkono)" in modules or
        "Test on RenesasRX65N (eAI Translator)" in modules):
        page += "You can find the results in the {} directory\n".format(save_dir)
    _clear_console()
    print(page)


if __name__ == "__main__":
    main()
