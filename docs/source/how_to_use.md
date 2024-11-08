# How To Use

It is easy to use *EdgeMark*. You just need to:

1. Clone the repository and navigate to the root directory of the project.
    ```bash
    git clone https://github.com/Black3rror/EdgeMark.git
    cd EdgeMark
    ```

2. Run the *main* Python script. (1)
{ .annotate }

    1. :man_raising_hand: We recommend running the script inside a virtual environment like Conda.

    === "Python"
        ```bash
        python -m edgemark.main
        ```

    === "Makefile"

        ```bash
        make # (1)!
        ```
        { .annotate }

        1. The Makefile is a wrapper around the Python script. It will run the script with the same command as in *Python*. It exists just for convenience.

3. Follow the instructions on the screen.

!!! info ""

    On the first run, the script will automatically install the necessary dependencies for itself. On the second run, the user can ask the script to install any additional Python dependencies required for the automation process. Third time is the charm! You should be greeted with the tool's main menu, where you'll be asked to choose your desired action.

## Project Setup

After cloning the repository, several tools need to be connected to the project in order to provide certain functionalities. While these requirements are mostly guided through the instructions of the *main* script, we will provide more detailed explanations here.

### Edge Impulse secrets

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> Edge Impulse v1.56.13 (date 12-09-2024) was used in our tests.</small>

To use the *Convert to Edge Impulse* module, you need to provide your Edge Impulse API Key and Project ID. For this purpose, you need to have an account on Edge Impulse. If you don't have one, you can create it [here](https://www.edgeimpulse.com).

Once your account is set up, locate the API Key and Project ID:

- The API key can be found in the *Keys* section of your project. (1)
{ .annotate }

    1.  :man_raising_hand: You can also access this page by its address: *https://studio.edgeimpulse.com/studio/{project_id}/keys*.

- The Project ID can be found on the project's main page under the *Project info* section. (1)
{ .annotate }
    1.  :man_raising_hand: You can also find it by looking at the URL of the page: *https://studio.edgeimpulse.com/studio/^^{project_id}^^*.

After obtaining this information, provide it to the project's *main* script in order to make the connection. The script will ask you for the API Key and Project ID (if they don't already exist) and saves them in *edgemark/models/platforms/EI/configs/EI_converter_user_config.yaml*.

Now, you are ready to use the *Convert to Edge Impulse* module.

### Ekkono SDK

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> Ekkono SDK v23.10 was used in our tests.</small>

As Ekkono is not a free tool, you will need a valid license to use it. You can learn more about Ekkono on their [website](https://www.ekkono.ai). After obtaining the license, you can either install it using *pip* or provide its path (1) to the *main* script.
{ .annotate }

1.  :man_raising_hand: For example, if you place the files in *edgemark/models/platforms/Ekkono*, the path to the Python wheel would be *edgemark/models/platforms/Ekkono/ekkono-sdk/primer/python/{distribution}/{python-version}/ekkono.primer-{name-suffix}.whl*.

Further, you need to download Ekkono's C inference code and place it in the corresponding unzipped project directory for your specific hardware platform. (1)
{ .annotate }

1.  :man_raising_hand: For NUCLEO-L4R5ZI, it will be *edgemark/Hardware/STM32/NUCLEO-L4R5ZI_Ekkono/Core/Inc/Ekkono_lib* and for RenesasRX65N, it will be *edgemark/Hardware/Renesas/RenesasRX_Ekkono/src/Ekkono_lib*.

At this point, you should be able to both generate Ekkono models using the *Generate Ekkono models* module and proceed with *Test on NUCLEO-L4R5ZI/RenesasRX65N*.

### STM32CubeIDE

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> STM32CubeIDE v1.14.1 was used in our tests.</small>

STM32CubeIDE is required to compile the C/C++ projects for the STM32 boards. You can download it from the [ST website](https://www.st.com/en/development-tools/stm32cubeide.html). After installation, you need to provide its executable path (1) and the workspace directory (2) to the *main* script. When prompted by the script, enter these details, which will be saved in the file *edgemark/models/automate/hardware_types/NUCLEO-L4R5ZI/configs/hardware_user_config.yaml*.
{ .annotate }

1.  :man_raising_hand: The executable path can be found at *{installation_dir}/STM32CubeIDE_{version}/STM32CubeIDE/stm32cubeide.exe*. For example, *C:/ST/STM32CubeIDE_1.14.1/STM32CubeIDE/stm32cubeide.exe*.
2.  :man_raising_hand: You can locate the workspace directory by opening STM32CubeIDE and navigating to *File > Switch Workspace > Other...*.

The next step is to ensure the projects are correctly set up and added to the workspace. The zipped projects can be found in *edgemark/Hardware/STM32*. Please unzip them, and then add them to the workspace by opening the STM32CubeIDE and selecting *File > Open Projects from File System...*. Choose the unzipped project folder and click *Finish*. The project should now appear in the *Project Explorer*.

!!! note
    Please note that the Ekkono library is excluded from the *NUCLEO-L4R5ZI_Ekkono* project. You need to add it manually. After obtaining the license, you need to download their C inference code and place it in *edgemark/Hardware/STM32/NUCLEO-L4R5ZI_Ekkono/Core/Inc/Ekkono_lib*

Now, the automation script should be able to compile the projects. To upload the compiled programs to the board, the automation script requires the STM32CubeProgrammer CLI, which we'll cover in the next step.

### STM32CubeCLT

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> STM32CubeProgrammer v2.16.0 was used in our tests.</small>

STM32Programmer CLI as a part of STM32CubeCLT allows the automation script to upload the compiled programs to the STM32 boards. You can download it from the [ST website](https://www.st.com/en/development-tools/stm32cubeclt.html). After installation, provide its executable path (1) to the *main* script when prompted. The path will be saved in *edgemark/models/automate/hardware_types/NUCLEO-L4R5ZI/configs/hardware_user_config.yaml*.
{ .annotate }

1.  :man_raising_hand: The executable will be in *{installation_dir}/STM32CubeCLT_{version}/STM32CubeProgrammer/bin/STM32_programmer_CLI.exe*. For example, *C:/ST/STM32CubeCLT_1.15.1/STM32CubeProgrammer/bin/STM32_programmer_CLI.exe*.

!!! tip
    When connecting your NUCLUEO-L4R5ZI board to your PC, you need to have the ST-Link driver installed. It's recommended to upload a project to your board for the first time using STM32CubeIDE to verify that the driver is correctly installed and everything is functioning as expected.

At this point, you should be ready to use the *Test on NUCLEO-L4R5ZI* module.

### Renesas e2 studio

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> Renesas e2 studio v24.1.1 was used in our tests.</small>

Renesas e2 studio is required to compile the C/C++ projects for the Renesas boards. You can download it from the [Renesas website](https://www.renesas.com/us/en/software-tool/e-studio). After installation, you need to provide its executable path (1) and the workspace directory (2) to the *main* script. When prompted by the script, enter these details, which will be saved in the file *edgemark/models/automate/hardware_types/RenesasRX65N/configs/hardware_user_config.yaml*.
{ .annotate }

1.  :man_raising_hand: The executable will be *{installation_dir}/Renesas/e2_studio/eclipse/e2studioc.exe*. For example, *C:/Renesas/e2_studio/eclipse/e2studioc.exe*.
2.  :man_raising_hand: You can locate the workspace directory by opening e2 studio and navigating to *File > Switch Workspace > Other...*.

The next step is to ensure the projects are correctly set up and added to the workspace. The zipped projects can be found in *edgemark/Hardware/Renesas*. Please unzip them, and then add them to the workspace by opening e2 studio and selecting *File > Open Projects from File System...*. Choose the unzipped project folder and click *Finish*. The project should now appear in the *Project Explorer*.

!!! note
    Please note that the Ekkono library is excluded from the *RenesasRX_Ekkono* project. You need to add it manually. After obtaining the license, you need to download their C inference code and place it in *edgemark/Hardware/Renesas/RenesasRX_Ekkono/src/Ekkono_lib*

Now, the automation script should be able to compile the projects. To upload the compiled programs to the board, the automation script requires the Renesas Flash Programmer, which we'll cover in the next step.

### Renesas Flash Programmer

<small><span style="color:rgb(156, 39, 176);"> :material-tag-outline: </span> Renesas Flash Programmer v3.15.00 was used in our tests.</small>

Renesas Flash Programmer allows the automation script to upload the compiled programs to the Renesas boards. You can download it from the [Renesas website](https://www.renesas.com/us/en/software-tool/renesas-flash-programmer-programming-gui#downloads). After installation, open it and create a new project by clicking *File > New Project...*. In the project creation window, configure it as follows:

- *Microcontroller*: RX65x
- *Project Name*: [Your preferred name]
- *Project Folder*: [Your preferred folder]
- *Tool*: E2 emulator Lite
- *Interface*: FINE

Additionally, click *Tool Details*, go to *Reset Settings*, and configure *Reset signal at Disconnection* to *Reset Pin as Hi-Z*. After these steps, click *Connect*.

In the next step you need to provide the RFP's executable path (1) and its project path (2) to the *main* script. Run the script and provide them when asked. The script will save them in *edgemark/models/automate/hardware_types/RenesasRX65N/configs/hardware_user_config.yaml*.
{ .annotate }

1.  :man_raising_hand: The executable will be *{installation_dir}/Renesas Electronics/Programming Tools/Renesas Flash Programmer V{version}/RFPV{version}.exe*. For example, *C:/Program Files (x86)/Renesas Electronics/Programming Tools/Renesas Flash Programmer V3.15/RFPV3.exe*
2.  :man_raising_hand: The project file is a *.rpj* file, located in the folder you selected earlier.

!!! tip
    We recommend uploading a project to your board manually via Renesas e2 studio the first time to ensure everything is set up correctly.

Unlike the NUCLEO-L4R5ZI board, the Renesas RX65N target board does not connect to the PC via its programmer for serial communication. To establish a serial connection, you’ll need to connect the board to the PC using a USB-to-TTL cable. Connect the USB side to the PC, and on the TTL side, connect the *GND* to the *GND* pin on the board (pin 61), and the *RX* to the *TX* pin on the board (pin 45). Be sure to install the necessary driver on your PC before proceeding.

Now, you're ready to use the *Test on RenesasRX65N* module.

## Modules

As mentioned in the [Home page](index.md), the project consists of multiple modules. The modules can be run one after another to reach the final goal. The *edgemark.main* script provides an intuitive interface for managing the execution of these modules. Below, we describe each module, their requirements, relationships, and key configurations that might be of your interest to change.

![Modules](figures/modules.png#only-light)
![Modules](figures/modules - dark.png#only-dark)

### Generate TF models

You should describe the desired models in *Model Description Files*. The *Model Generator* produces TensorFlow models based on these files.

The *Model Description Files* are YAML files located in *target_models* directory. The name of the model will be the path to the file without the extension. If the name of the file or any directory leading to that file begins with a dot (.), the file will be ignored. Please refer to the [Model Description Files](model_description_files.md) page for more detailed instructions on creating these files.

Configurations for this module are located in *edgemark/models/platforms/TensorFlow/configs/model_generator_config.yaml*. Below are the primary configurations you may want to modify:

- *wandb_online*: Enables cloud-based logging through W&B (Weights & Biases). Otherwise, logs are saved locally.
- *wandb_project_name*: Specifies the W&B project name.
- *train_models*: If enabled, the generated models will undergo training.
- *evaluate_models*: If enabled, the models will be evaluated after training.
- *measure_execution_time*: If enabled, the execution time of the models on your machine will be measured.
- *n_representative_data*: Number of representative data that will be used in some conversions of the *Convert to TFLite* module.
- *n_eqcheck_data*: Number of data samples that will be used to check the equivalence of the original and on-board models.
- *epochs*: If defined, overrides the number of epochs specified in the *Model Description Files*. This is useful for demonstration or debugging purposes.

### Generate Ekkono models

!!! warning "License Required"

    Ekkono is a commercial product and requires a valid license. You can find more information on their [website](https://www.ekkono.ai/).

Same as *Generate TF models*, this module takes in the *Model Description Files*. These files can be the same ones used for generating TensorFlow models; however, please note the limitations of Ekkono. For example, Ekkono does not support CNNs, so any file containing CNN layers will result in an error.

Since Ekkono operates in its own environment, we have streamlined the generation and conversion steps into a single module. The output of this module consists of Ekkono's Crystal models along with the necessary C files for benchmarking.

The configuration is similar to the TensorFlow module and can be found in *edgemark/models/platforms/Ekkono/configs/model_generator_config.yaml*. Key configurations include:

- *wandb_online*: Enables cloud-based logging through W&B (Weights & Biases). Otherwise, logs are saved locally.
- *wandb_project_name*: Specifies the W&B project name.
- *train_models*: If enabled, the generated models will undergo training.
- *evaluate_models*: If enabled, the models will be evaluated after training.
- *measure_execution_time*: If enabled, the execution time of the models on your machine will be measured.
- *n_eqcheck_data*: Number of data samples that will be used to check the equivalence of the original and on-board models.
- *epochs*: If defined, overrides the number of epochs specified in the *Model Description Files*. This is useful for demonstration or debugging purposes.

### Convert to TFLite

Once TensorFlow models are generated, you can use the *Convert to TFLite* module to convert the models into TFLite format. Various optimizations can be applied to the models during this conversion.

The configuration file is located at *edgemark/models/platforms/TFLite/configs/TFLite_converter_config.yaml*. Key configurations include:

- *conversion_timeout*: The maximum time in seconds that the conversion process can take.
- *optimizations*: List of optimizations that should be applied to each model.

The available optimizations are:

- *basic*: No optimization, just the standard conversion.
- *q_dynamic*: Dynamic range quantization. Weights are quantized to 8-bit integers, while activations are stored in 32-bit floats. Activations can be dynamically quantized to 8-bit integers to accelerate inference and then dequantized back to 32-bit floats for storage. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_quant).
- *q_full_int*: Full integer quantization. Both weights and activations are quantized to 8-bit integers. However, the input and output remain in 32-bit floats. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_integer_quant).
- *q_full_int_only*: Similar to *q_full_int*, but in this case, everything is quantized to integers without fallback to floats. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_integer_quant).
- *q_16x8*: 16x8 quantization. To improve the accuracy of the quantized model, the activations are quantized to 16-bit integers while the weights are quantized to 8-bit integers. Similar to *q_full_int*, the input and output are left in 32-bit floats. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_integer_quant_16x8).
- *q_16x8_int_only*: Same as *q_16x8*, but everything is quantized to integers without fallback to floats. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_integer_quant_16x8).
- *q_float16*: Float16 quantization. Weights are quantized to 16-bit floats, reducing model size with minimal impact on accuracy. Learn more [here](https://ai.google.dev/edge/litert/models/post_training_float16_quant).
- *p_{percentage}*: Post-training weight pruning. For example, "p_75" means 75% of the weights will be pruned. Learn more [here](https://www.tensorflow.org/model_optimization/guide/pruning).
- *c_{clusters}*: Weight clustering. For example, "c_16" clusters the weights into 16 groups. Learn more [here](https://www.tensorflow.org/model_optimization/guide/clustering).

!!! tip
    You can combine multiple optimizations using a plus (+) sign. For example, "p_25 + c_16 + q_full_int_only" will prune 25% of the weights, cluster the weights into 16 groups, and fully quantize everything into integers. Note: Quantization schemes should always be applied as a final step.

Once converted, the model is ready for further conversion to TFLM, Edge Impulse, or eAI Translator formats.

### Convert to TFLM

This module requires the TFLite models generated by the *Convert to TFLite* module and their corresponding *TFLM_info* files from *Generate TF models*. The latter contains information like an estimation of the required *arena_size* and the necessary operator resolver functions.

!!! note
    TFLM does not support *q_float16* quantization. Models with this optimization will be ignored.

The output of this module is a set of C++ files for each model, which can be used for benchmarking or integrated into any other C++ project.

### Convert to Edge Impulse

This module converts TFLite models into the Edge Impulse format by uploading the models to the Edge Impulse cloud and downloading the required files. The user must have an Edge Impulse account and provide their *API Key* and *Project ID*. These should be placed in the file located at *edgemark/models/platforms/EI/configs/EI_converter_user_config.yaml* under the keys *ei_api_key* and *ei_project_id*, respectively.

!!! note
    Edge Impulse only supports the *basic* and *q_full_int_only* optimizations. Other types of optimizations will be ignored.

### Convert to eAI Translator

This module in fact does not convert the models to eAI Translator format, but it generates the necessary C data files required for benchmarking. Since Renesas eAI Translator does not provide an automated method for converting models, users must manually perform the conversion. In the stage of testing the models on the board, the script will check if the eAI Translator models are available and if not, it will inform the user where they can find the TFLite models and where the eAI Translator models should be placed.

!!! note
    Renesas eAI Translator only supports the *basic* and *q_full_int_only* optimizations. Other types of optimizations will be ignored.

### Test on NUCLEO-L4R5ZI/RenesasRX65N

This module uses the generated C/C++ files from *Generate Ekkono models*, *Convert to TFLM*, *Convert to Edge Impulse*, or *Convert to eAI Translator* and integrates them into a benchmarking project for either the NUCLEO-L4R5ZI or RenesasRX65N board. Once integrated, the module compiles the program, uploads it to the board, and captures the benchmarking output.

An overview of the results are saved in an Excel file, while detailed results can be found in each model's directory. Further, comparison plots will be generated to visualize the performance of different model types.

The results are: *Execution Time*, *Flash and RAM Usage*, and *Error*.

!!! tip
    All details, generated files, results and error messages related to each model can be found in the model's directory under *saved_models/{TensorFlow/Ekkono}/{model_type}/{time_tag}*

The configuration options for this module can be found in *edgemark/models/automate/configs/automate_config.yaml* which includes:

- *arena_finder*: If enabled, the script will attempt to find the optimal arena size for each TFLM model.
- *benchmark_overall_timeout*: Specifies the maximum duration in seconds for the entire benchmarking process.
- *benchmark_silence_timeout*: Specifies the maximum duration in seconds that the board is allowed to remain silent without any output.

### (Bonus) Result plotter

The *Result plotter* script in *edgemark/models/utils/result_plotter.py* will help you generate figures comparing the results of your benchmark. Please see its API for more information.

!!! success "You are ready to go!"

    You now have all the information you need to use the EdgeMark. We hope you find it useful 😊
