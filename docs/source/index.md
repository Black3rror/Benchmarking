# EdgeMark

The *EdgeMark* project comes with two goals in mind:

- To provide an automation tool that can help developers, engineers, and researchers to save time and effort in the process of creating an embedded AI (eAI) system.
- To benchmark several eAI tools and libraries and provide a comparison of their performance and resource usage.

## Modules

We have tried to follow a modular approach in the design of this tool. This allows users to not only extend the benchmarking process by adding new tools and libraries but also enables them to utilize specific parts of the tool for their own applications. For example, the tool can be easily employed for generation and conversion of models to different formats using various optimization techniques.

![Modules](figures/modules.png#only-light)
![Modules](figures/modules - dark.png#only-dark)

Currently, the following modules are available:

- **Generate TF models**: Generates, compiles, trains, and evaluates TensorFlow models based on user-provided configuration files.
- **Generate Ekkono models**: Similar to the *Generate TF models* module and based on the same configuration files, it can generate and train models for the Ekkono platform. The main outputs of this module are the Ekkono Edge models and their Crystal representations in C.
- **Convert to TFLite**: Converts TensorFlow models to TensorFlow Lite format. Various optimization options are available during the conversion process.
- **Convert to TFLM**: Converts TensorFlow Lite models to their representation in TensorFlow Lite for Microcontrollers format (C++ code).
- **Convert to Edge Impulse**: Converts TensorFlow Lite models into the Edge Impulse format (C++ code).
- **Convert to eAI Translator**: Prepares the necessary data files for the eAI Translator's C project. Please note, however, the translation of the model cannot be automated - you will need to manually complete this step using the eAI Translator tool.
- **Test on NUCLEO-L4R5ZI**: Tests TFLM/Edge Impulse/Ekkono models on the NUCLEO-L4R5ZI board. The corresponding project will be compiled and flashed to the board. The results will be collected and stored in a YAML file, as well as being summarized in an Excel file. The results include: inference time, flash size, RAM usage, and error.
- **Test on RenesasRX65N**: Similar to the *Test on NUCLEO-L4R5ZI* module, but designed for testing on the Renesas RX65N board.

<br/>

In the next section, we will cover the details of how to use this tool.
