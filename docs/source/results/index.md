# Results

In pursue of evaluating the performance of different setups, we conducted a series of studies. The results are detailed on the following pages.

!!! example "Test Demon!"
    The results you will find in the following pages are based on thousands of tests, with each test taking approximately 2-3 minutes to complete.

Each test measures four key metrics:

- Execution Time: The time it takes to run the model on the target device. It is measured by iteratively running the model for 10 times and taking the average. (1)
{ .annotate }

    1.  :man_raising_hand: The input to the model remains the same for each iteration. Changing the input, having a warm-up phase, or re-running the test have negligible to no effect on the results. Additionally, the standard deviation of the execution time is negligible and is saved in the detailed results.

- Error: The error rate of the model. It is calculated as the average of the normalized absolute difference between the model's output and the expected output (the output produced when the model is run on a PC). To determine this, 10 different inputs are fed into the model, and the average error rate is computed. The standard deviation of the error rate is also included in the detailed results. An error rate below 0.05 is generally considered acceptable for a model.

- Flash Size: The model's flash memory size. (1)
{ .annotate }

    1.  :man_raising_hand: The GCC compiler reports the sizes of the *text*, *data*, and *bss* sections.
    <p>By default, *Flash* = *text* + *data*, but since the tested tools have a negligible effect on the *data* section and the *data* section is primarily used for storing data samples (which we want exclude from affecting the flash size), we assume *Flash* = *text*. The reported values in graphs are the difference between flash of the programs and corresponding base programs. Hence, these values indicate the flash size of the tool and the model.</p>
    <p><ul>
        <li>For the NUCLEO-L4R5ZI C/C++ project, the base program (including printf and timer functionalities but without any model or library) occupies (text=28000, data=476, and bss=3568) bytes.</li>
        <li>For the Renesas RX65N C project, the base program occupies (text=30972, data=3508, and bss=7368) bytes.</li>
        <li>For the Renesas RX65N C++ project, the base program occupies (text=33372, data=3500, and bss=7376) bytes.</li>
        <li>The minimum program size for each tool can be inferred from its *FC_0* model data.</li>
    </ul></p>

- RAM Usage: The program's RAM usage. (1)
{ .annotate }

    1.  :man_raising_hand: Since all tested tools follow a static memory allocation strategy, the RAM usage can be inferred from the GCC compiler's report. Typically, *RAM* = *data* + *bss*, but since the *data* section primarily holds data samples (which we do not want to affect RAM calculations), we assume *RAM* = *bss*. The reported values in graphs are the difference between RAM of the programs and corresponding base programs. Hence, these values indicate the RAM size of the tool and the model.
    <p>For TFLM, we set the *arena_size* parameter, which modifies the size of the *bss* section. To find the minimum viable *arena_size* for each model, we incrementally increased the *arena_size* with a small step-size until the program ran successfully. The *bss* size of the program with the minimum *arena_size* is taken as the program's RAM usage.</p>

!!! warning "Subjective Summary"
    The *Summary* section of each study offers a brief overview of the results. Please note that these summaries are based on overall behavior of models and may be somewhat subjective.


## Studies

- [TFLM Quantizations](TFLM_quantizations.md): Evaluating the performance of different quantization schemes.
- [TFLM Pruning and Clustering](TFLM_pruning_and_clustering.md): Exploring the effects of pruning and clustering.
- [TFLM vs Edge Impulse](TFLM_vs_Edge_Impulse.md): A comparison between TFLM and Edge Impulse.
- [TFLM vs Ekkono](TFLM_vs_Ekkono.md): A comparison between TFLM and Ekkono.
- [TFLM vs eAI Translator](TFLM_vs_eAI_Translator.md): A comparison between TFLM and Renesas eAI Translator.
- [RNN, LSTM, GRU](RNN_LSTM_GRU.md): Comparing the performance of different RNNs.
- [Compiler Optimization Levels](Compiler_optimization_levels.md): The impact of varying compiler optimization levels.
- [Importance of FPU](Importance_of_FPU.md): Situations where the Floating Point Unit (FPU) becomes beneficial.
- [STM vs Renesas](STM_vs_Renesas.md): A comparison between the NUCLEO-L4R5ZI and Renesas RX65N boards.
- [GCC vs CCRX](GCC_vs_CCRX.md): A comparison of the GCC and CCRX compilers for the Renesas RX65N.

## Models

In our experiments, we used four types of models:

- [FC](#fc): Fully connected neural network.
- [CNN](#cnn): Convolutional neural network.
- [RNN](#rnn): Recurrent neural network. It can be either Simple RNN, LSTM, or GRU.
- [TinyMLPerf](#tinymlperf): Models from the MLPerf Tiny benchmark suite.

### FC

We utilized 11 FC models in our experiments. These models consist of multiple fully connected layers with *Sigmoid* or *ReLU* activation functions. Classification models include a *Softmax* layer at the end. Some models may also use *dropout* and *batch normalization* layers.

*FC_0* is the simplest model, containing just one neuron in both the input and output layers. It's useful for comparing the minimum resource requirements across different tools.

As the model number increases, so do the network’s size and complexity. The figure below provides details on the number of parameters and MACs in each model.

<figure markdown="span">
    ![FC parameters and MACs](../figures/FC_params_MACs.png#only-light){: style="width:75%"}
    ![FC parameters and MACs](../figures/FC_params_MACs - dark.png#only-dark){: style="width:75%"}
    <figcaption>FC parameters and MACs</figcaption>
</figure>

### CNN

We employed 7 CNN models in our experiments. These models consist of multiple convolutional layers and, in some cases, additional fully connected (FC) layers. In addition to activation functions (*Sigmoid*, *ReLU*), dropout, and batch normalization layers, CNN models may also feature *pooling* layers.

As with the FC models, complexity increases with model number. The figure below outlines the number of parameters and MACs for each CNN model.

<figure markdown="span">
    ![CNN parameters and MACs](../figures/CNN_params_MACs.png#only-light){: style="width:75%"}
    ![CNN parameters and MACs](../figures/CNN_params_MACs - dark.png#only-dark){: style="width:75%"}
    <figcaption>CNN parameters and MACs</figcaption>
</figure>

### RNN

We utilized 7 different RNN models during our experiments. These include three Simple RNN models of varying sizes (*simple_0* represents an almost minimal RNN), two additional Simple RNNs trained on a collection of Shakespeare’s works, one LSTM model, and one GRU model. The Shakespeare models include an embedding layer at the beginning.

The table below provides detailed information about the RNN models.

| Model         | RNN Units | Sequence Length | Parameters | MACs    |
|---------------|-----------|-----------------|------------|---------|
| Simple 0      | 1         | 2               | 5          | 9       |
| Simple 1      | 64        | 100             | 8288       | 827200  |
| Simple 2      | 128       | 100             | 32960      | 3292800 |
| Shakespeare 1 | 64        | 100             | 12513      | 1056300 |
| Shakespeare 2 | 128       | 100             | 37249      | 3321900 |
| LSTM          | 64        | 100             | 26912      | 2702400 |
| GRU           | 64        | 100             | 20896      | 2094400 |

### TinyMLPerf

The MLPerf Tiny benchmark suite includes 4 models. While in this project we have the possibility to create these models with different options, we have loaded the default models provided by the benchmark suite.

Below is a figure that highlights the number of parameters and MACs for these models.

<figure markdown="span">
    ![TinyMLPerf parameters and MACs](../figures/TinyMLPerf_params_MACs.png#only-light){: style="width:75%"}
    ![TinyMLPerf parameters and MACs](../figures/TinyMLPerf_params_MACs - dark.png#only-dark){: style="width:75%"}
    <figcaption>TinyMLPerf parameters and MACs</figcaption>
</figure>


<style>
    figcaption {
        text-align: center;
        font-size: 14px;
        color: rgb(117, 117, 117);
        margin-top: 5px;
    }

    figcaption:hover {
        color: rgb(186, 104, 200);
    }
</style>
