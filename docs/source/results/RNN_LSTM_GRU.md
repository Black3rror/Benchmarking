# RNN, LSTM, GRU

This study highlights the resource requirements of Simple RNN, LSTM, and GRU models.

Only TFLM could be used for these models (1). Also, all the models were deployed on the NUCLEO-L4R5ZI board.
{ .annotate }

1.  :man_raising_hand: Edge Impulse requires an enterprise account, Renesas eAI Translator cannot convert RNNs (might be solved by further adjustments), and Ekkono does not support RNNs.

*Simple 0* is excluded from the figures due to its negligible resource requirements compared to the other models and keeping the figures readable. *Simple 2* is also excluded from the figures because its *basic* version failed to run. All models' information is available in the tables below.

| Model         | Variant   | Parameters | MACs    | Error    | Exe (ms) | Flash (kB)  | RAM (kB)    |
|---------------|-----------|------------|---------|----------|----------|-------------|-------------|
| Simple 0      | basic     | 5          | 9       | 0        | 0.107    | 110.4375    | 8.140625    |
| Simple 0      | int8 only | 5          | 9       | 0.005785 | 0.14567  | 111.640625  | 7.1328125   |
| Simple 1      | basic     | 8288       | 827200  | 0        | 107.2074 | 218.65625   | 112.6328125 |
| Simple 1      | int8 only | 8288       | 827200  | 0.004366 | 292.9049 | 590.8125    | 103.2578125 |
| Simple 2      | basic     | 32960      | 3292800 | -        | -        | -           | -           |
| Simple 2      | int8 only | 32960      | 3292800 | 0.004072 | 292.9049 | 615.1875    | 106.3828125 |
| Shakespeare 1 | basic     | 12513      | 1056300 | 0        | 141.1776 | 251.578125  | 127.625     |
| Shakespeare 1 | int8 only | 12513      | 1056300 | 0.020862 | 168.4068 | 620.84375   | 107.5859375 |
| Shakespeare 2 | basic     | 37249      | 3321900 | 0        | 377.1719 | 348.203125  | 175.625     |
| Shakespeare 2 | int8 only | 37249      | 3321900 | 0.021797 | 323.4767 | 645.1953125 | 107.5859375 |
| LSTM          | basic     | 26912      | 2702400 | 0        | 362.2378 | 472.9609375 | 268.734375  |
| LSTM          | int8 only | 26912      | 2702400 | 0.013989 | 565.3515 | 769.8203125 | 258.359375  |
| GRU           | basic     | 20896      | 2094400 | 0        | 276.7367 | 496.984375  | 298.734375  |
| GRU           | int8 only | 20896      | 2094400 | 0.044773 | 374.4158 | 809.7890625 | 295.359375  |

:man_raising_hand: *Simple_1* is named *Simple* in figures.

<br/>

## Models

<figure markdown="span">
    <img src="../../figures/results/TFLM - RNN - STM/params_MACs.png#only-light" alt="RNN parameters and MACs">
    <img src="../../figures/results/TFLM - RNN - STM/dark/params_MACs.png#only-dark" alt="RNN parameters and MACs">
    <figcaption>RNN parameters and MACs</figcaption>
</figure>

## Error

<figure markdown="span">
    <img src="../../figures/results/TFLM - RNN - STM/error.png#only-light" alt="RNN error">
    <img src="../../figures/results/TFLM - RNN - STM/dark/error.png#only-dark" alt="RNN error">
    <figcaption>RNN error</figcaption>
</figure>

## Execution Time

<figure markdown="span">
    <img src="../../figures/results/TFLM - RNN - STM/exe.png#only-light" alt="RNN execution time">
    <img src="../../figures/results/TFLM - RNN - STM/dark/exe.png#only-dark" alt="RNN execution time">
    <figcaption>RNN execution time</figcaption>
</figure>

## Flash Size

<figure markdown="span">
    <img src="../../figures/results/TFLM - RNN - STM/flash.png#only-light" alt="RNN flash size">
    <img src="../../figures/results/TFLM - RNN - STM/dark/flash.png#only-dark" alt="RNN flash size">
    <figcaption>RNN flash size</figcaption>
</figure>

## RAM Usage

<figure markdown="span">
    <img src="../../figures/results/TFLM - RNN - STM/ram.png#only-light" alt="RNN RAM usage">
    <img src="../../figures/results/TFLM - RNN - STM/dark/ram.png#only-dark" alt="RNN RAM usage">
    <figcaption>RNN RAM usage</figcaption>
</figure>

## Summary

- **Model Correctness**: Except the *basic* version of *Simple 2* which runs out of memory, all models got relatively acceptable error rates. The most concerning case belongs to *GRU* which might require some attention.

- **Execution Time**: It is surprising that except for *Shakespeare 2*, the *int8 only* versions of the models have higher execution times.

- **Flash Size**: It is surprising that the *int8 only* versions of the models have larger flash sizes.

- **RAM Usage**: *int8 only* requires less RAM. (1)
{ .annotate }

    1.  :man_raising_hand: We expected the *int8 only* to have the advantage with a bigger margin.

- **Conclusion**: In most cases, the *basic* version of the models is more efficient.

??? info "LSTM vs GRU"
    It's believed that LSTM is more powerful than GRU, but GRU is more parameter and computation efficient. Also, in our study, we see that LSTM has more parameters and MACs than GRU. However, it is interesting that using TFLM, GRU has a larger flash size and RAM usage than LSTM. Still, it's executed faster than LSTM.


<style>
    figure img {
        width: 65%;
        margin: auto;
        display: block;
    }

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
