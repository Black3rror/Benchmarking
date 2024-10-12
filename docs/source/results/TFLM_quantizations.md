# TFLM Quantizations

In this study, we evaluate the performance of various quantization schemes using the TensorFlow Lite for Microcontrollers (TFLM) platform. We chose TFLM because it is the only framework that supports the majority of available quantization methods. (1)
{ .annotate }

1.  :man_raising_hand: TFLM supports *basic*, *dynamic*, *int8*, *int8 only*, *16x8*, and *16x8 only* quantizations, but does not support {--*float16*--}. In contrast, Edge Impulse and Renesas eAI Translator only support *basic* and *int8 only* quantizations, while Ekkono supports only *basic* quantization.

All FC and CNN models were tested on the NUCLEO-L4R5ZI and RenesasRX65N boards.

!!! info "CMSIS-NN"
    The *CMSIS-NN* library can help to accelerate the execution of quantized models (1). This library is designed for ARM Cortex-M microcontrollers and was used with the NUCLEO-L4R5ZI board. Renesas has developed its own *CMSIS-NN* library for RX microcontrollers, but it was not employed in our evaluations.
    { .annotate }

    1.  :man_raising_hand: "CMSIS-NN supports int8 and int16 activations and int8 weights. It also supports int4 packed weights to some extent." [reference](https://github.com/tensorflow/tflite-micro/issues/2473#issuecomment-1966191977)

<br/>

Model Type:
<select id="modelTypeSelect">
    <option value="Any">Any</option>
    <option value="FC">FC</option>
    <option value="CNN">CNN</option>
</select>
&nbsp;&nbsp;&nbsp;&nbsp;Board:
<select id="boardSelect">
    <option value="Any">Any</option>
    <option value="STM">NUCLEO-L4R5ZI</option>
    <option value="Renesas">RenesasRX65N</option>
</select>

## Models

<div class="image-container">
<figure markdown="span" class="FC STM Renesas">
    <img src="../../figures/results/Quants/Quants - FC - STM/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/Quants/Quants - FC - STM/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN STM Renesas">
    <img src="../../figures/results/Quants/Quants - CNN - STM/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/Quants/Quants - CNN - STM/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/Quants/Quants - FC - STM/error.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - FC - STM/dark/error.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/error.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/dark/error.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/Quants/Quants - CNN - STM/error.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - CNN - STM/dark/error.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/error.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/dark/error.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/Quants/Quants - FC - STM/exe.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - FC - STM/dark/exe.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/exe.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/dark/exe.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/Quants/Quants - CNN - STM/exe.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - CNN - STM/dark/exe.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/exe.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/dark/exe.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/Quants/Quants - FC - STM/flash.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - FC - STM/dark/flash.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/flash.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/dark/flash.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/Quants/Quants - CNN - STM/flash.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - CNN - STM/dark/flash.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/flash.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/dark/flash.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/Quants/Quants - FC - STM/ram.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - FC - STM/dark/ram.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/ram.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - FC - Renesas/dark/ram.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/Quants/Quants - CNN - STM/ram.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Quants/Quants - CNN - STM/dark/ram.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/ram.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/Quants/Quants - CNN - Renesas/dark/ram.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**:
    - The *float16* quantization scheme is not supported by any of the tested frameworks and should be disregarded.

    - The *dynamic* quantization also lacks a proper support (1). So, it is recommended to avoid using the *dynamic* quantization.
    { .annotate }

        1.  :man_raising_hand: In most cases, either the models fail to run on the boards, or they have an unacceptable error. Even when ignoring the errors, we cannot say that the *dynamic* quantization has any superiority over other types of quantizations.

    - The RenesasRX65N board is unable to run some of the models. (1)
    { .annotate }

        1.  :man_raising_hand: The RenesasRX65N board cannot run the *basic*, *int8 only*, and *16x8 only* versions of the *FC_1* and *FC_2* models. The program halts during their execution.

    - The error rates of all other quantization schemes are acceptable. (1)
    { .annotate }

        1.  :man_raising_hand: *basic* is perfect, *int8* variants and *16x8* variants have a negligible errors, *16x8* being better than *int8*.

- **Execution Time**: It is complicated and hard to say which quantization scheme is better.
    - *FC Models*: The *only* variant of *int8* and *16x8* is faster.
        - *Small Models*: *16x8 only* is the best.
        - *Large Models*: *int8 only* is the best.
    - *CNN Models*: The *basic* model is very slow (1). The *int8* and *16x8* variants are close to each other. (2)
{ .annotate }

        1.  :man_raising_hand: Because of using the *CMSIS-NN*, the quantized models (especially the CNN quantized models) experience a significant speedup on the NUCLEO-L4R5ZI board. This library is not utilized on the RenesasRX65N board.
        2.  :man_raising_hand: Still, to some point, following the same pattern as FC for small and large models.

- **Flash Size**: The *basic* model gets worse as model size increases (1). The others are almost the same. (2)
{ .annotate }

    1.  :man_raising_hand: All quantization schemes start with relatively similar flash sizes, but as the model scales, the *basic* version's flash size increases fourfold (four bytes per parameter), whereas the other variants increase by only one byte per parameter.
    2.  :man_raising_hand: The *only* variants of *int8* and *16x8* are slightly more efficient in terms of flash usage.

- **RAM Usage**: It is complicated, but *int8 only* is either equally the best or better than all others.

- **Conclusion**: The choice of quantization scheme depends on many factors, however, the *int8 only* quantization is a good choice for most cases. (1)
{ .annotate }

    1.  :man_raising_hand: The *basic* and *int8 only* variants are the two model types that will be used in other studies.


<style>
    .image-container {
        display: flex;
        flex-wrap: wrap;        /* Allow images and captions to wrap onto the next line */
        justify-content: space-between;
    }

    .image-container figure {
        width: 48%;
        margin: 0 0 10px 0;     /* Bottom margin to provide space between rows */
    }

    .image-container img {
        width: 100%;
        display: block;
    }

    .image-container figcaption {
        text-align: center;
        font-size: 14px;
        color: rgb(117, 117, 117);
        margin-top: 5px;
    }

    .image-container figcaption:hover {
        color: rgb(186, 104, 200);
    }
</style>


<script>
    function filterFigures() {
        var selectedModelType = document.getElementById('modelTypeSelect').value;
        var selectedBoardType = document.getElementById('boardSelect').value;

        var figures = document.querySelectorAll('figure');

        figures.forEach(function(figure) {
            var matchesModelType = selectedModelType === 'Any' || figure.classList.contains(selectedModelType);
            var matchesBoardType = selectedBoardType === 'Any' || figure.classList.contains(selectedBoardType);

            if (matchesModelType && matchesBoardType) {
                figure.style.display = 'block';
            } else {
                figure.style.display = 'none';
            }
        });
    }

    document.getElementById('modelTypeSelect').addEventListener('change', filterFigures);
    document.getElementById('boardSelect').addEventListener('change', filterFigures);

    filterFigures();    // Call the function initially to apply any default filtering
</script>
