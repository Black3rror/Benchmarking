# STM vs Renesas

In this study, we are interested in comparing the performance of two similar boards from different manufacturers: the NUCLEO-L4R5ZI from STMicroelectronics and the Renesas RX65N.

!!! warning "Hobby Experiment"
    This study is a hobby experiment and should not be considered as a professional benchmark.

All four types of models were tested on both boards.

<br/>

Model Type:
<select id="modelTypeSelect">
    <option value="Any">Any</option>
    <option value="FC">FC</option>
    <option value="CNN">CNN</option>
    <option value="RNN">RNN</option>
    <option value="TinyMLPerf">TinyMLPerf</option>
</select>

## Models

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="RNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/params_MACs.png#only-light" alt="RNN parameters and MACs">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/dark/params_MACs.png#only-dark" alt="RNN parameters and MACs">
    <figcaption>RNN parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/params_MACs.png#only-light" alt="TinyMLPerf parameters and MACs">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/dark/params_MACs.png#only-dark" alt="TinyMLPerf parameters and MACs">
    <figcaption>TinyMLPerf parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/error.png#only-light" alt="FC error">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/dark/error.png#only-dark" alt="FC error">
    <figcaption>FC error</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/error.png#only-light" alt="CNN error">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/dark/error.png#only-dark" alt="CNN error">
    <figcaption>CNN error</figcaption>
</figure>

<figure markdown="span" class="RNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/error.png#only-light" alt="RNN error">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/dark/error.png#only-dark" alt="RNN error">
    <figcaption>RNN error</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/error.png#only-light" alt="TinyMLPerf error">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/dark/error.png#only-dark" alt="TinyMLPerf error">
    <figcaption>TinyMLPerf error</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/exe.png#only-light" alt="FC execution time">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/dark/exe.png#only-dark" alt="FC execution time">
    <figcaption>FC execution time</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/exe.png#only-light" alt="CNN execution time">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/dark/exe.png#only-dark" alt="CNN execution time">
    <figcaption>CNN execution time</figcaption>
</figure>

<figure markdown="span" class="RNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/exe.png#only-light" alt="RNN execution time">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/dark/exe.png#only-dark" alt="RNN execution time">
    <figcaption>RNN execution time</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/exe.png#only-light" alt="TinyMLPerf execution time">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/dark/exe.png#only-dark" alt="TinyMLPerf execution time">
    <figcaption>TinyMLPerf execution time</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/flash.png#only-light" alt="FC flash size">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/dark/flash.png#only-dark" alt="FC flash size">
    <figcaption>FC flash size</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/flash.png#only-light" alt="CNN flash size">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/dark/flash.png#only-dark" alt="CNN flash size">
    <figcaption>CNN flash size</figcaption>
</figure>

<figure markdown="span" class="RNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/flash.png#only-light" alt="RNN flash size">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/dark/flash.png#only-dark" alt="RNN flash size">
    <figcaption>RNN flash size</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/flash.png#only-light" alt="TinyMLPerf flash size">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/dark/flash.png#only-dark" alt="TinyMLPerf flash size">
    <figcaption>TinyMLPerf flash size</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/ram.png#only-light" alt="FC RAM usage">
    <img src="../../figures/results/STM vs Renesas/TFLM - FC - STM vs Renesas/dark/ram.png#only-dark" alt="FC RAM usage">
    <figcaption>FC RAM usage</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/ram.png#only-light" alt="CNN RAM usage">
    <img src="../../figures/results/STM vs Renesas/TFLM - CNN - STM vs Renesas/dark/ram.png#only-dark" alt="CNN RAM usage">
    <figcaption>CNN RAM usage</figcaption>
</figure>

<figure markdown="span" class="RNN">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/ram.png#only-light" alt="RNN RAM usage">
    <img src="../../figures/results/STM vs Renesas/TFLM - RNN - STM vs Renesas/dark/ram.png#only-dark" alt="RNN RAM usage">
    <figcaption>RNN RAM usage</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/ram.png#only-light" alt="TinyMLPerf RAM usage">
    <img src="../../figures/results/STM vs Renesas/TFLM - TinyMLPerf - STM vs Renesas/dark/ram.png#only-dark" alt="TinyMLPerf RAM usage">
    <figcaption>TinyMLPerf RAM usage</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**: The Renesas board fails to run some of the models. Other than that, the two boards provide similar results. (1)
{ .annotate }

    1.  :man_raising_hand: Except for the *int8 only* version of *TinyMLPerf_MBNet* model.

- **Execution Time**:
    - The Renesas board did not utilize the *CMSIS-NN* library, so we should exclude the *int8 only* versions from the comparison.

    - The Renesas board is just slightly faster than the STM board.

- **Flash Size**: STM is a bit better in terms of flash size.

- **RAM Usage**: The two boards use almost the same amount of RAM.

- **Conclusion**: The two boards seem to have a relatively similar performance. The Renesas board might be slightly faster, but the STM board has a bit smaller flash size. (1)
{ .annotate }

    1.  :man_raising_hand: These boards have many settings that might favor one over the other in certain cases. Our study was conducted in an almost default settings with a good optimization level, but it is not comprehensive enough to cover all possible scenarios.


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
        var figures = document.querySelectorAll('figure');

        figures.forEach(function(figure) {
            var matchesModelType = selectedModelType === 'Any' || figure.classList.contains(selectedModelType);

            if (matchesModelType) {
                figure.style.display = 'block';
            } else {
                figure.style.display = 'none';
            }
        });
    }

    document.getElementById('modelTypeSelect').addEventListener('change', filterFigures);

    filterFigures();    // Call the function initially to apply any default filtering
</script>
