# TFLM vs Renesas eAI Translator

The goal of this study is to compare the performance of TensorFlow Lite for Microcontrollers (TFLM) and Renesas eAI Translator.

According to the eAI Translator's documentation, it should be able to convert RNN models. However, we faced some errors during the conversion (1). As a result, this study focuses on FC, CNN, and TinyMLPerf models. Also, eAI Translator is designed to work with Renesas boards, so we only have the RenesasRX65N board in our study.
{ .annotate }

1.  :man_raising_hand: Still, we guess it should be possible to convert RNN models with some modifications and under certain conditions.

<br/>

Model Type:
<select id="modelTypeSelect">
    <option value="Any">Any</option>
    <option value="FC">FC</option>
    <option value="CNN">CNN</option>
    <option value="TinyMLPerf">TinyMLPerf</option>
</select>

## Models

<div class="image-container">
<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/params_MACs.png#only-light" alt="TinyMLPerf parameters and MACs">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/dark/params_MACs.png#only-dark" alt="TinyMLPerf parameters and MACs">
    <figcaption>TinyMLPerf parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/error.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/dark/error.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/error.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/dark/error.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/error.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/dark/error.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/exe.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/dark/exe.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/exe.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/dark/exe.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/exe.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/dark/exe.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/flash.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/dark/flash.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/flash.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/dark/flash.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/flash.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/dark/flash.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/ram.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - FC - Renesas/dark/ram.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/ram.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - CNN - Renesas/dark/ram.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/ram.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs eAI Translator/TFLM vs eAI Translator - TinyMLPerf - Renesas/dark/ram.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**:
    - Some models failed to run on the board. (1)
    { .annotate }

        1.  :man_raising_hand: For the failed FC models, the program halts for an unknown reason. For the failed TinyMLPerf models, the program is too large for the RenesasRX65N board.

    - The error rate of the TFLM and eAI Translator models are normally the same, but for some big models, the eAI Translator has an unacceptable error. (1)
    { .annotate }

        1.  :man_raising_hand: Namely, *CNN_5*, *CNN_6*, "CNN_7", "TinyMLPerf_MBNet", and "TinyMLPerf_ResNet" models.

- **Execution Time**:
    - Please note that we have not utilized *CMSIS-NN* with TFLM. As a result, the *int8 only* version of TFLM could potentially yield better results. Thus, we have excluded the *int8 only* variants from our execution time comparisons.

    - eAI Translator is usually better. (1)
    { .annotate }

        1.  :man_raising_hand: eAI Translator is better especially for smaller models. For bigger ones the two are almost the same, or TFLM might even get a bit better.

- **Flash Size**: eAI Translator is better.

- **RAM Usage**: eAI Translator is slightly better.

- **Conclusion**: If the error of the eAI Translator is acceptable, it is a better choice than TFLM for RenesasRX65N.


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
