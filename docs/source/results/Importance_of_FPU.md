# Importance of FPU

This study aims to reveal the importance of the Floating Point Unit (FPU) in TinyML applications.

We use FC and CNN models deployed on the NUCLEO-L4R5ZI board for our experiments.

<br/>

Model Type:
<select id="modelTypeSelect">
    <option value="Any">Any</option>
    <option value="FC">FC</option>
    <option value="CNN">CNN</option>
</select>

## Models

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/error.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/dark/error.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/error.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/dark/error.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/exe.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/dark/exe.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/exe.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/dark/exe.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/flash.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/dark/flash.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/flash.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/dark/flash.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/ram.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - FC - STM (FPU)/dark/ram.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/ram.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/FPU/TFLM - CNN - STM (FPU)/dark/ram.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Summary

The only effect of the FPU is on the execution time of the *basic* models which is very significant. Still, if using *int8 only*, utilizing the *CMSIS-NN* library can be even more beneficial and there is no need for the FPU.


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
