# Compiler Optimization Levels

In this study, we aim to find out the impact of varying compiler optimization levels (*O3*, *Of*, and *Os*) on the performance of TFLM models.

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
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/error.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/dark/error.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/error.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/dark/error.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/exe.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/dark/exe.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/exe.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/dark/exe.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/flash.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/dark/flash.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/flash.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/dark/flash.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/ram.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - FC - STM (opts)/dark/ram.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/ram.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/Optimization levels/TFLM - CNN - STM (opts)/dark/ram.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**: The optimization levels do not change the correctness of the models.

- **Execution Time**: The *O3* and *Of* are the same and faster than *Os*.

- **Flash Size**: The flash size of *Os* is slightly better than *O3* and *Of*. (1)
{ .annotate }

    1.  :man_raising_hand: The difference is in the program's base size (without the model) and vanishes as the model size increases.

- **RAM Usage**: The required RAM of all optimization levels is almost the same.

- **Conclusion**: The *O3* and *Of* optimization levels are similar and better than *Os* in terms of execution time. If the flash size is a concern, *Os* might be slightly better than *O3* and *Of*.


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
