# GCC vs CCRX

In this study, we look for the performance differences between a publicly available compiler (GCC) and a proprietary industrial compiler (CCRX) for the Renesas RX65N board.

FC and CNN models converted by Renesas eAI Translator were used in our experiments.

!!! info "No Flash Size or RAM Usage"
    Since we could not obtain the flash size and RAM usage for the CCRX compiler, we have omitted these metrics from the study.

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
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/error.png#only-light" alt="FC - Renesas RX65N">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/dark/error.png#only-dark" alt="FC - Renesas RX65N">
    <figcaption>FC - Renesas RX65N</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/error.png#only-light" alt="CNN - Renesas RX65N">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/dark/error.png#only-dark" alt="CNN - Renesas RX65N">
    <figcaption>CNN - Renesas RX65N</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/exe.png#only-light" alt="FC - Renesas RX65N">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - FC - Renesas (GCC vs CCRX)/dark/exe.png#only-dark" alt="FC - Renesas RX65N">
    <figcaption>FC - Renesas RX65N</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/exe.png#only-light" alt="CNN - Renesas RX65N">
    <img src="../../figures/results/GCC vs CCRX/eAI Translator - CNN - Renesas (GCC vs CCRX)/dark/exe.png#only-dark" alt="CNN - Renesas RX65N">
    <figcaption>CNN - Renesas RX65N</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**: As a result of using the eAI Translator, we have seen that some models fail to output correct results using GCC. The same applies to CCRX, and even the *int8 only* version of *CNN_4* which previously had acceptable error, now has a high error rate. Other than these cases, the two have the same error rates.

- **Execution Time**: The execution time of GCC is better than CCRX for all models.

- **Conclusion**: GCC is a bit more reliable than CCRX and executes the models faster.


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
