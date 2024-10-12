# TFLM vs Edge Impulse

The goal of this study is to compare the performance of TensorFlow Lite for Microcontrollers (TFLM) and Edge Impulse.

According to the Edge Impulse documentation, an enterprise account is needed to run RNN models. As a result, this study focuses on FC, CNN, and TinyMLPerf models.

<br/>

Model Type:
<select id="modelTypeSelect">
    <option value="Any">Any</option>
    <option value="FC">FC</option>
    <option value="CNN">CNN</option>
    <option value="TinyMLPerf">TinyMLPerf</option>
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
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN STM Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf STM Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/params_MACs.png#only-light" alt="TinyMLPerf parameters and MACs">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/dark/params_MACs.png#only-dark" alt="TinyMLPerf parameters and MACs">
    <figcaption>TinyMLPerf parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/error.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/dark/error.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/error.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/dark/error.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/error.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/dark/error.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/error.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/dark/error.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/error.png#only-light" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/dark/error.png#only-dark" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <figcaption>TinyMLPerf - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/error.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/dark/error.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/exe.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/dark/exe.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/exe.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/dark/exe.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/exe.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/dark/exe.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/exe.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/dark/exe.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/exe.png#only-light" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/dark/exe.png#only-dark" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <figcaption>TinyMLPerf - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/exe.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/dark/exe.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/flash.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/dark/flash.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/flash.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/dark/flash.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/flash.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/dark/flash.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/flash.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/dark/flash.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/flash.png#only-light" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/dark/flash.png#only-dark" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <figcaption>TinyMLPerf - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/flash.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/dark/flash.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/ram.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - STM/dark/ram.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/ram.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - FC - Renesas/dark/ram.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="CNN STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/ram.png#only-light" alt="CNN - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - STM/dark/ram.png#only-dark" alt="CNN - NUCLEO-L4R5ZI">
    <figcaption>CNN - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="CNN Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/ram.png#only-light" alt="CNN - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - CNN - Renesas/dark/ram.png#only-dark" alt="CNN - RenesasRX65N">
    <figcaption>CNN - RenesasRX65N</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf STM">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/ram.png#only-light" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - STM/dark/ram.png#only-dark" alt="TinyMLPerf - NUCLEO-L4R5ZI">
    <figcaption>TinyMLPerf - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="TinyMLPerf Renesas">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/ram.png#only-light" alt="TinyMLPerf - RenesasRX65N">
    <img src="../../figures/results/TFLM vs EI/TFLM vs EI - TinyMLPerf - Renesas/dark/ram.png#only-dark" alt="TinyMLPerf - RenesasRX65N">
    <figcaption>TinyMLPerf - RenesasRX65N</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**:
    - Since *TinyMLPerf_MBNet* was too large for all tests except its *int8 only* version with TFLM, we have excluded it from the comparison.

    - Some models failed to run on the RenesasRX65N board. (1)
    { .annotate }

        1.  :man_raising_hand: In all cases, the program halts for an unknown reason.

    - The error rates of the remaining models are acceptable. (1)
    { .annotate }

        1.  :man_raising_hand: *basic* is perfect, Edge Impulse is better than TFLM in *int8 only* models.

- **RenesasRX65N**:
    - On the Renesas board, TFLM outperforms Edge Impulse in terms of execution time, flash size, and RAM usage (1). As a result, TFLM is recommended for this board.
    { .annotate }

        1.  :man_raising_hand: One contributing factor could be that the GCC compiler had link-time optimization (`-flto`) enabled for TFLM, but this was not possible for Edge Impulse (Edge Impulse encountered errors with `-flto`). However, this cannot be the only reason, because in a few tested scenarios where we have turned off `-flto` for TFLM, the results were still better than Edge Impulse.

- **NUCLEO-L4R5ZI**:
    - **Execution Time**: TFLM performed better for FC models, particularly for smaller models. For other models, the performance of both is similar, with a slight edge for TFLM.

    - **Flash Size**: Edge Impulse is better than TFLM.

    - **RAM Usage**: For small FC models, Edge Impulse is slightly better. For the others, the two are almost the same.

- **Conclusion**:
    - For the RenesasRX65N board, TFLM is the preferable choice

    - For the NUCLEO-L4R5ZI board, TFLM is recommended if execution time is the priority. However, if flash size and RAM usage are more critical, Edge Impulse may be the better option.


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
