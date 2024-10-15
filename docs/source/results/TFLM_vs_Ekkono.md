# TFLM vs Ekkono

The goal of this study is to compare the performance of TensorFlow Lite for Microcontrollers (TFLM) and Ekkono.

Ekkono is only able to run FC models, so we have limited our comparison to FC models only.

!!! warning "Slightly Modified Models"
    Since Ekkono is not able to do classification, we have slightly changed some models to make them suitable for regression. The changes are minimal and should not have a noticeable impact on the results.

<br/>

Board:
<select id="boardSelect">
    <option value="Any">Any</option>
    <option value="STM">NUCLEO-L4R5ZI</option>
    <option value="Renesas">RenesasRX65N</option>
</select>

## Models

<div class="image-container">
<figure markdown="span" class="FC STM Renesas">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/error.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/dark/error.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/error.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/dark/error.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/exe.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/dark/exe.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/exe.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/dark/exe.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/flash.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/dark/flash.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/flash.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/dark/flash.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC STM">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/ram.png#only-light" alt="FC - NUCLEO-L4R5ZI">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - STM/dark/ram.png#only-dark" alt="FC - NUCLEO-L4R5ZI">
    <figcaption>FC - NUCLEO-L4R5ZI</figcaption>
</figure>

<figure markdown="span" class="FC Renesas">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/ram.png#only-light" alt="FC - RenesasRX65N">
    <img src="../../figures/results/TFLM vs Ekkono/TFLM vs Ekkono - FC - Renesas/dark/ram.png#only-dark" alt="FC - RenesasRX65N">
    <figcaption>FC - RenesasRX65N</figcaption>
</figure>
</div>

## Summary

- **Model Correctness**:
    - Some models failed to run on the RenesasRX65N board. (1)
    { .annotate }

        1.  :man_raising_hand: For *FC_1* and *FC_2* models, the program halts for an unknown reason.

    - Ekkono and TFLM *basic* are perfect. TFLM *int8 only* has a bit of error which is acceptable.

- **Execution Time**: For small models, Ekkono is faster than TFLM. However, as the model size increases, TFLM becomes faster.

- **Flash Size**: For small models, Ekkono has a smaller flash size. However, as the model size increases, TFLM *int8 only* becomes more efficient. (1)
{ .annotate }

    1.  :man_raising_hand: The Ekkono library itself has a smaller footprint than the TFLM library. As the model grows, the TFLM library's overhead becomes negligible compared to the model size. In case of TFLM *int8 only*, the model size grows 1/4th compared to TFLM *basic* or Ekkono which results in a smaller flash size.

- **RAM Usage**: For small models, Ekkono requires a smaller RAM. However, as the model size increases, TFLM *int8 only* becomes more efficient. (1)
{ .annotate }

    1.  :man_raising_hand: Same story as the flash size.

- **Conclusion**: For small models, Ekkono is more efficient. However, as the model size increases, TFLM *int8 only* becomes more efficient.


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
        var selectedBoardType = document.getElementById('boardSelect').value;
        var figures = document.querySelectorAll('figure');

        figures.forEach(function(figure) {
            var matchesBoardType = selectedBoardType === 'Any' || figure.classList.contains(selectedBoardType);

            if (matchesBoardType) {
                figure.style.display = 'block';
            } else {
                figure.style.display = 'none';
            }
        });
    }

    document.getElementById('boardSelect').addEventListener('change', filterFigures);

    filterFigures();    // Call the function initially to apply any default filtering
</script>
