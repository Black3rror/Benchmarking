# TFLM Pruning and Clustering

This study explores the impact of pruning and clustering on the performance of neural networks.

FC and CNN models were evaluated on the NUCLEO-L4R5ZI board, using 50% sparsity for pruning and clustering with 16 centroids.

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
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/params_MACs.png#only-light" alt="FC parameters and MACs">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/dark/params_MACs.png#only-dark" alt="FC parameters and MACs">
    <figcaption>FC parameters and MACs</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/params_MACs.png#only-light" alt="CNN parameters and MACs">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/dark/params_MACs.png#only-dark" alt="CNN parameters and MACs">
    <figcaption>CNN parameters and MACs</figcaption>
</figure>
</div>

## Error

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/error.png#only-light" alt="FC error">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/dark/error.png#only-dark" alt="FC error">
    <figcaption>FC error</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/error.png#only-light" alt="CNN error">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/dark/error.png#only-dark" alt="CNN error">
    <figcaption>CNN error</figcaption>
</figure>
</div>

## Execution Time

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/exe.png#only-light" alt="FC execution time">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/dark/exe.png#only-dark" alt="FC execution time">
    <figcaption>FC execution time</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/exe.png#only-light" alt="CNN execution time">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/dark/exe.png#only-dark" alt="CNN execution time">
    <figcaption>CNN execution time</figcaption>
</figure>
</div>

## Flash Size

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/flash.png#only-light" alt="FC flash size">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/dark/flash.png#only-dark" alt="FC flash size">
    <figcaption>FC flash size</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/flash.png#only-light" alt="CNN flash size">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/dark/flash.png#only-dark" alt="CNN flash size">
    <figcaption>CNN flash size</figcaption>
</figure>
</div>

## RAM Usage

<div class="image-container">
<figure markdown="span" class="FC">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/ram.png#only-light" alt="FC RAM usage">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - FC - STM/dark/ram.png#only-dark" alt="FC RAM usage">
    <figcaption>FC RAM usage</figcaption>
</figure>

<figure markdown="span" class="CNN">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/ram.png#only-light" alt="CNN RAM usage">
    <img src="../../figures/results/Pruning and Clustering/Prune Cluster - CNN - STM/dark/ram.png#only-dark" alt="CNN RAM usage">
    <figcaption>CNN RAM usage</figcaption>
</figure>
</div>

## Summary

Despite the popularity of pruning and clustering in reducing the size of neural networks, our results demonstrate that these techniques don't improve model performance and, in fact, they increase the error rate. This happens because the pruned or clustered weights still need to be stored in memory (occupying the same space as the original weights), and operations involving these weights still need to be executed (e.g., `x * 0` takes just as long as `x * y`).

One solution is to use structured pruning, which essentially involves designing a new model architecture—such as removing specific neurons or channels. Alternatively, you could use a hardware accelerator optimized for sparse weights or clustering. Another option is to “unfold” the matrix multiplication and eliminate unnecessary operations, though this approach requires a large amount of flash memory, making it impractical for most applications.

In conclusion, we advise against using unstructured pruning or clustering unless you have access to a hardware accelerator that specifically supports these techniques.


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
