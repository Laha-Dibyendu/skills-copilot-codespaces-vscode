# XGBoost Large-Scale Training Methodology

## Global histogram training through sequential data partitions

## 1. Executive summary

This document describes a training methodology for large tabular
datasets using XGBoost's histogram-based tree learner.

The central idea is to **partition the dataset for data movement and
memory management, while preserving one global boosting process**. Every
boosting iteration must account for the full training population (or an
explicitly defined sample of it) before a tree split is selected.

This is fundamentally different from training a model on batch 1,
appending more trees using batch 2, and repeating. Continuing from an
existing booster adds trees; it does not normally revisit earlier
observations when constructing those new trees.

The preferred implementation is XGBoost's external-memory histogram
workflow:

-   **CPU:** iterator-backed external-memory data with
    `tree_method="hist"`.
-   **GPU:** `ExtMemQuantileDMatrix` with `tree_method="hist"` and
    `device="cuda"` in a compatible XGBoost release.
-   **Distributed:** partition data across workers and aggregate local
    histogram statistics at each boosting iteration.

The single-device external-memory approach and distributed training can
implement the same global histogram objective. They differ in execution
schedule, hardware parallelism, communication, and possible numerical
details.

------------------------------------------------------------------------

## 2. Problem formulation

Let the complete training dataset be

\[ `\mathcal{D}`{=tex} = {(x_i,y_i)}\_{i=1}\^{N} \]

where:

-   \(N\) is the total number of observations;
-   (x_i) is the feature vector;
-   (y_i) is the target;
-   the objective is binary classification.

At boosting iteration (t), the current ensemble (F\_{t-1}) produces a
prediction for each training row. For a differentiable objective,
XGBoost calculates the first- and second-order derivatives:

\[ g_i = `\frac{\partial \ell(y_i,F_{t-1}(x_i))}`{=tex}
{`\partial `{=tex}F\_{t-1}(x_i)} \]

\[ h_i = `\frac{\partial^2 \ell(y_i,F_{t-1}(x_i))}`{=tex}
{`\partial `{=tex}F\_{t-1}(x_i)\^2} \]

The next tree is built using these gradients and Hessians.

The key requirement is that gradients and Hessians for all relevant
training observations are calculated against the **same current
ensemble** (F\_{t-1}). A data partition must not advance to a different
boosting state before the other partitions have contributed to the
current tree.

------------------------------------------------------------------------

## 3. Histogram-based split finding

XGBoost's `hist` tree method quantizes continuous feature values into a
finite number of bins. For each active tree node and feature, it
accumulates gradient and Hessian statistics in those bins.

For feature (j) and bin (b), define:

\[ G\_{j,b} = `\sum`{=tex}*{i `\in `{=tex}I*{j,b}} g_i `\qquad`{=tex}
H\_{j,b} = `\sum`{=tex}*{i `\in `{=tex}I*{j,b}} h_i \]

where (I\_{j,b}) contains the observations assigned to that node and
feature bin.

For a candidate split, prefix sums over the bins produce left-child and
right-child statistics:

\[ G_L = `\sum`{=tex}*{b `\leq `{=tex}s}G*{j,b}, `\qquad`{=tex} H_L =
`\sum`{=tex}*{b `\leq `{=tex}s}H*{j,b} \]

\[ G_R = G\_{`\text{parent}`{=tex}}-G_L, `\qquad`{=tex} H_R =
H\_{`\text{parent}`{=tex}}-H_L \]

A common regularized leaf-weight expression is:

\[ w\^\* = -`\frac{\operatorname{Threshold}(G,\alpha)}{H+\lambda}`{=tex}
\]

where (`\lambda`{=tex}) is L2 regularization and (`\alpha`{=tex}) is L1
regularization. XGBoost's split gain compares the objective improvement
from the candidate children against the parent, including configured
regularization and split penalties.

The important property for partitioned processing is that **gradient and
Hessian histogram statistics are additive**.

------------------------------------------------------------------------

## 4. Why sequential histogram accumulation is mathematically valid

Partition the dataset into (K) disjoint subsets:

\[ `\mathcal{D}`{=tex} =
`\bigcup`{=tex}\_{k=1}\^{K}`\mathcal{D}`{=tex}\_k \]

For a fixed boosting iteration, current model, tree node, feature and
bin, calculate a local histogram for each partition:

\[ G\_{j,b}\^{(k)} =
`\sum`{=tex}*{i`\in `{=tex}I*{j,b}`\cap`{=tex}`\mathcal{D}`{=tex}\_k}g_i
\]

\[ H\_{j,b}\^{(k)} =
`\sum`{=tex}*{i`\in `{=tex}I*{j,b}`\cap`{=tex}`\mathcal{D}`{=tex}\_k}h_i
\]

The global histograms are:

\[ G\_{j,b} = `\sum`{=tex}\_{k=1}^{K}G\_{j,b}^{(k)} \]

\[ H\_{j,b} = `\sum`{=tex}\_{k=1}^{K}H\_{j,b}^{(k)} \]

This follows directly from the additivity of sums over disjoint
partitions.

### Sequential single-device schedule

A single device can process the partitions one after another:

1.  Hold the current model (F\_{t-1}) fixed.
2.  Process partition 1 and accumulate its gradient/Hessian histograms.
3.  Process partition 2 and add its statistics to the same accumulators.
4.  Continue through all partitions.
5.  Select the global best split from the accumulated statistics.
6.  Continue tree construction using the global statistics.
7.  Complete tree (t), update the ensemble to (F_t), and start the next
    boosting iteration.

The accumulation buffer must represent the same tree iteration and
compatible node/bin definitions throughout the accumulation.

### Distributed schedule

With multiple workers, each worker calculates its local histogram
concurrently. Workers then sum the local histograms using an all-reduce
or equivalent aggregation operation:

\[ H\_{`\text{global}`{=tex}} = `\sum`{=tex}\_{k=1}^{K}H^{(k)} \]

The same applies to gradient histograms.

### Mathematical relationship

The distributed and sequential schedules implement the same global
histogram sum when they use:

-   the same observations and row weights;
-   the same current model and therefore the same gradients and
    Hessians;
-   compatible feature-bin definitions and missing-value treatment;
-   the same node membership and split rules;
-   the same objective, regularization and tree-growth configuration.

The order of floating-point additions can cause small numerical
differences. Quantile sketching, approximate cut construction,
categorical handling, sampling, and implementation-specific details can
also prevent bit-for-bit identity.

**Conclusion:** Sequential accumulation is mathematically equivalent to
distributed histogram aggregation at the level of the global histogram
statistics, subject to the conditions above. It is not distributed
computing: it has no concurrent workers and does not provide
multi-device compute throughput.

------------------------------------------------------------------------

## 5. Three approaches that must not be confused

### 5.1 Sequential incremental boosting on separate batches

Example pattern:

``` python
booster = xgb.train(params, dtrain_batch_1, num_boost_round=R1)

booster = xgb.train(
    params,
    dtrain_batch_2,
    num_boost_round=R2,
    xgb_model=booster
)
```

This continues boosting from the existing model by adding more trees.
The new trees are trained using the data supplied to the current
training call; earlier batches are not automatically replayed for every
new tree.

Consequences:

-   Tree construction is order-dependent.
-   Earlier batches do not necessarily contribute gradients to later
    trees.
-   It is not equivalent to one global training run over the union of
    all batches.
-   It can be useful for deliberate staged learning or certain update
    workflows, but it is not the preferred method for reproducing
    full-dataset training.

### 5.2 QuantileDMatrix built from an iterator

An iterator can be used to construct a `QuantileDMatrix` from batches.
The resulting quantized dataset is retained as the training matrix.

This is useful when the compressed representation fits in the relevant
memory. It is not the same as an external-memory matrix that fetches
cached batches on demand during training.

### 5.3 External-memory global histogram training

An iterator supplies data partitions to an external-memory matrix.
XGBoost manages the cache and data access required by the histogram
learner.

This preserves one global boosting process. Each tree is constructed
using the training dataset represented by the external-memory matrix,
rather than appending a separate set of trees for each input partition.

This is the recommended approach when the objective is to train one
model over a dataset that is too large to keep fully resident in the
primary compute memory.

------------------------------------------------------------------------

## 6. Recommended CPU methodology

### Architecture

1.  Store the dataset in reusable, sequentially readable partitions.
2.  Implement an `xgboost.DataIter` with `next()` and `reset()`.
3.  Supply batches to an external-memory `DMatrix` or
    `ExtMemQuantileDMatrix`, according to the selected XGBoost version
    and API.
4.  Train one booster with `tree_method="hist"`.
5.  Let XGBoost iterate over the external-memory cache during tree
    construction.

### Conceptual configuration

``` python
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cpu",
    "max_bin": 256,
    "grow_policy": "depthwise",
}
```

### CPU-specific considerations

-   Use fast local NVMe for disk-backed cache; external-memory training
    can become I/O-bound.
-   Use sufficiently large batches to amortize iterator and I/O
    overhead. Avoid tiny batches.
-   `grow_policy="depthwise"` is a sensible starting point for
    external-memory workloads because it can reduce repeated full data
    passes relative to node-at-a-time growth.
-   Benchmark thread count, batch size, storage throughput and cache
    behaviour.
-   Keep validation data separate and use the training matrix as the
    reference when constructing a compatible quantile validation matrix.

The CPU method is conceptually straightforward: partitions are fetched
from external storage/cache as the tree-building algorithm needs them.
It trades memory residency for repeated data access.

------------------------------------------------------------------------

## 7. Recommended GPU methodology

### Architecture

For a compatible XGBoost release, use:

-   `tree_method="hist"`;
-   `device="cuda"`;
-   `ExtMemQuantileDMatrix`;
-   a custom iterator that yields GPU-backed batches;
-   host memory as the external cache by default, where supported;
-   an asynchronous CUDA memory pool where supported and recommended by
    the installed XGBoost release.

Conceptual configuration:

``` python
params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",
    "max_bin": 256,
    "grow_policy": "depthwise",
}
```

### GPU-specific considerations
 
-   The iterator input batches for GPU external-memory training must be
    GPU-backed.
-   The external cache can be staged in host memory; disk-backed cache
    is possible but can impose substantial I/O costs.
-   Use the supported asynchronous memory-pool mechanism for the
    installed release.
-   Benchmark input batch size and cache settings. Batch size is a
    throughput and memory-management parameter, not a change to the
    intended global training objective.
-   Monitor device memory, host memory, data-transfer time and time per
    boosting iteration.
-   Use a validation matrix constructed with the training matrix as its
    reference so that the quantization setup is compatible.

### Version considerations

External-memory GPU training has changed significantly across XGBoost
releases. The GPU implementation in XGBoost 2.x should not be assumed to
behave like the newer external-memory implementation.

For a new implementation, use a current stable XGBoost release whose
documentation explicitly supports the intended GPU external-memory
workflow. Confirm the exact API and CUDA memory-pool requirements
against that release's documentation before deployment.

------------------------------------------------------------------------

## 8. CPU, GPU and distributed comparison

  -----------------------------------------------------------------------
  Dimension         CPU external      Single-GPU        Multi-GPU
                    memory            external memory   distributed
  ----------------- ----------------- ----------------- -----------------
  Data partitioning Yes               Yes               Yes

  One global        Yes               Yes               Yes
  boosting process                                      

  Global histogram  Yes               Yes               Yes
  statistics                                            

  Partition         Repeated          GPU batches with  Concurrent
  processing        cache/data passes external cache    workers

  Histogram         Local             Local             Cross-worker
  aggregation       accumulation      accumulation      reduction

  Compute           CPU cores         One GPU           Multiple
  parallelism                                           GPUs/workers

  Main data         Storage/cache to  Host/cache to GPU Worker-local data
  movement          CPU                                 and inter-worker
                                                        communication

  Distributed       No                No                Yes
  execution                                             

  Closest to        Yes, subject to   Yes, subject to   Yes, subject to
  full-data         same algorithm    same algorithm    same algorithm
  histogram         settings          settings          settings
  objective                                             
  -----------------------------------------------------------------------

The three approaches can target the same global histogram objective.
They are not guaranteed to produce byte-identical models because of
floating-point reduction order, quantization details, sampling and
implementation differences.

------------------------------------------------------------------------

## 9. Quantization and equivalence conditions

Histogram additivity alone is not sufficient to guarantee equivalence.
The partitions must agree on how feature values map to bins.

For exact equivalence of the histogram statistics at a given iteration,
all partitions must use compatible global cut points:

\[ q_j(x_i) = b \]

must assign a given feature value to the same bin regardless of which
partition contains the row.

If each batch independently constructs its own quantile cuts, bin 17 in
one batch may represent a different value range from bin 17 in another.
Adding those histograms would then be invalid.

Therefore:

-   Use XGBoost's supported global quantile/external-memory matrix
    construction rather than independently quantizing each batch.
-   Use the training matrix as the reference when constructing
    validation data.
-   Keep feature order, types, categorical dictionaries and
    missing-value semantics consistent across partitions.
-   Treat exact equivalence as an algorithmic claim under matched cuts
    and settings, not as a promise of bit-for-bit reproducibility.

------------------------------------------------------------------------

## 10. Categorical and missing-value handling

For mixed numerical and categorical data:

-   Use a consistent feature schema across all partitions.
-   Preserve categorical columns as categorical features when using
    XGBoost's categorical support.
-   Ensure category encodings/dictionaries are consistent across
    partitions.
-   Do not independently assign category codes per batch.
-   Keep missing values represented consistently.
-   Validate categorical support in the exact XGBoost release and
    external-memory mode being deployed.

For numerical features, quantile binning and histogram aggregation are
naturally compatible with the partitioned methodology when global cuts
are used.

------------------------------------------------------------------------

## 11. Experimental validation plan

Before a full-scale run, validate the methodology against an in-memory
baseline on a dataset small enough to fit.

### Experiment A: In-memory reference

Train one model on the complete subset using `hist`, fixed parameters
and a fixed seed.

### Experiment B: External-memory single-device

Train on exactly the same observations using the iterator-backed
external-memory matrix.

### Experiment C: Distributed reference

If a multi-worker environment is available, train using row-partitioned
distributed histogram training with the same feature schema and
compatible settings.

### Compare

-   validation AUC and log loss;
-   predictions on a fixed evaluation set;
-   tree structures and split features;
-   number of boosting rounds;
-   training time and time per iteration;
-   peak CPU RAM and GPU memory;
-   data-loading and transfer overhead.

Do not require exact model-file equality. Compare prediction agreement,
metric differences and split behaviour within a defined tolerance.
Investigate material differences in quantile cuts, sampling, categorical
encoding, missing-value handling, seeds and floating-point reduction.

------------------------------------------------------------------------

## 12. Final recommendation

Use **one global XGBoost histogram-training process over an
external-memory dataset**.

-   For CPU execution, use iterator-backed external-memory training with
    `hist`.
-   For GPU execution, use the supported `ExtMemQuantileDMatrix`
    workflow with GPU-backed iterator batches.
-   For multi-GPU execution, use distributed row partitioning and
    histogram aggregation.
-   Do not use sequential `xgb_model` continuation across disjoint
    batches when the goal is equivalent full-dataset training.

The core principle is:

> Partition the data for storage and computation, but do not partition
> the boosting history. At each boosting iteration, all partitions must
> contribute to the same global tree-building decision.

This is the most defensible way to describe the methodology and its
relationship to distributed histogram training.

------------------------------------------------------------------------

## 13. References

1.  [XGBoost --- Using External
    Memory](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html)
2.  [XGBoost --- GPU
    Support](https://xgboost.readthedocs.io/en/stable/gpu/)
3.  [XGBoost --- Distributed
    XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/dask.html)
4.  [XGBoost --- Python API:
    ExtMemQuantileDMatrix](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
5.  [XGBoost --- Distributed
    Training](https://xgboost.readthedocs.io/en/stable/tutorials/distributed.html)

*Implementation note: external-memory APIs and GPU cache behaviour are
version-sensitive. Pin the XGBoost version and verify the matching
release documentation before treating code or performance
characteristics as production guarantees.*
