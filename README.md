# mojo-kmodes

`mojo-kmodes` is a Mojo port of the compute-heavy parts of
[`kmodes`](https://github.com/nicodv/kmodes), with Python estimators for
k-modes clustering of categorical data and k-prototypes clustering of mixed
numerical/categorical data.

The Python API mirrors upstream's estimator names, constructor signatures,
methods, and fitted attributes under the `mojo_kmodes` package. Categorical
encoding and Python-facing validation stay in Python; assignment, cost,
Cao initialization, frequency updates, modes, and numerical prototype updates
run in one compiled Mojo shared library.

## Coverage

| Area | Covered |
| --- | --- |
| Estimators | `KModes`, `KPrototypes` |
| Initializers | `Cao`, `Huang`, `random`, explicit centroids |
| Methods | `fit`, `fit_predict`, `predict`, `get_params`, `set_params` |
| Attributes | `cluster_centroids_`, `labels_`, `cost_`, `n_iter_`, `epoch_costs_`, `gamma` |
| Functional API | `k_modes`, `k_prototypes`, both `labels_cost` functions |
| Utilities | feature encode/decode, unique rows, Cao/Huang initialization |
| Dissimilarities | matching, squared Euclidean, binary/label Jaccard, Ng |
| Other behavior | sample weights, unknown categories at prediction, NumPy inputs |

The port is tested directly against `kmodes 0.12.2`, which is installed in the
Pixi environment from conda-forge. The parity suite checks labels, decoded
centroids, costs, gamma, iteration counts, and per-epoch costs on the same
inputs and initial random states.

Not covered:

- custom dissimilarity functions during fitting; the functional
  `labels_cost` APIs do accept them through a Python fallback
- parallel `n_jobs` execution; the parameter is accepted for compatibility,
  but restarts currently execute sequentially
- sparse matrices, which upstream also rejects
- bit-for-bit reproduction of upstream's random empty-cluster recovery on
  degenerate data; this port moves the first point from the largest cluster
  instead

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/kmodes.mojo` to
`dist/libmojo-kmodes.so`. Importing the Python package also rebuilds the
library automatically if the Mojo source is newer.

## Usage

K-modes:

```python
import numpy as np
from mojo_kmodes.kmodes import KModes

X = np.array([
    ["red", "small"],
    ["red", "medium"],
    ["blue", "large"],
    ["blue", "large"],
])

model = KModes(n_clusters=2, init="Cao", random_state=0).fit(X)
print(model.cluster_centroids_)
print(model.predict([["red", "large"]]))
```

K-prototypes:

```python
import numpy as np
from mojo_kmodes.kprototypes import KPrototypes

X = np.array([
    [22.0, 35_000.0, "student"],
    [25.0, 42_000.0, "student"],
    [51.0, 95_000.0, "professional"],
    [48.0, 88_000.0, "professional"],
], dtype=object)

model = KPrototypes(
    n_clusters=2, init="Cao", n_init=1, random_state=0
).fit(X, categorical=[2])

labels = model.predict(X, categorical=[2])
print(labels, model.cost_, model.gamma)
```

Run either example inside the environment with `pixi run python`.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Times are the best of three runs, except fitting
cases, which use the best of two. Both implementations receive identical
arrays and initialization.

| case | mojo-kmodes | kmodes 0.12.2 | speedup |
| --- | ---: | ---: | ---: |
| matching_dissim (1M x 16) | 13.3 ms | 65.5 ms | 4.92x |
| KModes.predict (200k x 12, k=8) | 248.6 ms | 4421.7 ms | 17.79x |
| KModes.fit (50k x 10, k=8) | 722.8 ms | 5676.9 ms | 7.85x |
| KPrototypes.predict (120k, 4 num + 6 cat, k=8) | 334.6 ms | 2977.3 ms | 8.90x |
| KPrototypes.fit (30k, 3 num + 5 cat, k=6) | 198.1 ms | 3471.7 ms | 17.53x |

The largest gains come from replacing upstream's per-row Python assignment
loops with one FFI call over the full contiguous dataset. Each timed estimator
call includes its input conversion, encoding, and output allocation. Dataset
generation, estimator construction, and warm-up calls are outside the timing.

No GPU path is included. The distance kernels perform substantially less than
two arithmetic operations per byte moved, while categorical encoding is
comparison- and branch-heavy. Neither has enough arithmetic intensity to
offset device transfer and launch costs.

## How it works

The Mojo code is a single compilation unit with C ABI exports. Python calls it
through `ctypes`; arrays cross the boundary as integer addresses plus explicit
shape values. Python owns every input, output, and scratch allocation, so the
Mojo library does not allocate memory or retain pointers.

Numerical columns are contiguous row-major `float64`. Categorical columns are
encoded per feature to contiguous row-major `int64`, with unknown prediction
values represented by `-1`. Cluster category frequencies use a dense
`float64` table indexed by cluster and per-column category offsets. This
layout makes mode changes constant-time and uses
`O(k * sum(column cardinalities))` scratch space instead of a Python
dictionary per cluster and feature.

The k-modes loop maintains memberships and weighted category frequencies as
points move. K-prototypes adds weighted numerical sums and combines squared
Euclidean cost with `gamma * matching_cost`. Convergence follows upstream:
stop when no points move or the epoch cost no longer decreases.

## License

MIT
