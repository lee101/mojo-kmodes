"""K-modes clustering for categorical data."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from ._lib import addr, f64, i64, lib
from .util import (
    _encode_features_i64,
    category_layout,
    decode_centroids,
    encode_features,
    get_unique_rows,
    pandas_to_numpy,
)
from .util.dissim import matching_dissim, ng_dissim
from .util.init_methods import init_cao, init_huang


class KModes(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters=8,
        max_iter=100,
        cat_dissim=matching_dissim,
        init="Cao",
        n_init=10,
        verbose=0,
        random_state=None,
        n_jobs=1,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cat_dissim = cat_dissim
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        if (
            (
                isinstance(self.init, str)
                and self.init == "Cao"
            )
            or hasattr(self.init, "__array__")
        ) and self.n_init > 1:
            if self.verbose:
                print(
                    "Initialization method and algorithm are deterministic. "
                    "Setting n_init to 1."
                )
            self.n_init = 1

    def fit(self, X, y=None, sample_weight=None, **kwargs):
        X = pandas_to_numpy(X)
        random_state = check_random_state(self.random_state)
        _validate_sample_weight(
            sample_weight,
            n_samples=X.shape[0],
            n_clusters=self.n_clusters,
        )
        (
            self._enc_cluster_centroids,
            self._enc_map,
            self.labels_,
            self.cost_,
            self.n_iter_,
            self.epoch_costs_,
        ) = k_modes(
            X,
            self.n_clusters,
            self.max_iter,
            self.cat_dissim,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.n_jobs,
            sample_weight,
        )
        return self

    def fit_predict(self, X, y=None, **kwargs):
        return self.fit(X, **kwargs).predict(X, **kwargs)

    def predict(self, X, **kwargs):
        assert hasattr(self, "_enc_cluster_centroids"), "Model not yet fitted."
        if self.verbose and self.cat_dissim == ng_dissim:
            print(
                "Ng's dissimilarity measure was used to train this model, "
                "but now that it is predicting the model will fall back to "
                "using simple matching dissimilarity."
            )
        X = check_array(pandas_to_numpy(X), dtype=None)
        if _default_dissim(self.cat_dissim):
            encoded, _ = _encode_features_i64(X, enc_map=self._enc_map)
            modes = i64(self._enc_cluster_centroids)
            labels = np.empty(encoded.shape[0], dtype=np.uint16)
            lib().mkm_mode_labels(
                addr(encoded),
                addr(modes),
                addr(labels),
                encoded.shape[0],
                encoded.shape[1],
                modes.shape[0],
            )
            return labels
        encoded, _ = encode_features(X, enc_map=self._enc_map)
        return labels_cost(encoded, self._enc_cluster_centroids, self.cat_dissim)[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, "_enc_cluster_centroids"):
            return decode_centroids(
                self._enc_cluster_centroids, self._enc_map
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute "
            "'cluster_centroids_' because the model is not yet fitted."
        )


def _default_dissim(dissim) -> bool:
    return dissim is matching_dissim or (
        getattr(dissim, "__name__", None) == "matching_dissim"
        and getattr(dissim, "__module__", "").endswith("util.dissim")
    )


def _weights(sample_weight, n_points):
    if sample_weight is None:
        return np.ones(n_points, dtype=np.float64)
    _validate_sample_weight(sample_weight, n_points, 1)
    return f64(sample_weight)


def labels_cost(X, centroids, dissim, membship=None, sample_weight=None):
    X = check_array(X)
    centroids = check_array(centroids)
    if X.shape[1] != centroids.shape[1]:
        raise ValueError(
            "X and centroids must have the same number of attributes."
        )
    n_points = X.shape[0]
    weights = _weights(sample_weight, n_points)
    if _default_dissim(dissim):
        encoded = i64(X)
        modes = i64(centroids)
        labels = np.empty(n_points, dtype=np.int64)
        cost = lib().mkm_mode_labels_cost(
            addr(encoded),
            addr(modes),
            addr(weights),
            addr(labels),
            n_points,
            encoded.shape[1],
            modes.shape[0],
        )
        return labels.astype(np.uint16), float(cost)

    labels = np.empty(n_points, dtype=np.uint16)
    cost = 0.0
    for row, point in enumerate(X):
        distances = dissim(
            centroids, point, X=X, membship=membship
        )
        cluster = np.argmin(distances)
        labels[row] = cluster
        cost += distances[cluster] * weights[row]
    return labels, float(cost)


def _single(
    X,
    n_clusters,
    max_iter,
    dissim,
    init,
    init_no,
    verbose,
    random_state,
    sample_weight,
):
    rng = check_random_state(random_state)
    n_points, n_attrs = X.shape
    if isinstance(init, str) and init.lower() == "huang":
        centroids = init_huang(X, n_clusters, dissim, rng)
    elif isinstance(init, str) and init.lower() == "cao":
        centroids = init_cao(X, n_clusters, dissim)
    elif isinstance(init, str) and init.lower() == "random":
        seeds = rng.choice(range(n_points), n_clusters)
        centroids = X[seeds]
    elif hasattr(init, "__array__"):
        candidate = init
        if len(candidate.shape) == 1:
            candidate = np.atleast_2d(candidate).T
        assert candidate.shape[0] == n_clusters, (
            f"Wrong number of initial centroids in init "
            f"({candidate.shape[0]}, should be {n_clusters})."
        )
        assert candidate.shape[1] == n_attrs, (
            f"Wrong number of attributes in init "
            f"({candidate.shape[1]}, should be {n_attrs})."
        )
        centroids = candidate
    else:
        raise NotImplementedError

    encoded = i64(X)
    modes = i64(centroids, copy=True)
    weights = _weights(sample_weight, n_points)
    offsets = category_layout(encoded)
    total_categories = int(offsets[-1])
    membership = np.empty(n_points, dtype=np.int64)
    labels = np.empty(n_points, dtype=np.int64)
    counts = np.empty(
        n_clusters * total_categories, dtype=np.float64
    )
    cluster_sizes = np.empty(n_clusters, dtype=np.int64)
    epoch_costs = np.empty(max_iter + 1, dtype=np.float64)
    iterations = lib().mkm_kmodes_fit(
        addr(encoded),
        addr(modes),
        addr(weights),
        addr(membership),
        addr(labels),
        addr(counts),
        addr(cluster_sizes),
        addr(offsets),
        addr(epoch_costs),
        n_points,
        n_attrs,
        n_clusters,
        total_categories,
        max_iter,
    )
    costs = epoch_costs[: iterations + 1].tolist()
    if verbose:
        print(
            f"Run {init_no + 1}, iterations: {iterations}/{max_iter}, "
            f"cost: {costs[-1]}"
        )
    return (
        modes,
        labels.astype(np.uint16),
        float(costs[-1]),
        int(iterations),
        costs,
    )


def k_modes(
    X,
    n_clusters,
    max_iter,
    dissim,
    init,
    n_init,
    verbose,
    random_state,
    n_jobs,
    sample_weight=None,
):
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-modes does not support sparse data.")
    X = check_array(X, dtype=None)
    if not _default_dissim(dissim):
        raise NotImplementedError(
            "Mojo training currently supports matching_dissim only."
        )
    encoded, enc_map = _encode_features_i64(X)
    n_points = encoded.shape[0]
    _validate_cluster_count(n_clusters, n_points)
    unique = get_unique_rows(encoded)
    if unique.shape[0] <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = unique.shape[0]
        init = unique

    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    results = [
        _single(
            encoded,
            n_clusters,
            max_iter,
            dissim,
            init,
            run,
            verbose,
            seeds[run],
            sample_weight,
        )
        for run in range(n_init)
    ]
    best = int(np.argmin([result[2] for result in results]))
    if n_init > 1 and verbose:
        print(f"Best run was number {best + 1}")
    centroids, labels, cost, iterations, costs = results[best]
    return centroids, enc_map, labels, cost, iterations, costs


def _validate_sample_weight(sample_weight, n_samples, n_clusters):
    if sample_weight is None:
        return
    if len(sample_weight) != n_samples:
        raise ValueError("sample_weight should be of equal size as samples.")
    if any(
        not isinstance(weight, (int, float, np.integer, np.floating))
        for weight in sample_weight
    ):
        raise ValueError(
            "sample_weight elements should either be int or floats."
        )
    if any(sample < 0 for sample in sample_weight):
        raise ValueError("sample_weight elements should be positive.")
    if not np.isfinite(np.asarray(sample_weight, dtype=np.float64)).all():
        raise ValueError("sample_weight elements should be finite.")
    if sum(value > 0 for value in sample_weight) < n_clusters:
        raise ValueError(
            "Number of non-zero sample_weight elements should be larger "
            "than the number of clusters."
        )


def _validate_cluster_count(n_clusters, n_points):
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > np.iinfo(np.uint16).max + 1:
        raise ValueError("n_clusters exceeds the uint16 label capacity.")
    if n_clusters > n_points:
        raise ValueError(
            f"Cannot have more clusters ({n_clusters}) than data points "
            f"({n_points})."
        )
