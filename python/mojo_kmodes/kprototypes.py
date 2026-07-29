"""K-prototypes clustering for mixed numerical and categorical data."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array

from . import kmodes
from ._lib import addr, f64, i64, lib
from .util import (
    _encode_features_i64,
    category_layout,
    decode_centroids,
    encode_features,
    get_unique_rows,
    pandas_to_numpy,
)
from .util.dissim import euclidean_dissim, matching_dissim
from .util.init_methods import init_cao, init_huang

MAX_INIT_TRIES = 20
RAISE_INIT_TRIES = 100


class KPrototypes(kmodes.KModes):
    def __init__(
        self,
        n_clusters=8,
        max_iter=100,
        num_dissim=euclidean_dissim,
        cat_dissim=matching_dissim,
        init="Cao",
        n_init=10,
        gamma=None,
        verbose=0,
        random_state=None,
        n_jobs=1,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_dissim = num_dissim
        self.cat_dissim = cat_dissim
        self.init = init
        self.n_init = n_init
        self.gamma = gamma
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        if isinstance(self.init, list) and self.n_init > 1:
            if self.verbose:
                print(
                    "Initialization method is deterministic. "
                    "Setting n_init to 1."
                )
            self.n_init = 1

    def fit(
        self, X, y=None, categorical=None, sample_weight=None
    ):
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), (
                "The 'categorical' argument needs to be an integer with the "
                "index of the categorical column in your data, or a list or "
                f"tuple of several of them, but it is a {type(categorical)}."
            )
        X = pandas_to_numpy(X)
        random_state = check_random_state(self.random_state)
        kmodes._validate_sample_weight(
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
            self.gamma,
        ) = k_prototypes(
            X,
            categorical,
            self.n_clusters,
            self.max_iter,
            self.num_dissim,
            self.cat_dissim,
            self.gamma,
            self.init,
            self.n_init,
            self.verbose,
            random_state,
            self.n_jobs,
            sample_weight,
        )
        return self

    def predict(self, X, categorical=None, **kwargs):
        assert hasattr(self, "_enc_cluster_centroids"), "Model not yet fitted."
        if categorical is not None:
            assert isinstance(categorical, (int, list, tuple)), (
                "The 'categorical' argument needs to be an integer with the "
                "index of the categorical column in your data, or a list or "
                f"tuple of several of them, but it is a {type(categorical)}."
            )
        Xnum, Xcat = _split_num_cat(
            pandas_to_numpy(X), categorical
        )
        Xnum = check_array(Xnum)
        Xcat = check_array(Xcat, dtype=None)
        Xcat, _ = _encode_features_i64(Xcat, enc_map=self._enc_map)
        return labels_cost(
            Xnum,
            Xcat,
            self._enc_cluster_centroids,
            self.num_dissim,
            self.cat_dissim,
            self.gamma,
        )[0]

    @property
    def cluster_centroids_(self):
        if hasattr(self, "_enc_cluster_centroids"):
            return np.hstack(
                (
                    self._enc_cluster_centroids[0],
                    decode_centroids(
                        self._enc_cluster_centroids[1], self._enc_map
                    ),
                )
            )
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute "
            "'cluster_centroids_' because the model is not yet fitted."
        )


def _default_num(dissim):
    return dissim is euclidean_dissim or (
        getattr(dissim, "__name__", None) == "euclidean_dissim"
        and getattr(dissim, "__module__", "").endswith("util.dissim")
    )


def labels_cost(
    Xnum,
    Xcat,
    centroids,
    num_dissim,
    cat_dissim,
    gamma,
    membship=None,
    sample_weight=None,
):
    Xnum = check_array(Xnum)
    Xcat = check_array(Xcat)
    centers_num = check_array(centroids[0])
    centers_cat = check_array(centroids[1])
    if Xnum.shape[0] != Xcat.shape[0]:
        raise ValueError("Xnum and Xcat must have the same number of rows.")
    if Xnum.shape[1] != centers_num.shape[1]:
        raise ValueError(
            "Xnum and numerical centroids must have matching attributes."
        )
    if Xcat.shape[1] != centers_cat.shape[1]:
        raise ValueError(
            "Xcat and categorical centroids must have matching attributes."
        )
    if centers_num.shape[0] != centers_cat.shape[0]:
        raise ValueError(
            "Numerical and categorical centroids must have matching rows."
        )
    n_points = Xnum.shape[0]
    weights = kmodes._weights(sample_weight, n_points)
    if _default_num(num_dissim) and kmodes._default_dissim(cat_dissim):
        numeric = f64(Xnum)
        categorical = i64(Xcat)
        centers_num = f64(centers_num)
        centers_cat = i64(centers_cat)
        labels = np.empty(n_points, dtype=np.int64)
        cost = lib().mkm_prototype_labels_cost(
            addr(numeric),
            addr(categorical),
            addr(centers_num),
            addr(centers_cat),
            addr(weights),
            addr(labels),
            n_points,
            numeric.shape[1],
            categorical.shape[1],
            centers_num.shape[0],
            float(gamma),
        )
        return labels.astype(np.uint16), float(cost)

    labels = np.empty(n_points, dtype=np.uint16)
    cost = 0.0
    for row in range(n_points):
        numeric_costs = num_dissim(centers_num, Xnum[row])
        categorical_costs = cat_dissim(
            centers_cat,
            Xcat[row],
            X=Xcat,
            membship=membship,
        )
        total = numeric_costs + gamma * categorical_costs
        cluster = np.argmin(total)
        labels[row] = cluster
        cost += total[cluster] * weights[row]
    return labels, float(cost)


def _single(
    Xnum,
    Xcat,
    n_clusters,
    max_iter,
    num_dissim,
    cat_dissim,
    gamma,
    init,
    init_no,
    verbose,
    random_state,
    sample_weight,
):
    rng = check_random_state(random_state)
    n_points, nnumattrs = Xnum.shape
    ncatattrs = Xcat.shape[1]
    weights = kmodes._weights(sample_weight, n_points)
    offsets = category_layout(Xcat)
    total_categories = int(offsets[-1])
    membership = np.empty(n_points, dtype=np.int64)
    labels = np.empty(n_points, dtype=np.int64)
    counts = np.empty(
        n_clusters * total_categories, dtype=np.float64
    )
    sums = np.empty(n_clusters * nnumattrs, dtype=np.float64)
    member_weights = np.empty(n_clusters, dtype=np.float64)
    cluster_sizes = np.empty(n_clusters, dtype=np.int64)
    epoch_costs = np.empty(max_iter + 1, dtype=np.float64)

    active_init = init
    for attempt in range(1, RAISE_INIT_TRIES + 1):
        if isinstance(active_init, str) and active_init.lower() == "huang":
            centers_cat = init_huang(
                Xcat, n_clusters, cat_dissim, rng
            )
        elif isinstance(active_init, str) and active_init.lower() == "cao":
            centers_cat = init_cao(Xcat, n_clusters, cat_dissim)
        elif isinstance(active_init, str) and active_init.lower() == "random":
            seeds = rng.choice(range(n_points), n_clusters)
            centers_cat = Xcat[seeds]
        elif isinstance(active_init, list):
            prepared = [
                np.atleast_2d(value).T
                if len(value.shape) == 1
                else value
                for value in active_init
            ]
            assert prepared[0].shape == (n_clusters, nnumattrs), (
                "Wrong shape for initial numerical centroids."
            )
            assert prepared[1].shape == (n_clusters, ncatattrs), (
                "Wrong shape for initial categorical centroids."
            )
            centers_num = f64(prepared[0], copy=True)
            centers_cat = i64(prepared[1], copy=True)
        else:
            raise NotImplementedError(
                "Initialization method not supported."
            )

        if not isinstance(active_init, list):
            centers_num = (
                np.mean(Xnum, axis=0)
                + rng.randn(n_clusters, nnumattrs)
                * np.std(Xnum, axis=0)
            )
            centers_num = f64(centers_num)
            centers_cat = i64(centers_cat, copy=True)

        iterations = lib().mkm_kprototypes_fit(
            addr(Xnum),
            addr(Xcat),
            addr(centers_num),
            addr(centers_cat),
            addr(weights),
            addr(membership),
            addr(labels),
            addr(counts),
            addr(sums),
            addr(member_weights),
            addr(cluster_sizes),
            addr(offsets),
            addr(epoch_costs),
            n_points,
            nnumattrs,
            ncatattrs,
            n_clusters,
            total_categories,
            max_iter,
            float(gamma),
        )
        if iterations >= 0:
            break
        if attempt == MAX_INIT_TRIES:
            active_init = "random"
    else:
        raise ValueError(
            "Clustering algorithm could not initialize. Consider assigning "
            "the initial clusters manually."
        )

    costs = epoch_costs[: iterations + 1].tolist()
    if verbose:
        print(
            f"Run: {init_no + 1}, iterations: {iterations}/{max_iter}, "
            f"cost: {costs[-1]}"
        )
    return (
        [centers_num, centers_cat],
        labels.astype(np.uint16),
        float(costs[-1]),
        int(iterations),
        costs,
    )


def k_prototypes(
    X,
    categorical,
    n_clusters,
    max_iter,
    num_dissim,
    cat_dissim,
    gamma,
    init,
    n_init,
    verbose,
    random_state,
    n_jobs,
    sample_weight=None,
):
    random_state = check_random_state(random_state)
    if sparse.issparse(X):
        raise TypeError("k-prototypes does not support sparse data.")
    if categorical is None or not categorical:
        raise NotImplementedError(
            "No categorical data selected, effectively doing k-means. "
            "Present a list of categorical columns, or use scikit-learn's "
            "KMeans instead."
        )
    if isinstance(categorical, int):
        categorical = [categorical]
    assert len(categorical) != X.shape[1], (
        "All columns are categorical, use k-modes instead of k-prototypes."
    )
    assert max(categorical) < X.shape[1], (
        "Categorical index larger than number of columns."
    )
    if not _default_num(num_dissim) or not kmodes._default_dissim(cat_dissim):
        raise NotImplementedError(
            "Mojo training currently supports euclidean_dissim plus "
            "matching_dissim only."
        )

    Xnum, Xcat = _split_num_cat(X, categorical)
    Xnum = f64(check_array(Xnum))
    Xcat = check_array(Xcat, dtype=None)
    Xcat, enc_map = _encode_features_i64(Xcat)
    n_points = Xnum.shape[0]
    kmodes._validate_cluster_count(n_clusters, n_points)

    unique = get_unique_rows(np.asarray(X))
    if unique.shape[0] <= n_clusters:
        max_iter = 0
        n_init = 1
        n_clusters = unique.shape[0]
        unique_num, unique_cat = _split_num_cat(unique, categorical)
        unique_cat, _ = _encode_features_i64(unique_cat, enc_map)
        init = [f64(unique_num), unique_cat]

    if gamma is None:
        gamma = 0.5 * np.mean(Xnum.std(axis=0))

    seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
    results = [
        _single(
            Xnum,
            Xcat,
            n_clusters,
            max_iter,
            num_dissim,
            cat_dissim,
            gamma,
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
    return centroids, enc_map, labels, cost, iterations, costs, gamma


def _split_num_cat(X, categorical):
    numeric = np.asanyarray(
        X[
            :,
            [
                index
                for index in range(X.shape[1])
                if index not in categorical
            ],
        ]
    ).astype(np.float64)
    categories = np.asanyarray(X[:, categorical])
    return numeric, categories
