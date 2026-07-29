"""Huang and Cao initialization methods."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .._lib import addr, i64, lib
from . import category_layout


def init_huang(X, n_clusters, dissim, random_state):
    n_attrs = X.shape[1]
    centroids = np.empty((n_clusters, n_attrs), dtype="object")
    for attribute in range(n_attrs):
        choices = sorted(X[:, attribute])
        centroids[:, attribute] = random_state.choice(choices, n_clusters)
    for cluster in range(n_clusters):
        indices = np.argsort(dissim(X, centroids[cluster]))
        while (
            np.all(X[indices[0]] == centroids, axis=1).any()
            and indices.shape[0] > 1
        ):
            indices = np.delete(indices, 0)
        centroids[cluster] = X[indices[0]]
    return centroids


def init_cao(X, n_clusters, dissim):
    array = np.asarray(X)
    if (
        array.dtype.kind in "biu"
        and array.ndim == 2
        and np.min(array) >= 0
    ):
        encoded = i64(array)
        offsets = category_layout(encoded)
        total = int(offsets[-1])
        centroids = np.empty((n_clusters, encoded.shape[1]), dtype=np.int64)
        density = np.empty(encoded.shape[0], dtype=np.float64)
        frequencies = np.empty(total, dtype=np.float64)
        lib().mkm_cao_init(
            addr(encoded),
            addr(centroids),
            addr(density),
            addr(frequencies),
            addr(offsets),
            encoded.shape[0],
            encoded.shape[1],
            n_clusters,
            total,
        )
        return centroids

    n_points, n_attrs = array.shape
    centroids = np.empty((n_clusters, n_attrs), dtype="object")
    density = np.zeros(n_points)
    for attribute in range(n_attrs):
        frequency = defaultdict(int)
        for value in array[:, attribute]:
            frequency[value] += 1
        for row in range(n_points):
            density[row] += (
                frequency[array[row, attribute]]
                / float(n_points)
                / float(n_attrs)
            )
    centroids[0] = array[np.argmax(density)]
    for cluster in range(1, n_clusters):
        weighted = np.empty((cluster, n_points))
        for previous in range(cluster):
            weighted[previous] = (
                dissim(array, centroids[previous]) * density
            )
        centroids[cluster] = array[np.argmax(np.min(weighted, axis=0))]
    return centroids
