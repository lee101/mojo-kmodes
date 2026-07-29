"""Dissimilarity measures compatible with :mod:`kmodes.util.dissim`."""

from __future__ import annotations

import numpy as np

from .._lib import addr, f64, i64, lib


def matching_dissim(a, b, **_):
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if (
        a_array.ndim == 2
        and b_array.ndim == 1
        and a_array.shape[1] == b_array.shape[0]
        and a_array.dtype.kind in "biu"
        and b_array.dtype.kind in "biu"
    ):
        ai = i64(a_array)
        bi = i64(b_array)
        result = np.empty(ai.shape[0], dtype=np.float64)
        lib().mkm_matching_dissim(
            addr(ai), addr(bi), addr(result), ai.shape[0], ai.shape[1]
        )
        return result
    return np.sum(a_array != b_array, axis=1)


def euclidean_dissim(a, b, **_):
    af = f64(a)
    bf = f64(b)
    if np.isnan(af).any() or np.isnan(bf).any():
        raise ValueError("Missing values detected in numerical columns.")
    result = np.empty(af.shape[0], dtype=np.float64)
    lib().mkm_euclidean_dissim(
        addr(af), addr(bf), addr(result), af.shape[0], af.shape[1]
    )
    return result


def jaccard_dissim_binary(a, b, **__):
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if ((a_array == 0) | (a_array == 1)).all() and (
        (b_array == 0) | (b_array == 1)
    ).all():
        numerator = np.sum(np.bitwise_and(a_array, b_array), axis=1)
        denominator = np.sum(np.bitwise_or(a_array, b_array), axis=1)
        if (denominator == 0).any():
            raise ValueError("Insufficient Number of data since union is 0")
        return 1 - numerator / denominator
    raise ValueError("Missing or non Binary values detected in Binary columns.")


def jaccard_dissim_label(a, b, **__):
    a_array = np.asarray(a)
    b_array = np.asarray(b)
    if np.isnan(a_array.astype("float64")).any() or np.isnan(
        b_array.astype("float64")
    ).any():
        raise ValueError("Missing values detected in Numeric columns.")
    intersect = np.empty(len(a_array), dtype=int)
    union = np.empty(len(a_array), dtype=int)
    for index, row in enumerate(a_array):
        intersect[index] = len(np.intersect1d(row, b_array))
        union[index] = (
            len(np.unique(row))
            + len(np.unique(b_array))
            - intersect[index]
        )
    if (union == 0).any():
        raise ValueError("Insufficient Number of data since union is 0")
    return 1 - intersect / union


def ng_dissim(a, b, X=None, membship=None):
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(value, data, membership, attribute):
        indices = np.where(membership == 1)
        return float(
            (np.take(data, indices, axis=0)[0][:, attribute] == value[attribute])
            .sum(0)
        )

    def calc_dissim(value, data, membership, attribute):
        size = float(np.sum(membership))
        return (
            1.0 - calc_cjr(value, data, membership, attribute) / size
            if size != 0.0
            else 0.0
        )

    if len(membship) != np.asarray(a).shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError(
            "'membship' must be a rectangular array where the number of rows "
            "in 'membship' equals the number of rows in 'a' and the number of "
            "columns in 'membship' equals the number of rows in 'X'."
        )
    return np.array(
        [
            np.array(
                [
                    calc_dissim(b, X, membship[cluster], attribute)
                    if b[attribute] == category
                    else 1.0
                    for attribute, category in enumerate(center)
                ]
            ).sum(0)
            for cluster, center in enumerate(a)
        ]
    )
