"""Generic utilities matching :mod:`kmodes.util`."""

from __future__ import annotations

import numpy as np


def pandas_to_numpy(x):
    return x.values if "pandas" in str(x.__class__) else x


def get_max_value_key(dic):
    values = np.array(list(dic.values()))
    keys = np.array(list(dic.keys()))
    maxima = np.where(values == np.max(values))[0]
    if len(maxima) == 1:
        return keys[maxima[0]]
    return keys[maxima[np.argmin(keys[maxima])]]


def _encode_features(X, enc_map, dtype):
    X = np.asarray(X)
    fit = enc_map is None
    if fit:
        enc_map = []
    encoded = np.empty(X.shape, dtype=dtype)
    for column in range(X.shape[1]):
        if fit:
            values, inverse = np.unique(
                X[:, column], return_inverse=True
            )
            mapping = {
                value: index
                for index, value in enumerate(values)
            }
            enc_map.append(mapping)
            encoded[:, column] = inverse
            continue
        encoded[:, column] = np.fromiter(
            (enc_map[column].get(value, -1) for value in X[:, column]),
            dtype=dtype,
            count=X.shape[0],
        )
    return encoded, enc_map


def encode_features(X, enc_map=None):
    return _encode_features(X, enc_map, np.int32)


def _encode_features_i64(X, enc_map=None):
    array = np.asarray(X)
    if (
        enc_map is not None
        and array.ndim == 2
        and array.dtype.kind == "U"
        and array.flags.c_contiguous
    ):
        from .._lib import addr, lib

        flattened_keys = [
            key for mapping in enc_map for key in mapping
        ]
        key_width = max(
            1, max(len(str(key)) for key in flattened_keys)
        )
        keys = np.asarray(
            flattened_keys, dtype=f"U{key_width}"
        )
        offsets = np.empty(len(enc_map) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(
            [len(mapping) for mapping in enc_map],
            out=offsets[1:],
        )
        codes = np.fromiter(
            (
                code
                for mapping in enc_map
                for code in mapping.values()
            ),
            dtype=np.int64,
            count=len(flattened_keys),
        )
        encoded = np.empty(array.shape, dtype=np.int64)
        lib().mkm_encode_unicode(
            addr(array.view(np.uint32)),
            addr(keys.view(np.uint32)),
            addr(offsets),
            addr(codes),
            addr(encoded),
            array.shape[0],
            array.shape[1],
            array.dtype.itemsize // np.dtype("U1").itemsize,
            key_width,
        )
        return encoded, enc_map
    return _encode_features(X, enc_map, np.int64)


def decode_centroids(encoded, mapping):
    decoded = []
    for column in range(encoded.shape[1]):
        inverse = {value: key for key, value in mapping[column].items()}
        decoded.append(
            np.vectorize(inverse.__getitem__)(encoded[:, column])
        )
    return np.atleast_2d(np.array(decoded)).T


def get_unique_rows(array):
    return np.vstack(list({tuple(row) for row in array}))


def category_layout(encoded):
    cardinalities = np.max(encoded, axis=0).astype(np.int64) + 1
    offsets = np.empty(encoded.shape[1] + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(cardinalities, out=offsets[1:])
    return offsets
