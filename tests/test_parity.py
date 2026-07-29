"""Behavioral parity with kmodes 0.12.2."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.utils import check_random_state

import mojo_kmodes as mojo
from mojo_kmodes import kmodes as mk
from mojo_kmodes import kprototypes as mkp
from mojo_kmodes.util import (
    _encode_features_i64,
    decode_centroids,
    encode_features,
    get_max_value_key,
)
from mojo_kmodes.util import dissim as md
from mojo_kmodes.util import init_methods as mi

from kmodes import kmodes as upstream_kmodes
from kmodes import kprototypes as upstream_kprototypes
from kmodes.util import (
    decode_centroids as upstream_decode,
    encode_features as upstream_encode,
    get_max_value_key as upstream_max_key,
)
from kmodes.util import dissim as ud
from kmodes.util import init_methods as ui


@pytest.fixture(scope="module")
def categorical_data():
    rng = np.random.RandomState(42)
    return np.vstack(
        [
            rng.randint(0, 2, (80, 6)),
            rng.randint(2, 4, (80, 6)),
            rng.randint(4, 6, (80, 6)),
        ]
    ).astype(str)


@pytest.fixture(scope="module")
def mixed_data():
    rng = np.random.RandomState(13)
    numeric = np.vstack(
        [
            rng.normal(0, 0.3, (40, 3)),
            rng.normal(5, 0.3, (40, 3)),
            rng.normal(10, 0.3, (40, 3)),
        ]
    )
    categories = np.vstack(
        [
            np.tile(["a", "x"], (40, 1)),
            np.tile(["b", "y"], (40, 1)),
            np.tile(["c", "z"], (40, 1)),
        ]
    )
    result = np.empty((120, 5), dtype=object)
    result[:, :3] = numeric
    result[:, 3:] = categories
    return result


def test_public_signatures_match():
    for actual, expected in (
        (mojo.KModes, upstream_kmodes.KModes),
        (mojo.KPrototypes, upstream_kprototypes.KPrototypes),
    ):
        actual_params = inspect.signature(actual).parameters
        expected_params = inspect.signature(expected).parameters
        assert list(actual_params) == list(expected_params)
        for name in actual_params:
            actual_default = actual_params[name].default
            expected_default = expected_params[name].default
            if callable(actual_default):
                assert actual_default.__name__ == expected_default.__name__
            else:
                assert actual_default == expected_default
    assert inspect.signature(mojo.KModes.fit) == inspect.signature(
        upstream_kmodes.KModes.fit
    )
    assert inspect.signature(mojo.KPrototypes.fit) == inspect.signature(
        upstream_kprototypes.KPrototypes.fit
    )


def test_estimator_parameter_protocol():
    model = mojo.KModes(n_clusters=3, random_state=7)
    assert model.get_params()["n_clusters"] == 3
    assert model.set_params(max_iter=4) is model
    assert model.max_iter == 4


def test_encode_features_parity():
    X = np.array(
        [["red", "S"], ["blue", "M"], ["red", "L"]], dtype=object
    )
    actual, actual_map = encode_features(X)
    expected, expected_map = upstream_encode(X)
    assert np.array_equal(actual, expected)
    assert actual_map == expected_map


def test_encode_unknown_category_parity():
    train = np.array([["a"], ["b"]])
    query = np.array([["a"], ["missing"]])
    _, mapping = encode_features(train)
    actual, _ = encode_features(query, mapping)
    expected, _ = upstream_encode(query, mapping)
    assert np.array_equal(actual, expected)
    assert actual[:, 0].tolist() == [0, -1]


def test_unicode_encoder_does_not_truncate_mapping_keys():
    _, mapping = encode_features(np.array([["long"], ["x"]]))
    query = np.array([["l"], ["x"]], dtype="U1")
    actual, _ = _encode_features_i64(query, mapping)
    assert actual[:, 0].tolist() == [-1, 1]


def test_decode_centroids_parity():
    X = np.array([["red", "S"], ["blue", "M"], ["green", "L"]])
    encoded, mapping = encode_features(X)
    assert np.array_equal(
        decode_centroids(encoded, mapping),
        upstream_decode(encoded, mapping),
    )


def test_max_value_key_tie_parity():
    values = {4: 2.0, 1: 2.0, 9: 1.0}
    assert get_max_value_key(values) == upstream_max_key(values) == 1


@pytest.mark.parametrize(
    "a,b",
    [
        (
            np.array([[0, 1, 2], [0, 2, 2], [3, 1, 0]]),
            np.array([0, 1, 2]),
        ),
        (
            np.array([["a", "b"], ["a", "c"]]),
            np.array(["a", "b"]),
        ),
    ],
)
def test_matching_dissim_parity(a, b):
    assert np.array_equal(md.matching_dissim(a, b), ud.matching_dissim(a, b))


def test_matching_dissim_simd_tail():
    rng = np.random.default_rng(18)
    a = rng.integers(0, 7, size=(19, 11), dtype=np.int64)
    b = rng.integers(0, 7, size=11, dtype=np.int64)
    assert np.array_equal(md.matching_dissim(a, b), ud.matching_dissim(a, b))


def test_matching_dissim_parallel_threshold():
    rng = np.random.default_rng(19)
    a = rng.integers(0, 7, size=(24_000, 11), dtype=np.int64)
    b = rng.integers(0, 7, size=11, dtype=np.int64)
    assert np.array_equal(md.matching_dissim(a, b), ud.matching_dissim(a, b))


def test_euclidean_dissim_parity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 7))
    b = rng.normal(size=7)
    assert np.allclose(
        md.euclidean_dissim(a, b), ud.euclidean_dissim(a, b)
    )


def test_euclidean_dissim_rejects_nan():
    with pytest.raises(ValueError, match="Missing values"):
        md.euclidean_dissim(np.array([[np.nan]]), np.array([0.0]))


def test_jaccard_binary_parity():
    a = np.array([[1, 0, 1], [1, 1, 0]])
    b = np.array([1, 0, 0])
    assert np.allclose(
        md.jaccard_dissim_binary(a, b),
        ud.jaccard_dissim_binary(a, b),
    )


def test_jaccard_label_parity():
    a = np.array([[1, 2, 3], [2, 3, 4]])
    b = np.array([1, 3, 5])
    assert np.allclose(
        md.jaccard_dissim_label(a, b),
        ud.jaccard_dissim_label(a, b),
    )


def test_ng_dissim_parity():
    X = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    centers = np.array([[0, 0], [1, 1]])
    membership = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    point = np.array([0, 1])
    assert np.allclose(
        md.ng_dissim(centers, point, X=X, membship=membership),
        ud.ng_dissim(centers, point, X=X, membship=membership),
    )


def test_cao_initialization_parity():
    X = np.array(
        [[0, 0], [0, 1], [0, 1], [1, 2], [1, 2], [2, 2]],
        dtype=np.int64,
    )
    assert np.array_equal(
        mi.init_cao(X, 3, md.matching_dissim),
        ui.init_cao(X, 3, ud.matching_dissim).astype(np.int64),
    )


def test_huang_initialization_parity():
    X = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [2, 1]])
    actual = mi.init_huang(
        X, 2, md.matching_dissim, check_random_state(11)
    )
    expected = ui.init_huang(
        X, 2, ud.matching_dissim, check_random_state(11)
    )
    assert np.array_equal(actual, expected)


def test_kmodes_labels_cost_with_weights():
    X = np.array([[0, 0], [0, 1], [2, 2], [2, 1]])
    centers = np.array([[0, 0], [2, 2]])
    weights = [1.0, 3.0, 2.0, 4.0]
    actual = mk.labels_cost(
        X, centers, md.matching_dissim, sample_weight=weights
    )
    expected = upstream_kmodes.labels_cost(
        X, centers, ud.matching_dissim, sample_weight=weights
    )
    assert np.array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


@pytest.mark.parametrize("initialization", ["Cao", "Huang", "random"])
def test_kmodes_fit_parity(categorical_data, initialization):
    actual = mojo.KModes(
        n_clusters=3,
        init=initialization,
        n_init=1,
        max_iter=20,
        random_state=7,
    ).fit(categorical_data)
    expected = upstream_kmodes.KModes(
        n_clusters=3,
        init=initialization,
        n_init=1,
        max_iter=20,
        random_state=7,
    ).fit(categorical_data)
    assert np.array_equal(actual.cluster_centroids_, expected.cluster_centroids_)
    assert np.array_equal(actual.labels_, expected.labels_)
    assert actual.cost_ == expected.cost_
    assert actual.n_iter_ == expected.n_iter_
    assert np.allclose(actual.epoch_costs_, expected.epoch_costs_)


def test_kmodes_explicit_init_and_sample_weights():
    X = np.array([["a"], ["a"], ["b"], ["b"], ["c"], ["c"]])
    weights = [1.0, 3.0, 1.0, 2.0, 1.0, 4.0]
    init = np.array([[0], [2]])
    actual = mojo.KModes(
        2, init=init, n_init=1, random_state=0
    ).fit(X, sample_weight=weights)
    expected = upstream_kmodes.KModes(
        2, init=init, n_init=1, random_state=0
    ).fit(X, sample_weight=weights)
    assert np.array_equal(actual.cluster_centroids_, expected.cluster_centroids_)
    assert np.array_equal(actual.labels_, expected.labels_)
    assert actual.cost_ == expected.cost_


def test_kmodes_predict_unknown_category(categorical_data):
    actual = mojo.KModes(3, random_state=3).fit(categorical_data)
    expected = upstream_kmodes.KModes(3, random_state=3).fit(
        categorical_data
    )
    query = categorical_data[:5].copy()
    query[0, 0] = "unknown"
    assert np.array_equal(actual.predict(query), expected.predict(query))
    assert np.array_equal(
        actual.fit_predict(categorical_data), actual.labels_
    )


@pytest.mark.parametrize("n_points", [7, 50_000])
def test_kmodes_predict_parallel_threshold(categorical_data, n_points):
    actual = mojo.KModes(3, n_init=1, random_state=3).fit(categorical_data)
    expected = upstream_kmodes.KModes(
        3, n_init=1, random_state=3
    ).fit(categorical_data)
    query = np.resize(categorical_data, (n_points, categorical_data.shape[1]))
    assert np.array_equal(actual.predict(query), expected.predict(query))


def test_kmodes_function_parity(categorical_data):
    args = dict(
        X=categorical_data,
        n_clusters=3,
        max_iter=20,
        dissim=md.matching_dissim,
        init="Cao",
        n_init=1,
        verbose=0,
        random_state=check_random_state(5),
        n_jobs=1,
    )
    actual = mk.k_modes(**args)
    args["dissim"] = ud.matching_dissim
    args["random_state"] = check_random_state(5)
    expected = upstream_kmodes.k_modes(**args)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[2], expected[2])
    assert actual[3] == expected[3]


def test_sample_weight_validation():
    with pytest.raises(ValueError, match="equal size"):
        mojo.KModes(2).fit(np.array([["a"], ["b"]]), sample_weight=[1])
    with pytest.raises(ValueError, match="positive"):
        mojo.KModes(2).fit(
            np.array([["a"], ["b"]]), sample_weight=[1, -1]
        )


def test_kprototypes_labels_cost_with_weights():
    Xnum = np.array([[0.0], [0.2], [5.0], [5.2]])
    Xcat = np.array([[0], [0], [1], [1]])
    centers = [np.array([[0.1], [5.1]]), np.array([[0], [1]])]
    weights = [1.0, 2.0, 3.0, 4.0]
    actual = mkp.labels_cost(
        Xnum,
        Xcat,
        centers,
        md.euclidean_dissim,
        md.matching_dissim,
        1.5,
        sample_weight=weights,
    )
    expected = upstream_kprototypes.labels_cost(
        Xnum,
        Xcat,
        centers,
        ud.euclidean_dissim,
        ud.matching_dissim,
        1.5,
        sample_weight=weights,
    )
    assert np.array_equal(actual[0], expected[0])
    assert actual[1] == pytest.approx(expected[1])


@pytest.mark.parametrize("initialization", ["Cao", "Huang", "random"])
def test_kprototypes_fit_parity(mixed_data, initialization):
    actual = mojo.KPrototypes(
        n_clusters=3,
        init=initialization,
        n_init=1,
        max_iter=20,
        random_state=7,
    ).fit(mixed_data, categorical=[3, 4])
    expected = upstream_kprototypes.KPrototypes(
        n_clusters=3,
        init=initialization,
        n_init=1,
        max_iter=20,
        random_state=7,
    ).fit(mixed_data, categorical=[3, 4])
    assert np.array_equal(actual.labels_, expected.labels_)
    assert actual.cost_ == pytest.approx(expected.cost_, rel=1e-14)
    assert actual.n_iter_ == expected.n_iter_
    assert actual.gamma == expected.gamma
    assert np.all(actual.cluster_centroids_ == expected.cluster_centroids_)
    assert np.allclose(actual.epoch_costs_, expected.epoch_costs_)


def test_kprototypes_explicit_init_and_sample_weights():
    X = np.array(
        [
            [0.0, "a"],
            [0.1, "a"],
            [5.0, "b"],
            [5.2, "b"],
            [10.0, "c"],
            [10.1, "c"],
        ],
        dtype=object,
    )
    init = [np.array([[0.0], [10.0]]), np.array([[0], [2]])]
    weights = [1.0, 3.0, 1.0, 2.0, 1.0, 4.0]
    actual = mojo.KPrototypes(
        2, init=init, n_init=1, gamma=1.0, random_state=0
    ).fit(X, categorical=[1], sample_weight=weights)
    expected = upstream_kprototypes.KPrototypes(
        2, init=init, n_init=1, gamma=1.0, random_state=0
    ).fit(X, categorical=[1], sample_weight=weights)
    assert np.all(actual.cluster_centroids_ == expected.cluster_centroids_)
    assert np.array_equal(actual.labels_, expected.labels_)
    assert actual.cost_ == expected.cost_


def test_kprototypes_predict_and_fit_predict(mixed_data):
    actual = mojo.KPrototypes(
        3, n_init=1, random_state=9
    ).fit(mixed_data, categorical=[3, 4])
    expected = upstream_kprototypes.KPrototypes(
        3, n_init=1, random_state=9
    ).fit(mixed_data, categorical=[3, 4])
    query = mixed_data[:8].copy()
    query[0, 3] = "unknown"
    assert np.array_equal(
        actual.predict(query, categorical=[3, 4]),
        expected.predict(query, categorical=[3, 4]),
    )
    labels = actual.fit_predict(mixed_data, categorical=[3, 4])
    assert np.array_equal(labels, actual.labels_)


def test_kprototypes_gamma_parity(mixed_data):
    actual = mojo.KPrototypes(3, n_init=1, random_state=4).fit(
        mixed_data, categorical=[3, 4]
    )
    expected = upstream_kprototypes.KPrototypes(
        3, n_init=1, random_state=4
    ).fit(mixed_data, categorical=[3, 4])
    assert actual.gamma == expected.gamma


def test_kprototypes_function_parity(mixed_data):
    args = dict(
        X=mixed_data,
        categorical=[3, 4],
        n_clusters=3,
        max_iter=20,
        num_dissim=md.euclidean_dissim,
        cat_dissim=md.matching_dissim,
        gamma=None,
        init="Cao",
        n_init=1,
        verbose=0,
        random_state=check_random_state(8),
        n_jobs=1,
    )
    actual = mkp.k_prototypes(**args)
    args["num_dissim"] = ud.euclidean_dissim
    args["cat_dissim"] = ud.matching_dissim
    args["random_state"] = check_random_state(8)
    expected = upstream_kprototypes.k_prototypes(**args)
    assert np.array_equal(actual[2], expected[2])
    assert actual[3] == pytest.approx(expected[3])
    assert actual[4] == expected[4]
    assert actual[6] == expected[6]


def test_kprototypes_requires_mixed_columns(mixed_data):
    with pytest.raises(NotImplementedError, match="No categorical"):
        mojo.KPrototypes(3).fit(mixed_data, categorical=None)
    with pytest.raises(AssertionError, match="All columns"):
        mojo.KPrototypes(3).fit(
            mixed_data, categorical=list(range(mixed_data.shape[1]))
        )


def test_custom_training_dissimilarity_is_explicitly_rejected(
    categorical_data,
):
    def custom(a, b, **kwargs):
        return np.sum(a != b, axis=1)

    with pytest.raises(NotImplementedError, match="matching_dissim"):
        mojo.KModes(3, cat_dissim=custom).fit(categorical_data)


def test_functional_ffi_inputs_are_shape_checked():
    with pytest.raises(ValueError, match="same number of attributes"):
        mk.labels_cost(
            np.zeros((3, 2), dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            md.matching_dissim,
        )
    with pytest.raises(ValueError, match="same number of rows"):
        mkp.labels_cost(
            np.zeros((3, 1)),
            np.zeros((2, 1), dtype=np.int64),
            [np.zeros((2, 1)), np.zeros((2, 1), dtype=np.int64)],
            md.euclidean_dissim,
            md.matching_dissim,
            1.0,
        )
    with pytest.raises(ValueError, match="equal size"):
        mk.labels_cost(
            np.zeros((3, 1), dtype=np.int64),
            np.zeros((2, 1), dtype=np.int64),
            md.matching_dissim,
            sample_weight=[1.0],
        )


def test_sample_weights_must_be_finite():
    with pytest.raises(ValueError, match="finite"):
        mojo.KModes(2).fit(
            np.array([["a"], ["b"]]), sample_weight=[1.0, np.nan]
        )


def test_kprototypes_weighted_numeric_centers_remain_weighted_means():
    rng = np.random.RandomState(0)
    n_points = 30
    X = np.empty((n_points, 3), dtype=object)
    X[:, :2] = rng.randn(n_points, 2) * 3
    X[:, 2] = rng.randint(0, 3, n_points).astype(str)
    weights = rng.randint(1, 9, n_points).astype(float)
    model = mojo.KPrototypes(
        3, init="random", n_init=1, random_state=0, max_iter=20
    ).fit(X, categorical=[2], sample_weight=weights)
    numeric = X[:, :2].astype(float)
    for cluster in range(3):
        members = model.labels_ == cluster
        assert np.allclose(
            model._enc_cluster_centroids[0][cluster],
            np.average(numeric[members], axis=0, weights=weights[members]),
        )


def test_labels_cost_custom_dissimilarity_fallbacks():
    def categorical(a, b, **kwargs):
        return np.sum(a != b, axis=1)

    def numeric(a, b):
        return np.sum((a - b) ** 2, axis=1)

    Xcat = np.array([[0], [1], [2]])
    cat_centers = np.array([[0], [2]])
    labels, cost = mk.labels_cost(Xcat, cat_centers, categorical)
    assert labels.tolist() == [0, 0, 1]
    assert cost == 1.0

    Xnum = np.array([[0.0], [4.0]])
    labels, cost = mkp.labels_cost(
        Xnum,
        Xcat[:2],
        [np.array([[0.0], [4.0]]), cat_centers],
        numeric,
        categorical,
        1.0,
    )
    assert labels.tolist() == [0, 1]
    assert cost == 1.0
