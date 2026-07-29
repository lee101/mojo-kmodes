"""Benchmarks against kmodes 0.12.2 on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from kmodes.kmodes import KModes as UpstreamKModes  # noqa: E402
from kmodes.kprototypes import KPrototypes as UpstreamKPrototypes  # noqa: E402
from kmodes.util.dissim import matching_dissim as upstream_matching  # noqa: E402
from mojo_kmodes import KModes, KPrototypes  # noqa: E402
from mojo_kmodes.util.dissim import matching_dissim  # noqa: E402


def time_best(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def categorical_clusters(n, d, k, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.arange(n) % k
    data = np.empty((n, d), dtype=np.int64)
    for column in range(d):
        base = labels * 4 + column % 3
        noise = rng.randint(0, 3, n)
        data[:, column] = base + noise
    rng.shuffle(data)
    return data.astype(str)


def mixed_clusters(n, dnum, dcat, k, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.arange(n) % k
    numeric = labels[:, None] * 4.0 + rng.normal(
        scale=0.6, size=(n, dnum)
    )
    categorical = np.empty((n, dcat), dtype=object)
    for column in range(dcat):
        categorical[:, column] = labels * 3 + rng.randint(0, 2, n)
    data = np.empty((n, dnum + dcat), dtype=object)
    data[:, :dnum] = numeric
    data[:, dnum:] = categorical
    order = rng.permutation(n)
    initial_num = np.repeat(
        (np.arange(k) * 4.0)[:, None], dnum, axis=1
    )
    initial_cat = np.repeat(
        (np.arange(k) * 2)[:, None], dcat, axis=1
    )
    return (
        data[order],
        list(range(dnum, dnum + dcat)),
        [initial_num, initial_cat],
    )


CASES = []


def case(name, repeat=3):
    def decorate(builder):
        CASES.append((name, repeat, builder))
        return builder

    return decorate


@case("matching_dissim (1M x 16)")
def matching_case():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 32, size=(1_000_000, 16), dtype=np.int64)
    b = rng.integers(0, 32, size=16, dtype=np.int64)
    return (
        lambda: matching_dissim(a, b),
        lambda: upstream_matching(a, b),
    )


@case("KModes.predict (200k x 12, k=8)")
def kmodes_predict_case():
    train = categorical_clusters(20_000, 12, 8, seed=1)
    query = categorical_clusters(200_000, 12, 8, seed=2)
    ours = KModes(8, n_init=1, random_state=3).fit(train)
    theirs = UpstreamKModes(8, n_init=1, random_state=3).fit(train)
    return lambda: ours.predict(query), lambda: theirs.predict(query)


@case("KModes.fit (50k x 10, k=8)", repeat=2)
def kmodes_fit_case():
    data = categorical_clusters(50_000, 10, 8, seed=4)
    return (
        lambda: KModes(8, n_init=1, random_state=5).fit(data),
        lambda: UpstreamKModes(8, n_init=1, random_state=5).fit(data),
    )


@case("KPrototypes.predict (120k, 4 num + 6 cat, k=8)")
def kprototypes_predict_case():
    train, categorical, initial = mixed_clusters(
        20_000, 4, 6, 8, seed=6
    )
    query, _, _ = mixed_clusters(120_000, 4, 6, 8, seed=7)
    ours = KPrototypes(8, init=initial, n_init=1, random_state=8).fit(
        train, categorical=categorical
    )
    theirs = UpstreamKPrototypes(
        8, init=initial, n_init=1, random_state=8
    ).fit(
        train, categorical=categorical
    )
    return (
        lambda: ours.predict(query, categorical=categorical),
        lambda: theirs.predict(query, categorical=categorical),
    )


@case("KPrototypes.fit (30k, 3 num + 5 cat, k=6)", repeat=2)
def kprototypes_fit_case():
    data, categorical, initial = mixed_clusters(
        30_000, 3, 5, 6, seed=9
    )
    return (
        lambda: KPrototypes(
            6, init=initial, n_init=1, random_state=10
        ).fit(
            data, categorical=categorical
        ),
        lambda: UpstreamKPrototypes(
            6, init=initial, n_init=1, random_state=10
        ).fit(
            data, categorical=categorical
        ),
    )


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-kmodes | kmodes 0.12.2 | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, repeat, builder in CASES:
        ours, theirs = builder()
        ours()
        mojo_time = time_best(ours, repeat)
        upstream_time = time_best(theirs, repeat)
        print(
            f"| {name} | {mojo_time * 1000:.1f} ms | "
            f"{upstream_time * 1000:.1f} ms | "
            f"{upstream_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
