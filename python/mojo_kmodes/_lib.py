"""ctypes loader for the compiled Mojo kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.path.join(ROOT, "dist", "libmojo-kmodes.so")
BUILD_SCRIPT = os.path.join(ROOT, "build", "build.sh")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mkm_matching_dissim": ([I, I, I, I, I], None),
    "mkm_euclidean_dissim": ([I, I, I, I, I], None),
    "mkm_encode_unicode": ([I] * 9, None),
    "mkm_mode_labels": ([I, I, I, I, I, I], None),
    "mkm_mode_labels_cost": ([I, I, I, I, I, I, I], F),
    "mkm_prototype_labels_cost": ([I] * 10 + [F], F),
    "mkm_cao_init": ([I] * 9, None),
    "mkm_kmodes_fit": ([I] * 14, I),
    "mkm_kprototypes_fit": ([I] * 19 + [F], I),
}

_library: ctypes.CDLL | None = None


def _sources() -> list[str]:
    return [
        os.path.join(path, name)
        for path, _, names in os.walk(os.path.join(ROOT, "src"))
        for name in names
        if name.endswith(".mojo")
    ]


def build(force: bool = False) -> str:
    sources = _sources()
    stale = (
        not os.path.exists(LIB_PATH)
        or max(os.path.getmtime(path) for path in sources)
        > os.path.getmtime(LIB_PATH)
    )
    if force or stale:
        subprocess.run(
            ["bash", BUILD_SCRIPT],
            cwd=ROOT,
            check=True,
            timeout=1800,
        )
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        initialize = _library.mkm_initialize_runtime
        initialize.argtypes = []
        initialize.restype = None
        initialize()
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    return array.ctypes.data


def i64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)
