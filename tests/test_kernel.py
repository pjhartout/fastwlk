#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fastwlk` package."""

import json
import pickle
from typing import Callable, Iterable

import numpy as np
import pytest
from fastwlk.kernel import WeisfeilerLehmanKernel
from grakel import WeisfeilerLehman, graph_from_networkx
from pyprojroot import here

N_JOBS = 6

with open(here() / "data/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

with open("data/test_encoding_graph_1.json", "r") as f:
    encoding = json.load(f)


KX = np.array([[3608, 5062, 5009], [5062, 14532, 9726], [5009, 9726, 13649]])
KXY = np.array(
    [
        [6481, 6794, 6718, 7014],
        [14273, 15091, 15595, 14968],
        [12569, 12520, 12882, 13661],
    ]
)


test_validity_biased = [
    (graphs[:3], graphs[:3], KX),
    (graphs[:3], graphs[3:7], KXY),
]


@pytest.mark.parametrize("X, Y, expected", test_validity_biased)
def test_compute_gram_matrix_multithreaded(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=4,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_equal(K_fastwlk, expected)


@pytest.mark.parametrize("X, Y, expected", test_validity_biased)
def test_compute_gram_matrix_single_threaded(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=None,
        n_iter=4,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_equal(K_fastwlk, expected)


KX_unbiased = np.array([[0, 5062, 5009], [5062, 0, 9726], [5009, 9726, 0]])

test_validity_unbiased = [
    (graphs[:3], graphs[:3], KX_unbiased),
    (graphs[:3], graphs[3:7], KXY),
]


@pytest.mark.parametrize("X, Y, expected", test_validity_unbiased)
def test_compute_gram_matrix_unbiased(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=None,
        n_iter=4,
        node_label="residue",
        biased=False,
        verbose=False,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_equal(K_fastwlk, expected)


@pytest.mark.parametrize("X, Y, expected", test_validity_biased)
def test_compute_gram_matrix_precomputed(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=None,
        precomputed=True,
        n_iter=4,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    hashes_X = [wl_kernel.compute_wl_hashes(graph) for graph in X]
    hashes_Y = [wl_kernel.compute_wl_hashes(graph) for graph in Y]
    K_fastwlk = wl_kernel.compute_gram_matrix(hashes_X, hashes_Y)
    np.testing.assert_array_equal(K_fastwlk, expected)


test_compute_wl_hashes_data = [(graphs[1], encoding)]


@pytest.mark.parametrize("X, expected", test_compute_wl_hashes_data)
def test_compute_wl_hashes(X, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=2,
        precomputed=False,
        n_iter=2,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    hashes = wl_kernel.compute_wl_hashes(graphs[1])
    assert hashes == expected


test_grakel_equality_data = [
    (graphs[:10], graphs[:10]),
    (graphs[:10], graphs[10:30]),
]


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt


@pytest.mark.parametrize("X, Y", test_grakel_equality_data)
def test_grakel_equivalence(X, Y):
    Xt = networkx2grakel(X)
    Yt = networkx2grakel(Y)
    wl_kernel_grakel = WeisfeilerLehman(n_jobs=N_JOBS, n_iter=3)
    KXY_grakel = wl_kernel_grakel.fit(Xt).transform(Yt).T
    wl_kernel_fastwlk = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=3,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    KXY_fastwlk = wl_kernel_fastwlk.compute_gram_matrix(X, Y)
    np.testing.assert_array_equal(KXY_fastwlk, KXY_grakel)
    return KXY


KX_norm = np.array(
    [
        [1.0, 0.73037069, 0.74368576],
        [0.73037069, 1.0, 0.71130753],
        [0.74368576, 0.71130753, 1.0],
    ]
)
KXY_norm = np.array(
    [
        [0.79378634, 0.71006284, 0.74347089, 0.75475861],
        [0.86111583, 0.77691583, 0.85014706, 0.79339745],
        [0.78030479, 0.6632504, 0.72261873, 0.74512092],
    ]
)

test_validity_normalized = [
    (graphs[:3], graphs[:3], KX_norm),
    (graphs[:3], graphs[3:7], KXY_norm),
]


@pytest.mark.parametrize("X, Y, expected", test_validity_normalized)
def test_compute_gram_matrix_normalized(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=3,
        normalize=True,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_almost_equal(K_fastwlk, expected, decimal=8)


KX_norm_unbiased = np.array(
    [
        [0.0, 0.73037069, 0.74368576],
        [0.73037069, 0.0, 0.71130753],
        [0.74368576, 0.71130753, 0.0],
    ]
)

test_validity_normalized_unbiased = [
    (graphs[:3], graphs[:3], KX_norm_unbiased),
    (graphs[:3], graphs[3:7], KXY_norm),
]


@pytest.mark.parametrize("X, Y, expected", test_validity_normalized)
def test_unbiased_normalized(X, Y, expected):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=3,
        normalize=True,
        node_label="residue",
        biased=False,
        verbose=True,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_almost_equal(K_fastwlk, expected, decimal=8)
