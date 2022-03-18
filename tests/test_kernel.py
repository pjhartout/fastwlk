#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fastwlk` package."""

import pickle
from typing import Callable, Iterable

import numpy as np
import pytest
from fastwlk.kernel import WeisfeilerLehmanKernel
from fastwlk.utils.functions import compute_wl_hashes
from pyprojroot import here

N_JOBS = 6

with open(here() / "data/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

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
    hashes_X = [
        compute_wl_hashes(graph, node_label="residue", n_iter=4) for graph in X
    ]
    hashes_Y = [
        compute_wl_hashes(graph, node_label="residue", n_iter=4) for graph in Y
    ]
    K_fastwlk = wl_kernel.compute_gram_matrix(hashes_X, hashes_Y)
    np.testing.assert_array_equal(K_fastwlk, expected)
