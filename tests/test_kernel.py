#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `fastwlk` package."""

import pickle
from typing import Callable, Iterable

import numpy as np
import pytest
from fastwlk.kernel import WeisfeilerLehmanKernel
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


test_validity = [
    (graphs[:3], graphs[:3], KX),
    (graphs[:3], graphs[3:7], KXY),
]

# pytestmark = pytest.mark.filterwarnings("error")


@pytest.mark.parametrize("X, Y, expected", test_validity)
def test_compute_gram_matrix(X, Y, expected):
    """Sample pytest test function with the pytest fixture as an argument."""

    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=4,
        node_label="residue",
        biased=True,
        verbose=False,
    )
    K_fastwlk = wl_kernel.compute_gram_matrix(X, Y)
    np.testing.assert_array_equal(K_fastwlk, expected)
