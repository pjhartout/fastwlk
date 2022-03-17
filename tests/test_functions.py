#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_filename.py

***file description***

"""

import json
import pickle
from typing import List

import pytest
from fastwlk.utils.functions import (
    chunks,
    compute_wl_hashes,
    distribute_function,
    flatten_lists,
    generate_random_strings,
)
from pyprojroot import here

# Load toy data
with open(here() / "data/graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

with open("data/test_encoding_graph_1.json", "r") as f:
    encoding = json.load(f)


def test_generate_random_strings_passes():
    gen = generate_random_strings(10, 10)
    assert len(gen) == 10


def test_generate_random_strings_fails():
    with pytest.raises(Exception) as e_info:
        generate_random_strings(1, 70)


def test_distribute_function():
    def test_func(x):
        return x * x

    X = list(range(100))

    res = distribute_function(test_func, X, -1)
    assert type(res) == type(list())


test_lists_to_flatten = [
    ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 1, 2, 3]),
    ([[1, 2, 3], [[1, 2, 3], [1, 2, 3]]], [1, 2, 3, [1, 2, 3], [1, 2, 3]]),
]


@pytest.mark.parametrize("input, expected", test_lists_to_flatten)
def test_flatten_lists(input, expected):
    assert list(flatten_lists(input)) == expected


# test chuncks
test_chunks = [
    (
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10]],
    ),
    ([1, 2, 3, 4], [[1], [2], [3], [4]],),
]


@pytest.mark.parametrize("input, expected", test_chunks)
def test_chunks(input, expected):
    assert list(chunks(input, 4)) == expected


test_compute_wl_hashes = [(graphs[1], encoding)]


@pytest.mark.parametrize("X, expected", test_compute_wl_hashes)
def test_compute_wl_hashes(X, expected):
    hashes = compute_wl_hashes(graphs[1], node_label="residue", n_iter=2)
    assert hashes == expected
