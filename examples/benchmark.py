#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""benchmark.py

Benchmarks our implementation of the Weisfeiler Lehman kernel with other
popular alternatives.

"""

import functools
import itertools
import pickle
import time
import tracemalloc
from typing import Callable, Iterable

import numpy as np
from fastwlk.kernel import WeisfeilerLehmanKernel
from grakel import WeisfeilerLehman, graph_from_networkx
from pyprojroot import here
from tqdm import tqdm

N_JOBS = 6


class bcolors:
    OKGREEN = "\033[92m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"


def timeit(func: Callable) -> Callable:
    """Timeit decorator

    Args:
        func (Callable): function to time

    Returns:
        Callable: function used to time
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs):
        """time_wrapper's doc string"""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        time_elapsed = time.perf_counter() - start
        print(
            "Function Name       :"
            + bcolors.OKGREEN
            + f"{func.__name__}"
            + bcolors.ENDC
        )
        print(
            "Time                :"
            + bcolors.OKGREEN
            + f"{time_elapsed} seconds"
            + bcolors.ENDC
        )
        return result

    return time_closure


def measure_memory(func: Callable, *args, **kwargs):
    """Decorator used to measure memory footprint
    Partly from: https://gist.github.com/davidbegin/d24e25847a952441b62583d973b6c62e

    Args:
        func (Callable): function to measure memory footprint of.
    """

    @functools.wraps(func)
    def memory_closure(*args, **kwargs):
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        print(
            "Function Name       :"
            + bcolors.OKGREEN
            + f"{func.__name__}"
            + bcolors.ENDC
        )
        print(
            "Current memory usage:"
            + bcolors.OKGREEN
            + f"{current / 10**6}MB"
            + bcolors.ENDC
        )
        print(
            f"Peak                :"
            + bcolors.OKGREEN
            + f"{peak / 10**6}MB"
            + bcolors.ENDC
        )
        tracemalloc.stop()
        return result

    return memory_closure


def networkx2grakel(X: Iterable) -> Iterable:
    Xt = list(graph_from_networkx(X, node_labels_tag="residue"))
    return Xt


@timeit
@measure_memory
def grakel_benchmark(X, Y):
    X = networkx2grakel(X)
    Y = networkx2grakel(Y)
    wl_kernel = WeisfeilerLehman(n_jobs=N_JOBS, n_iter=3, normalize=True)
    KXY = wl_kernel.fit(X).transform(Y).T
    return KXY


@timeit
@measure_memory
def fastwlk_benchmark(X, Y):
    wl_kernel = WeisfeilerLehmanKernel(
        n_jobs=N_JOBS,
        n_iter=3,
        node_label="residue",
        biased=True,
        verbose=False,
        normalize=True,
    )
    KXY = wl_kernel.compute_matrix(X, Y)
    return KXY


def main():
    with open(here() / "data/graphs.pkl", "rb") as f:
        graphs = pickle.load(f)

    # Benchmarking
    print("====================================")
    print("Benchmarking Weisfeiler Lehman Kernel - Self-Similarity Test")
    print("## Grakel Implementation")
    KX_grakel = grakel_benchmark(graphs, graphs)

    print("## fastwlk Implementation")
    KX_fastwlk = fastwlk_benchmark(graphs, graphs)
    np.testing.assert_array_equal(KX_fastwlk, KX_grakel)
    print(KX_fastwlk)
    print("====================================")
    print("Benchmarking Weisfeiler Lehman Kernel - Similarity Test")
    print("## Grakel Implementation")
    KXY_grakel = grakel_benchmark(graphs[:50], graphs[50:])
    print("## fastwlk Implementation")
    KXY_fastwlk = fastwlk_benchmark(graphs[:50], graphs[50:])
    np.testing.assert_array_equal(KXY_fastwlk, KXY_grakel)
    print("Done")


if __name__ == "__main__":
    main()
