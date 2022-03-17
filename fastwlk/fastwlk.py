from abc import ABCMeta
from itertools import combinations, combinations_with_replacement, product
from typing import Any, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np

from .utils.functions import (
    chunks,
    compute_wl_hashes,
    distribute_function,
    flatten_lists,
    generate_random_strings,
)
from .utils.validation import check_wl_input

1


class Kernel(metaclass=ABCMeta):
    """Defines skeleton of descriptor classes"""

    def __init__(self):
        pass

    def compute_gram_matrix(self, X: Iterable, Y: Iterable = None) -> Iterable:
        return X


class WeisfeilerLehmanKernel(Kernel):
    """Weisfeiler-Lehmann kernel"""

    def __init__(
        self,
        n_jobs: int,
        n_iter: int = 3,
        node_label: str = "residue",
        normalize: bool = False,
        biased: bool = False,
        verbose: bool = False,
    ):
        self.n_iter = n_iter
        self.node_label = node_label
        self.normalize = normalize
        if n_jobs is not None:
            self.n_jobs = int(n_jobs)
        else:
            self.n_jobs = None
        self.biased = biased
        self.verbose = verbose

    def compute_gram_matrix(
        self, X: List[nx.Graph], Y: Union[List[nx.Graph], None] = None
    ) -> np.ndarray:
        def parallel_dot_product(lst: Iterable) -> Iterable:
            res = list()
            for x in lst:
                res.append(
                    {
                        list(x.keys())[0]: [
                            list(x.values())[0][0],
                            dot_product(list(x.values())[0][1]),
                        ]
                    }
                )
            return res

        def dot_product(dicts: Tuple) -> int:

            running_sum = 0
            # 0 * x = 0 so we only need to iterate over common keys
            for key in set(dicts[0].keys()).intersection(dicts[1].keys()):
                running_sum += dicts[0][key] * dicts[1][key]
            return running_sum

        def handle_hashes_single_threaded(
            X: Iterable[nx.Graph],
        ) -> Iterable[nx.Graph]:
            X_hashed = list()
            for g in X:
                X_hashed.append(
                    compute_wl_hashes(
                        g, node_label=self.node_label, n_iter=self.n_iter
                    )
                )
            return X_hashed

        check_wl_input(X)
        check_wl_input(Y)

        if Y == None:
            Y = X

        if Y == X and self.n_jobs is not None:
            X_hashed = distribute_function(
                compute_wl_hashes,
                X,
                self.n_jobs,
                show_tqdm=self.verbose,
                tqdm_label="Compute hashes of X",
                node_label=self.node_label,
                n_iter=self.n_iter,
            )
            Y_hashed = X_hashed
        elif X == Y and self.n_jobs is None:
            X_hashed = handle_hashes_single_threaded(X)
            Y_hashed = X_hashed
        elif X != Y and self.n_jobs is None:
            X_hashed = handle_hashes_single_threaded(X)
            Y_hashed = X_hashed = handle_hashes_single_threaded(Y)
        else:
            X_hashed = distribute_function(
                compute_wl_hashes,
                X,
                n_jobs=self.n_jobs,
                show_tqdm=self.verbose,
                node_label=self.node_label,
                tqdm_label="Compute hashes of X",
                n_iter=self.n_iter,
            )
            Y_hashed = distribute_function(
                compute_wl_hashes,
                Y,
                n_jobs=self.n_jobs,
                show_tqdm=self.verbose,
                tqdm_label="Compute hashes of Y",
                node_label=self.node_label,
                n_iter=self.n_iter,
            )

        # It's faster to process n_jobs lists than to have one list and
        # dispatch one item at a time.
        if not self.biased and X == Y:
            iters_data = list(list(combinations(X_hashed, 2)))
            iters_idx = list(combinations(range(len(X_hashed)), 2))
        elif self.biased and X == Y:
            iters_data = list(list(combinations_with_replacement(X_hashed, 2)))
            iters_idx = list(
                combinations_with_replacement(range(len(X_hashed)), 2)
            )
        else:
            iters_data = list(list(product(X_hashed, Y_hashed)))
            iters_idx = list(
                product(range(len(X_hashed)), range(len(Y_hashed)))
            )

        if X != Y and not self.biased:
            raise RuntimeWarning(
                "Ignoring biased parameter. X_i is never "
                "compared to itself when X != Y. Set biased=False to get rid "
                "of this warning."
            )

        keys = generate_random_strings(10, len(flatten_lists(iters_data)))
        iters = [
            {key: [idx, data]}
            for key, idx, data in zip(keys, iters_idx, iters_data)
        ]
        if self.n_jobs is not None:
            iters = list(chunks(iters, self.n_jobs,))
            matrix_elems = flatten_lists(
                distribute_function(
                    parallel_dot_product,
                    iters,
                    self.n_jobs,
                    show_tqdm=self.verbose,
                    tqdm_label="Compute dot products",
                )
            )
        else:
            matrix_elems = parallel_dot_product(iters)

        K = np.zeros((len(X_hashed), len(Y_hashed)), dtype=int)
        for elem in matrix_elems:
            coords = list(elem.values())[0][0]
            val = list(elem.values())[0][1]
            K[coords[0], coords[1]] = val
        if X == Y:
            # mirror the matrix along diagonal
            K = np.triu(K) + np.triu(K, 1).T
        return K
