# -*- coding: utf-8 -*-

"""kernel.py

This is the main module implementing fastwlk. It contains the Weisfeiler-Lehman
kernel version working fastest on sparse graphs.

"""


from abc import ABCMeta
from collections import Counter
from itertools import combinations, combinations_with_replacement, product
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np

from .utils.functions import (
    chunks,
    distribute_function,
    flatten_lists,
    generate_random_strings,
)
from .utils.validation import check_wl_input


class WeisfeilerLehmanKernel:
    """Weisfeiler-Lehmann kernel"""

    def __init__(
        self,
        n_jobs: int = 4,
        precomputed: bool = False,
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
        self.precomputed = precomputed

    def compute_wl_hashes(self, G: nx.Graph) -> Dict:
        """Computes Weisfeiler-Lehman hash histogram

        Args:
            G (nx.Graph): graph to compte the histogram of

            node_label (str): node label to use as the starting node label of the
                Weisfeiler-Lehman hashing process

            n_iter (int): number of iterations of the Weisfeiler-Lehman algorithm
                to run

        Returns:
            Dict: dictionary of the format {hash_value: n_nodes}.
        """
        hash_iter_0 = dict(
            Counter(list(dict(G.nodes(self.node_label)).values()))
        )
        hashes = dict(
            Counter(
                flatten_lists(
                    list(
                        nx.weisfeiler_lehman_subgraph_hashes(
                            G,
                            node_attr=self.node_label,
                            iterations=self.n_iter,
                        ).values()
                    )
                )
            )
        )
        return hashes | hash_iter_0

    def diagonal(self, X, Y, K):
        if X == Y:
            return np.diag(K), np.diag(K)
        else:
            # Compute the diagonal of the self-similarity kernel matrix
            diag_X = [self.dot_product((elem, elem)) for elem in X]
            diag_Y = [self.dot_product((elem, elem)) for elem in Y]
            return diag_X, diag_Y

    def parallel_dot_product(
        self, lst: Iterable,
    ) -> Iterable:  # pragma: no cover
        """Computes the inner product of elements in lst.

        Args:
            lst (Iterable): Iterable to compute the inner product of.

        Returns:
            Iterable: computed inner products.
        """
        res = list()
        for x in lst:
            res.append(
                {
                    list(x.keys())[0]: [
                        list(x.values())[0][0],
                        self.dot_product(list(x.values())[0][1]),
                    ]
                }
            )
        return res

    def dot_product(self, dicts: Tuple) -> int:  # pragma: no cover
        """Computes the inner product of two dictionaries using common
        keys. This dramatically improves computation times when the number
        of keys is large but the overlap between the two dictionaries in
        the tuple is low.

        Args:
            dicts (Tuple): pair of dictionaries to compute the kernel from.

        Returns:
            int: dot product value of dicts
        """
        running_sum = 0
        # 0 * x = 0 so we only need to iterate over common keys
        for key in set(dicts[0].keys()).intersection(dicts[1].keys()):
            running_sum += dicts[0][key] * dicts[1][key]
        return running_sum

    def compute_matrix(
        self, X: List[nx.Graph], Y: Union[List[nx.Graph], None] = None
    ) -> np.ndarray:
        """Computes the Gram matrix of the Weisfeiler-Lehman kernel.

        Args:
            X (List[nx.Graph]): List of graphs to use in the kernel.
            Y (Union[List[nx.Graph], None], optional): List of graphs to use in the kernel. Defaults to None.

        Returns:
            np.ndarray: _description_
        """

        def handle_hashes_single_threaded(
            X: Iterable[nx.Graph],
        ) -> Iterable[Dict]:  # pragma: no cover
            """Handles hashes when n_jobs = None on a single thread.

            Args:
                X (Iterable[nx.Graph]): Iterable of graphs to compute the hashes from

            Returns:
                Iterable[Dict]: hash histograms
            """
            X_hashed = list()
            for g in X:
                X_hashed.append(self.compute_wl_hashes(g))
            return X_hashed

        if not self.biased and self.normalize:
            self.biased = True
            if self.verbose:
                print("Setting biased=True to allow for normalization")

        check_wl_input(X)

        if Y is not None:
            check_wl_input(Y)

        if not self.precomputed:
            if Y == None:  # pragma: no cover
                Y = X

            if Y == X and self.n_jobs is not None:
                X_hashed = distribute_function(
                    self.compute_wl_hashes,
                    X,
                    self.n_jobs,
                    show_tqdm=self.verbose,
                    tqdm_label="Compute hashes of X",
                )
                Y_hashed = X_hashed
            elif X == Y and self.n_jobs is None:
                X_hashed = handle_hashes_single_threaded(X)
                Y_hashed = X_hashed
            elif X != Y and self.n_jobs is None:
                X_hashed = handle_hashes_single_threaded(X)
                Y_hashed = handle_hashes_single_threaded(Y)
            elif X != Y and self.n_jobs is not None:
                X_hashed = distribute_function(
                    self.compute_wl_hashes,
                    X,
                    n_jobs=self.n_jobs,
                    show_tqdm=self.verbose,
                    tqdm_label="Compute hashes of X",
                )
                Y_hashed = distribute_function(
                    self.compute_wl_hashes,
                    Y,
                    n_jobs=self.n_jobs,
                    show_tqdm=self.verbose,
                    tqdm_label="Compute hashes of Y",
                )
        else:
            X_hashed = X
            if X != Y and Y is not None:
                Y_hashed = Y
            else:
                Y_hashed = X_hashed

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

        keys = generate_random_strings(10, len(flatten_lists(iters_data)))
        iters = [
            {key: [idx, data]}
            for key, idx, data in zip(keys, iters_idx, iters_data)
        ]
        if self.n_jobs is not None:
            iters = list(chunks(iters, self.n_jobs,))
            matrix_elems = flatten_lists(
                distribute_function(
                    self.parallel_dot_product,
                    iters,
                    self.n_jobs,
                    show_tqdm=self.verbose,
                    tqdm_label="Compute dot products",
                )
            )
        else:
            matrix_elems = self.parallel_dot_product(iters)

        K = np.zeros((len(X_hashed), len(Y_hashed)), dtype=int)
        for elem in matrix_elems:
            coords = list(elem.values())[0][0]
            val = list(elem.values())[0][1]
            K[coords[0], coords[1]] = val
        if X == Y:
            # mirror the matrix along diagonal
            K = np.triu(K) + np.triu(K, 1).T

        if self.normalize:
            # From https://github.com/ysig/GraKeL/blob/33ffff18d99c13f8afc0438a5691cb1206b119fb/grakel/kernels/weisfeiler_lehman.py#L300
            X_diag, Y_diag = self.diagonal(X_hashed, Y_hashed, K)
            K = np.nan_to_num(np.divide(K, np.sqrt(np.outer(X_diag, Y_diag))))
        return K

    def compute_gram_matrix(
        self, X: List[nx.Graph], Y: Union[List[nx.Graph], None] = None
    ) -> np.ndarray:
        """Computes the Gram matrix of the Weisfeiler-Lehman kernel.
        If X == Y, then this is equivalent to calling compute_matrix. If X != Y, however, the self-similarity matrices of X and Y will be added in the upper-left and lower-right quadrants of the matrix, respectively, to ensure the output is actually a gram matrix.

        Args:
            X (List[nx.Graph]): set of data used for the kernel
            Y (Union[List[nx.Graph], None], optional): set of data used for the other kernel. Defaults to None.

        Returns:
            np.ndarray: Gram matrix
        """
        if Y is None:
            Y = X  # pragma: no cover
        if X == Y:
            return self.compute_matrix(X)
        else:
            K_XY = self.compute_matrix(X, Y)
            K_XX = self.compute_matrix(X)
            K_YY = self.compute_matrix(Y)
            full_K = np.zeros(
                (K_XX.shape[0] + K_YY.shape[0], K_XX.shape[0] + K_YY.shape[0])
            )
            full_K[: K_XX.shape[0], : K_XX.shape[0]] = K_XX
            full_K[K_XX.shape[0] :, K_XX.shape[0] :] = K_YY
            full_K[: K_XX.shape[0], K_XX.shape[0] :] = K_XY
            full_K[K_XX.shape[0] :, : K_XX.shape[0]] = K_XY.T
            return full_K
