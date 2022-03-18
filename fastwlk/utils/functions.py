# -*- coding: utf-8 -*-
import contextlib
from collections import Counter
from itertools import product
from string import ascii_letters
from typing import Any, Callable, Dict, Iterable, List

import joblib
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm

from .exception import UniquenessError


def generate_random_strings(string_length: int, n_strings: int) -> List[str]:
    """Generates random strings

    Args:
        string_length (int): string length of the elements in the returned List.
        n_strings (int): number of strings to return

    Raises:
        UniquenessError: if the number of unique strings cannot be returned
            from the given string length.

    Returns:
        List[str]: unique strings
    """
    unique_strings = list()
    for idx, item in enumerate(product(ascii_letters, repeat=string_length)):
        if idx == n_strings:
            break
        unique_strings.append("".join(item))

    if len(unique_strings) != n_strings:
        raise UniquenessError(
            f"Cannot generate enough unique strings from given string length "
            "of {string_length}. Please increase the string_length to continue."
        )
    return unique_strings


def distribute_function(
    func: Callable,
    X: Iterable,
    n_jobs: int,
    tqdm_label: str = None,
    total: int = 1,
    show_tqdm: bool = True,
    **kwargs,
) -> Any:
    """Wraps a function to execute it across multiple threads.
    **kwargs are passed to the function to be parallelized.

    Args:
        func (Callable): function to be parallelized
        X (Iterable): data to use in func.
        n_jobs (int): number of threads to execute func
        tqdm_label (str, optional): Label of the tqdm progress bar. Defaults to None.
        total (int, optional): Total length of X if it cannot be deduced from X. Defaults to 1.
        show_tqdm (bool, optional): Whether or not to show tqdm. Defaults to True.

    Returns:
        Any: result of the parallel execution of func on X.
    """
    if total == 1:
        total = len(X)
    if show_tqdm:
        with tqdm_joblib(tqdm(desc=tqdm_label, total=total)) as progressbar:
            Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    else:
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def flatten_lists(lists: List) -> List:
    """Removes one level of nested list.
    i.e.

    flatten_lists([[1],[1,1,[1,1]]])
    becomes
    [1,1,1,[1,1]]

    Args:
        lists (list): list to remove one level of nestedness from.

    Returns:
        list: flattened list
    """
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:  # pragma: no cover
            continue
    return result


def chunks(lst: List, n: int) -> List:
    """Divides lst into n roughly equally-sized chunks.

    Args:
        lst (List): list to divide.
        n (int): number of chunks to divide lst

    Returns:
        List: lst divided into n roughly equally sized chunks.
    """
    k, m = divmod(len(lst), n)
    return (
        lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)
    )


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar.

    Code stolen from https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def compute_wl_hashes(G: nx.Graph, node_label: str, n_iter: int) -> Dict:
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
    hash_iter_0 = dict(Counter(list(dict(G.nodes(node_label)).values())))
    hashes = dict(
        Counter(
            flatten_lists(
                list(
                    nx.weisfeiler_lehman_subgraph_hashes(
                        G, node_attr=node_label, iterations=n_iter,
                    ).values()
                )
            )
        )
    )
    return hashes | hash_iter_0
