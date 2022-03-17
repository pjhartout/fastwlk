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
    """Simply distributes the execution of func across multiple cores to process X faster"""
    if total == 1:
        total = len(X)
    if show_tqdm:
        with tqdm_joblib(tqdm(desc=tqdm_label, total=total)) as progressbar:
            Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    else:
        Xt = Parallel(n_jobs=n_jobs)(delayed(func)(x, **kwargs) for x in X)
    return Xt


def flatten_lists(lists: list) -> list:
    """Removes nested lists"""
    result = list()
    for _list in lists:
        _list = list(_list)
        if _list != []:
            result += _list
        else:  # pragma: no cover
            continue
    return result


def chunks(lst, n):
    """returns lst divided into n chunks approx. the same size"""
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
