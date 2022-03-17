# -*- coding: utf-8 -*-
import functools
import time
import tracemalloc
from typing import Callable


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
