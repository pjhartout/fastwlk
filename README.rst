=============================
FastWLK
=============================

.. image:: https://github.com/pjhartout/fastwlk/actions/workflows/main/badge.svg
        :target: https://github.com/pjhartout/fastwlk/


.. image:: https://img.shields.io/pypi/v/fastwlk.svg
        :target: https://pypi.python.org/pypi/fastwlk


.. image:: https://codecov.io/gh/pjhartout/fastwlk/branch/main/graph/badge.svg?token=U054MJONED
      :target: https://codecov.io/gh/pjhartout/fastwlk


.. image:: https://readthedocs.org/projects/fastwlk/badge/?version=latest
        :target: https://fastwlk.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



``fastwlk`` is a Python package that implements a fast version of the
Weisfeiler-Lehman kernel. It manages to outperform current state-of-the-art
implementations on sparse graphs by implementing a number of improvements
compared to vanilla implementations:

1. It parallelizes the execution of Weisfeiler-Lehman hash computations since
   each graph's hash can be computed independently prior to computing the
   kernel.

2. It parallelizes the computation of similarity of graphs in RKHS by computing
   batches of the inner products independently.

3. On sparse graphs, lots of computations are spent processing positions/hashes
   that do not actually overlap between graph representations. As such, we
   manually loop over the overlapping keys, outperforming numpy dot
   product-based implementations.

The emphasis here being placed on `sparse`. If you are not sure the graphs you
are using are either sparse or dissimilar enough, try to benchmark this package
with others out there.
