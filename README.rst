=============================
FastWLK
=============================

.. image:: https://github.com/pjhartout/fastwlk/actions/workflows/main.yml/badge.svg
        :target: https://github.com/pjhartout/fastwlk/


.. image:: https://img.shields.io/pypi/v/fastwlk.svg
        :target: https://pypi.python.org/pypi/fastwlk


.. image:: https://codecov.io/gh/pjhartout/fastwlk/branch/main/graph/badge.svg?token=U054MJONED
      :target: https://codecov.io/gh/pjhartout/fastwlk

.. image:: https://img.shields.io/website-up-down-green-red/http/shields.io.svg
   :target: https://pjhartout.github.io/fastwlk/


Quick Links
-------------------------
`Documentation`_

`Installation`_

`Usage`_

`Contributing`_


What does ``fastwlk`` do?
-------------------------


``fastwlk`` is a Python package that implements a fast version of the
Weisfeiler-Lehman kernel. It manages to outperform current state-of-the-art
implementations on sparse graphs by implementing a number of improvements
compared to vanilla implementations:

1. It parallelizes the execution of Weisfeiler-Lehman hash computations since
   each graph's hash can be computed independently prior to computing the
   kernel.

2. It parallelizes the computation of similarity of graphs in RKHS by computing
   batches of the inner products independently.

3. When comparing graphs, lots of computations are spent processing
   positions/hashes that do not actually overlap between Weisfeiler-Lehman
   histograms. As such, we manually loop over the overlapping keys,
   outperforming numpy dot product-based implementations on most graphs derived
   from proteins.

This implementation works best when graphs have relatively few connections and
are reasonably dissimilar from one another. If you are not sure the graphs you
are using are either sparse or dissimilar enough, try to benchmark this package
with others out there.

How fast is ``fastwlk``?
-------------------------

Running the benchmark script in ``examples/benchmark.py`` shows that for the
graphs in ``data/graphs.pkl``, we get an approximately 80% speed improvement
over other implementations like `grakel`_. The example dataset contains 2-nn
graphs extracted from 100 random proteins from the human proteome from the
`AlphaFold EBI database`_.

To see how much faster this implementation is for your use case:

.. code-block:: console

   $ git clone git://github.com/pjhartout/fastwlk
   $ poetry install
   $ poetry run python examples/benchmark.py

You will need to swap out the provided ``graphs.pkl`` with a pickled iterable of
graphs from the database you are interested in.

.. _Documentation: https://pjhartout.github.io/fastwlk/
.. _Installation: https://pjhartout.github.io/fastwlk/installation.html
.. _Usage: https://pjhartout.github.io/fastwlk/usage.html
.. _Contributing: https://pjhartout.github.io/fastwlk/contributing.html
.. _grakel: https://github.com/ysig/GraKeL
.. _AlphaFold EBI database: https://alphafold.ebi.ac.uk/download
