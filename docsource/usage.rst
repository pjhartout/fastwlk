=====
Usage
=====

Here's an example of how to use ``fastwlk``::

  import fastwlk
  from pyproject import here

  # Let's first load some graphs from a pickle file
  # graphs.pkl contains 2-nn graphs extracted from
  # the AlphaFold human proteome database.

  with open(here() / "data/graphs.pkl", "rb") as f:
      graphs = pickle.load(f)

  wl_kernel = WeisfeilerLehmanKernel(
      n_jobs=6, n_iter=4, node_label="residue", biased=True, verbose=True
  )

  # Returns self-similarity kernel matrix
  KX = wl_kernel.compute_gram_matrix(graphs)
  # Returns the kernel between two graph distributions
  KXY = wl_kernel.compute_gram_matrix(graphs[:30], graphs[:30])