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

You can also precompute the hashes prior to computing the kernel::

  # If for whatever reason you are comparing the same graphs multiple times
  # but with a different kernel config, you can precompute the hashes and set
  # the precomputed flag to True.

  import fastwlk
  wl_kernel = WeisfeilerLehmanKernel(
      n_jobs=6, n_iter=4, node_label="residue", precomputed=True, biased=True, verbose=True
  )
  hashes = [wl_kernel.compute_wl_hashes(graph) for graph in graphs]
  wl_kernel.compute_gram_matrix(hashes)
