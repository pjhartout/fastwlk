.. highlight:: shell

============
Installation
============


Stable release
--------------

To install FastWLK, run this command in your terminal:

.. code-block:: console

    $ pip install fastwlk

This is the preferred method to install FastWLK, as it will always install the
most recent stable release. If you are using poetry (which is used to develop
``fastwlk``), you can also install it via:

.. code-block:: console

    $ poetry add fastwlk


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources and for contributions
------------------------------------

The sources for FastWLK can be downloaded from the `Github repo`_.

Clone the repository, install `poetry`_ and build the project:

.. code-block:: console

   $ git clone git://github.com/pjhartout/fastwlk
   $ poetry install
   $ poetry build

If you are wondering where ``setup.py`` is, it is no longer required for pip. See `PEP 518`_


.. _Github repo: https://github.com/pjhartout/fastwlk
.. _poetry: https://python-poetry.org/
.. _pep 518: https://peps.python.org/pep-0518/
