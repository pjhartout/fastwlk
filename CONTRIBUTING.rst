.. highlight:: shell

============
Contributing
============

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/pjhartout/fastwlk/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Making a PR for bug fixes or enhancements
---------------

To start on a new feature

.. code-block:: console

   $ git clone git://github.com/pjhartout/fastwlk
   $ poetry install


To work inside the virtual environment provided by poetry:

.. code-block:: console

   $ poetry shell


Before committing changes, make sure you make feature branch:

.. code-block:: console

   $ git switch -c my-awesome-improvement

The above command is only available for git > 2.23. Otherwise:

.. code-block:: console

   $ git checkout -b my-awesome-improvement

After making your changes:

.. code-block:: console

   $ poetry shell
   $ make test
   $ make coverage
