Installation
============

``skfeaturellm`` currently supports:

- Python versions 3.10, 3.11, and 3.12.
- Operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher

See here for a `full list of precompiled wheels available on PyPI <https://pypi.org/simple/skfeaturellm/>`_.
There are three different installation types, depending on your use case:

- Installing stable ``skfeaturellm`` releases - for most users, for production environments
- Installing the latest unstable ``skfeaturellm`` development version - for pre-release tests
- For developers of ``skfeaturellm`` and 3rd party extensions: Developer setup for extensions and contributions

Each of these three setups are explained below.

Installing Release Versions
~~~~~~~~~~~~~~~~~
This installation method is recommended for most users. It installs the latest stable release of ``skfeaturellm`` from `PyPI <https://pypi.org/project/skfeaturellm/>`_.
To install ``skfeaturellm`` with core dependencies, via ``pip``, type:

.. code-block:: bash

    pip install skfeaturellm

This will install the latest stable release along with its dependencies.


Installing latest Development Version
~~~~~~~~~~~~~~~~~~~
For:
- pre-release tests, e.g., early testing of new features
- not for reliable production use
- not for contributors or extenders

This type of ``skfeaturellm`` installation obtains a latest static snapshot of the repository. It is intended for developers that wish to build or test code using a version of the library that contains the all of the latest and current updates
To install the latest version of ``skfeaturellm`` directly from the repository, you can use the ``pip`` package manager to install directly from the GitHub repository:

.. code-block:: bash

    pip install git+https://github.com/RobertoCorti/skfeaturellm.git

To install from a specific branch, use the following command:

.. code-block:: bash
    pip install git+https://github.com/RobertoCorti/skfeaturellm.git@<branch_name>


Installing Full Developer Setup
~~~~~~~~~~~~~~~~~~~
For whom:
- contributors to the sktime project
- developers of extensions in closed code bases
- developers of 3rd party extensions released as open source

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/RobertoCorti/skfeaturellm.git
    cd skfeaturellm

2. Install development dependencies:

.. code-block:: bash

    pip install -e ".[dev]"
    # or with Poetry
    poetry install
