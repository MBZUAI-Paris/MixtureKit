.. utiliKit documentation master file, created by
   sphinx-quickstart on Wed Feb 26 13:49:34 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MixtureKit: The utility package for Mixture-of-Experts (MoE) algorithms
=====================================================================

The MixtureKit package provides high-level helper function to merge pretrained and finetuned models
into a unified framework, integrating a Mixture-of-Experts (MoE) architecture.

Installation
------------

if you want the latest version available::

  https://github.com/MBZUAI-Paris/MixtureKit
  cd MixtureKit
  pip install -e .

Documentation & Examples
------------------------

Documentation about the main utiliKit functions is available
`here <api.html>`_ and examples are available `here <auto_examples/index.html>`_.

Build the documentation
-----------------------

To build the documentation you will need to run:

.. code-block::

    pip install -U '.[doc]'
    cd doc_configuration
    sphinx-build -b html doc_configuration docs

API
---

.. toctree::
    :maxdepth: 1

    api.rst

