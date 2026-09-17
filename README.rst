.. image:: https://img.shields.io/badge/license-BSD--3--Clause-blue
    :target: https://github.com/kbwestfall/dc3/blob/main/LICENSE.rst
    :alt: License

===
DC3
===

**Detector-Censored Cross-Correlation** — stellar kinematics (:math:`V`,
:math:`\sigma`) from galaxy-continuum spectra.

``dc3`` cross-correlates a galaxy spectrum ``G`` with a stellar template ``T``
and fits the resulting cross-correlation function ``X = G ∘ T`` with the model
``X_T = (T⊗B)' ∘ T``, where ``B`` is a parameterized broadening function.

The method's distinguishing feature is that the broadened template is
**detector-censored in exactly the same way as the galaxy spectrum** before
correlation — the same masking, truncation and apodization are applied to both.
It deliberately does *not* use the commutation ``X_T = (T ∘ T) ⊗ B``, which is
exact only for untruncated, unmasked data and introduces a systematic error of
order 10% in :math:`\sigma` on real spectra.

The algorithm is described in `Westfall, Bershady & Verheijen (2011, ApJS 193,
21) <https://ui.adsabs.harvard.edu/abs/2011ApJS..193...21W/abstract>`__, Paper
III of the DiskMass Survey. This package is a Python reimplementation of the
original C++ code, which is not distributed.


Status
------

**Pre-alpha, under active development.** The package is being built in phases;
nothing is stable, and there has been no release. See ``claude/`` for the
internal design record:

- ``dc3-original-implementation.md`` — a survey of the original C++ code base.
- ``dc3-python-port-plan.md`` — the phased plan this work follows.
- ``dc3-python-implementation.md`` — what has actually been built, and where it
  departed from the plan.


Installation
------------

.. code-block:: console

    pip install -e ".[dev]"

Requires Python 3.12 or later.

.. note::

    ``dc3`` depends on `ppxf <https://pypi.org/project/ppxf/>`__ for its
    variable-σ resolution matching and its analytic LOSVD Fourier transform.
    ``ppxf`` is not open-source — its license permits use but prohibits
    redistribution — so it is installed from PyPI as a dependency and no part
    of it is included here. This has consequences for anyone packaging ``dc3``
    downstream; see ``licenses/README.rst``.


Documentation
-------------

Documentation is built with Sphinx under ``doc/`` and will be hosted at
https://dc3.readthedocs.io/.


License
-------

BSD 3-Clause. See ``LICENSE.rst``, and ``licenses/README.rst`` for the terms
governing third-party code that ``dc3`` adapts or depends on.
