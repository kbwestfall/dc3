Licenses
========

``dc3`` is distributed under the BSD 3-Clause License; see ``LICENSE.rst`` in
the repository root. This directory holds the licenses of third-party code that
``dc3`` either incorporates or depends on, together with the terms each one
imposes.

There are two distinct categories here, and the difference matters:

- **Adapted code** — source that has been copied into ``dc3`` and modified.
  This is permitted only for permissively licensed projects, and every adapted
  module carries an attribution note in its module docstring naming the project
  and the file it came from.
- **Dependencies** — packages ``dc3`` imports at run time but does not
  redistribute. Their licenses do not govern ``dc3``'s own terms, but one of
  them constrains how ``dc3`` may be packaged.


Adapted code
------------

PypeIt — ``PYPEIT_LICENSE.rst``
    BSD 3-Clause. https://github.com/pypeit/PypeIt

    Adapted into ``dc3/pkg/`` (logging, exceptions, cache, data-path registry),
    ``dc3/par/`` (the ``ParSet`` and ``FuncPar`` parameter system),
    ``dc3/core/bitmask.py`` and ``dc3/scripts/scriptbase.py``, and used as the
    template for this package's build and documentation configuration.

MaNGA Data Analysis Pipeline (``mangadap``) — ``MANGADAP_LICENSE.md``
    BSD 3-Clause. https://github.com/sdss/mangadap

    Adapted into the spectral-resolution, resampling and template-library
    handling of ``dc3/core/`` and ``dc3/templates.py``.

Both permit redistribution with attribution, so adapted code may be shipped in
the ``dc3`` wheel.


Dependencies
------------

.. warning::

    **ppxf is not redistributable, and no ppxf code may be copied into dc3.**

``ppxf`` (Cappellari) is a *hard dependency* of ``dc3`` — it supplies
``ppxf_util.varsmooth`` (variable-σ resolution matching) and
``ppxf_util.losvd_rfft`` (the analytic LOSVD Fourier transform). Its license is
declared as ``Other/Proprietary License``, and reads in relevant part:

    Copyright (c) 2001-2025 Michele Cappellari

    This software is provided as is with no warranty. You may use it for
    non-commercial purposes and modify it for personal or internal use, as long
    as you include this copyright and disclaimer in all copies. **You may not
    redistribute the code.**

No copy of ``ppxf`` is included in this directory, because ``dc3`` does not
redistribute it: ``pip`` installs it from PyPI as a declared dependency. Two
consequences hold for the life of this project:

1. **Never vendor, copy, or adapt ppxf source into dc3.** Import it. A copied
   twenty-line helper would make the ``dc3`` wheel non-redistributable and
   silently void the BSD-3 license above. Writing a clean-room implementation
   from the published papers is permissible — a line-by-line translation is
   not, in any language.

2. **Expect packaging friction downstream.** conda-forge and distribution
   packagers cannot redistribute ``ppxf``, so anything that wants to ship
   ``dc3`` as a system package will need the ppxf-dependent paths to be
   optional. ``ppxf`` imports are therefore confined to
   ``dc3/core/resolution.py`` and ``dc3/core/losvd.py``.

All other dependencies (numpy, scipy, astropy, matplotlib, specutils, and the
optional ``pyfftw``, ``jax`` and MCMC extras) carry permissive licenses — BSD,
Apache-2.0, or equivalent — that impose no conditions on ``dc3``.
