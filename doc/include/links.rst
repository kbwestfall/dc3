.. Named link targets shared across the documentation.
..
.. Every module docstring ends with ``.. include:: ../include/links.rst``; the
.. path is relative to the page doing the including, and ``sphinx-apidoc``
.. writes every module's page directly into ``doc/api/``, whatever the module's
.. depth in the package.
..
.. Prefer an explicit Sphinx role resolved through intersphinx -- e.g.
.. :class:`numpy.ndarray` -- over a named link here.  Add a target here only for
.. a package that publishes no object inventory, of which ppxf is the case in
.. point.

.. _ppxf: https://pypi.org/project/ppxf/
.. _mangadap: https://sdss-mangadap.readthedocs.io/
.. _PypeIt: https://pypeit.readthedocs.io/
