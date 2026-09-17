.. _testdata:

=================
dc3 test fixtures
=================

This directory holds the data fixtures used by the ``dc3`` test suite.

**The bulk data in this directory is not shipped in the wheel.** ``MANIFEST.in``
excludes it *by file type* (``*.fits``, ``*.gz``, ``*.npz``, ``*.db``), which is
deliberate: this README and ``PROVENANCE.toml`` are not stripped, so the
provenance of the fixtures remains readable even from an installed package that
contains none of them. Missing files are fetched through the package cache
(:mod:`dc3.pkg.cache`), so a clean checkout with no external drive mounted still
runs the full suite.

Fixtures are staged from their original locations by ``dc3_stage_testdata``,
which is run **once**, by hand, with the source path as an argument. Nothing in
the package reads the original source paths at run time.

``PROVENANCE.toml`` records, for every staged file: the source path, the source
modification time, its size, its SHA-256, the rows retained if the file was
trimmed, and the staging date.


Provenance, and how far to trust these files
--------------------------------------------

Fixtures fall into two classes, and the distinction governs how they may be
used in tests.

**Inputs are version-independent and fully trustworthy.** Galaxy spectra,
templates, instrumental-dispersion vectors, mask tables and parameter files are
*data*, not code output. They are the main value of this collection: realistic,
non-synthetic, correctly formatted inputs at the right sampling.

**Outputs are indicative only.** The archived C++ outputs were *not* produced by
the version of the C++ source that survives. The production campaign ran in
February 2013, but the surviving sources were modified between 2013 and 2016,
and none of the three mutually inconsistent versions can be rebuilt. The
outputs therefore pin down file formats and orders of magnitude, and make a good
smoke test, but a numerical disagreement cannot be attributed to the port
without first establishing which build wrote the reference.

.. warning::

    **Do not gate CI on tight tolerances against the archived outputs.**
    Comparisons against them are reported, not asserted. Tight tolerances belong
    to the self-consistency identities and to the *published* numbers of
    Westfall et al. (2011), which are build-independent.


Fixture sets
------------

Populated by ``dc3_stage_testdata``; see ``PROVENANCE.toml`` for what is
actually present.

PPak, UGC 6918 and UGC 448 (February 2013)
    A production campaign providing real inputs and the corresponding C++
    outputs. Note these are **PPak** data with template HR 6654 — *not* the
    SparsePak/HD 167042 configuration of Westfall et al. (2011) §5. Do not
    present agreement here as reproducing the paper's Figures 7–9.

    The instrumental-dispersion (``_cd``) products are doubly untrustworthy —
    the provenance problem above, plus an argument-indexing bug in the driver
    that silently swapped two inputs. Stage them as inputs only, never as
    references.

SparsePak, UGC 6918 (November 2008 – January 2009)
    The demonstration data behind Westfall et al. (2011) §5 — Figures 7–9 and
    Table 1 — with template HR 6817 (= HD 167042). **The highest-value fixture
    set in the project**, because its results can be checked against *published*
    numbers rather than against an unidentified build, and can therefore carry
    tight tolerances.

    Caveat: this run used the interactive ``DC3`` binary, a much earlier code
    path than ``DC3_express``, with an entirely different output layout. The
    *inputs* and the *published numbers* are the valuable parts; the
    intermediate tables need their formats worked out before they can be used,
    and the provenance caution above applies to them at least as strongly.


.. note::

    The port plan calls this provenance file ``MANIFEST.toml``. It is named
    ``PROVENANCE.toml`` here to avoid colliding with the unrelated
    ``MANIFEST.in`` at the repository root, which is a setuptools directive file
    in a different format serving a different purpose.
