.. _templatedata:

========================
dc3 template libraries
========================

This directory holds stellar template libraries, and the configuration files
that define them.

**Library spectra are not shipped in the wheel.** They are fetched on demand
through the package cache (:mod:`dc3.pkg.cache`); see
:class:`~dc3.pkg.dc3data.DC3DataPaths`. This README is version-controlled so
that the directory always exists, even when it is otherwise empty.

A library is defined by a key in a configuration file, and is processed once per
run through the two-step template-preparation pipeline — resolution matching to
a fiducial galaxy resolution, then resampling at an integer ``velscale_ratio``.
The *prepared* product is cached separately, keyed on everything that changes
it: the library, the fiducial galaxy resolution, the velocity scale,
``velscale_ratio``, ``epsilon_sigma`` and ``sigma_floor``.
