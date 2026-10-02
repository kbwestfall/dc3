.. include:: include/links.rst

===
DC3
===

**Detector-Censored Cross-Correlation**: stellar kinematics (:math:`V`,
:math:`\sigma`) from galaxy-continuum spectra.

``dc3`` cross-correlates a galaxy spectrum :math:`G` with a stellar template
:math:`T` and fits the resulting cross-correlation function :math:`X = G \circ
T` with the model :math:`X_T = (T \otimes B)' \circ T`, where :math:`B` is a
parameterized broadening function.

The method's distinguishing feature is that the broadened template is
**detector-censored in exactly the same way as the galaxy spectrum** before
correlation: the same masking, truncation and apodization are applied to both.
It deliberately does *not* use the commutation :math:`X_T = (T \circ T) \otimes
B`, which is exact only for untruncated, unmasked data and introduces a
systematic error of order 10% in :math:`\sigma` on real spectra.

The algorithm is described in `Westfall, Bershady & Verheijen (2011, ApJS 193,
21) <https://ui.adsabs.harvard.edu/abs/2011ApJS..193...21W/abstract>`__, Paper
III of the DiskMass Survey.  This package is a Python reimplementation of the
original C++ code, which is not distributed.

.. warning::

    ``dc3`` is under active development and is not yet usable for science.
    Interfaces will change without notice.

----

.. toctree::
   :caption: Development
   :maxdepth: 2

   dev/index

.. toctree::
   :caption: Reference
   :maxdepth: 1

   API <api/dc3>
   whatsnew
