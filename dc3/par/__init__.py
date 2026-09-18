"""
The machinery for declaring runtime parameters.

The parameter sets themselves are **not** here.  Each is declared in the module
whose code it configures -- :class:`~dc3.templates.TemplatePar` in
:mod:`dc3.templates`, and so on -- leaving :mod:`dc3.par.dc3par` with the
top-level :class:`~dc3.par.dc3par.DC3Par` that nests them.

.. warning::

    **Never re-export a concrete parameter set here.**  Those modules import
    :mod:`dc3.par.parset`, which runs this file; were this file to import them
    back, the cycle would close.  It would not fail on ``import dc3.par``, which
    is the import that looks suspicious -- it fails on ``import dc3.templates``,
    reporting a partially initialized module and pointing at the parameter set
    rather than at this line.  Keeping this module to the machinery, which
    imports nothing from the rest of the package, is what makes the arrangement
    safe.  See ``dc3.tests.test_dc3par.test_consumer_modules_import_alone``.
"""

from .parset import ParSet
from .funcpar import FuncPar

__all__ = [
    'FuncPar',
    'ParSet',
]
