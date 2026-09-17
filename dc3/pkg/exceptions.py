"""
Provides dc3-specific exceptions.

.. note::

    Adapted from ``pypeit/pkg/exceptions.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.
"""

__all__ = [
    'DC3Error',
    'DC3BitMaskError',
    'DC3DataModelError',
    'DC3PathError',
    'DC3ParameterError',
    'DC3ResolutionError',
]


class DC3Error(Exception):
    """Base class for all dc3-specific exceptions."""
    pass


class DC3BitMaskError(DC3Error):
    """Raised for invalid bitmask definitions or flag operations."""
    pass


class DC3DataModelError(DC3Error):
    """Raised when data does not conform to its declared datamodel."""
    pass


class DC3PathError(DC3Error):
    """Raised for missing or invalid filesystem paths."""
    pass


class DC3ParameterError(DC3Error):
    """Raised for invalid parameter values or parameter-set definitions."""
    pass


class DC3ResolutionError(DC3Error):
    """
    Raised when a spectral-resolution operation cannot be performed as
    requested.

    This covers the cases that must never be handled silently: a requested
    preparation kernel that would require deconvolution, and a kernel dispersion
    below the floor imposed by the variable-sigma convolution.
    """
    pass
