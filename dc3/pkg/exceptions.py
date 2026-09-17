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
    'DC3CodingError',
    'DC3DataModelError',
    'DC3PathError',
    'DC3ParameterError',
    'DC3ResolutionError',
]


class DC3Error(Exception):
    """Base class for all dc3-specific exceptions."""
    pass


class DC3CodingError(DC3Error):
    """
    Raised for a fault in how dc3 itself is written, not in how it is used.

    This marks a condition that no user input can produce: a class declared
    incorrectly, an invariant the code is supposed to maintain, or a branch that
    should be unreachable.  Raising this type rather than
    :class:`DC3ParameterError` (or similar) keeps the two audiences apart --
    a user seeing this has found a bug to report, not a mistake to correct.
    """
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
