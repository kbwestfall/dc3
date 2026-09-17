r"""
Conversions between velocity and redshift.

Three velocities, and which one is meant
----------------------------------------

Three different quantities are all called "velocity" in this context, and
conflating them is a recurring source of error -- the original C++ carried an
unresolved to-do about exactly this.  They are given separate names here rather
than a flag, so that a call site says which is meant:

:func:`log_velocity`
    :math:`V = c \ln(1+z)`.  **The convention used internally**, and the one
    ``ppxf`` reports.  It is the only one of the three that is *additive* on a
    logarithmic grid, so that composing two shifts adds their velocities.  This
    is what makes de-redshifting separable from fitting: the velocity removed
    and the velocity fitted simply add.

:func:`relativistic_velocity`
    :math:`V = c\,[(1+z)^2-1]/[(1+z)^2+1]`.  The special-relativistic Doppler
    velocity.

:func:`classical_velocity`
    :math:`V = cz`.

.. todo::

    **Revisit whether the relativistic conversions belong here at all.**

    Two distinct things get conflated under the word "redshift", and the
    relativistic form is appropriate to neither of the cases ``dc3`` actually
    meets:

    - A galaxy's **systemic redshift** is predominantly *cosmological*, which is
      metric expansion rather than motion through space.  The
      special-relativistic Doppler formula does not describe it, so applying it
      is not a refinement but a category error.
    - The **stellar kinematics** ``dc3`` measures are peculiar velocities of
      order a few hundred km/s, where :math:`\beta \sim 10^{-3}` and the
      relativistic correction is of order :math:`\beta^2/2 \sim 5\times10^{-7}`
      -- sub-m/s, and far below any measurement uncertainty here.

    So ``dc3`` will rarely if ever operate outside the classical regime, and the
    relativistic functions may be removed.  They are retained for now, unused on
    any internal path, pending that decision.

.. include:: ../include/links.rst
"""

import numpy as np
from astropy import constants


__all__ = [
    'SPEED_OF_LIGHT',
    'classical_velocity',
    'log_velocity',
    'redshift_from_classical_velocity',
    'redshift_from_log_velocity',
    'redshift_from_relativistic_velocity',
    'relativistic_velocity',
]


SPEED_OF_LIGHT = constants.c.to('km/s').value
"""Speed of light in km/s, the unit used for every velocity in ``dc3``."""


def log_velocity(z):
    r"""
    Return the velocity that shifts a logarithmic grid by redshift ``z``.

    This is :math:`V = c \ln(1+z)`, the convention used internally and by
    ``ppxf``.  It is additive: shifting by ``z1`` and then ``z2`` gives the sum
    of the two velocities.

    Parameters
    ----------
    z : float, :class:`numpy.ndarray`
        Redshift.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Velocity in km/s.
    """
    return SPEED_OF_LIGHT * np.log(1 + np.asarray(z, dtype=float))


def redshift_from_log_velocity(v):
    """
    Return the redshift corresponding to a logarithmic-grid velocity.

    The inverse of :func:`log_velocity`.

    Parameters
    ----------
    v : float, :class:`numpy.ndarray`
        Velocity in km/s.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Redshift.
    """
    return np.expm1(np.asarray(v, dtype=float) / SPEED_OF_LIGHT)


def relativistic_velocity(z):
    r"""
    Return the special-relativistic Doppler velocity for a redshift.

    This is :math:`V = c\,[(1+z)^2-1]/[(1+z)^2+1]`.

    .. warning::

        Not used on any internal path, and under review; see the module
        documentation.  A cosmological redshift is not a Doppler shift, and the
        peculiar velocities ``dc3`` measures are far too small for the
        relativistic correction to matter.

    Parameters
    ----------
    z : float, :class:`numpy.ndarray`
        Redshift.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Velocity in km/s.
    """
    opz2 = (1 + np.asarray(z, dtype=float)) ** 2
    return SPEED_OF_LIGHT * (opz2 - 1) / (opz2 + 1)


def redshift_from_relativistic_velocity(v):
    """
    Return the redshift corresponding to a special-relativistic velocity.

    The inverse of :func:`relativistic_velocity`.

    .. warning::

        Not used on any internal path, and under review; see the module
        documentation.

    Parameters
    ----------
    v : float, :class:`numpy.ndarray`
        Velocity in km/s.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Redshift.
    """
    beta = np.asarray(v, dtype=float) / SPEED_OF_LIGHT
    return np.sqrt((1 + beta) / (1 - beta)) - 1


def classical_velocity(z):
    """
    Return the classical velocity :math:`V = cz`.

    Parameters
    ----------
    z : float, :class:`numpy.ndarray`
        Redshift.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Velocity in km/s.
    """
    return SPEED_OF_LIGHT * np.asarray(z, dtype=float)


def redshift_from_classical_velocity(v):
    """
    Return the redshift corresponding to a classical velocity.

    The inverse of :func:`classical_velocity`.

    Parameters
    ----------
    v : float, :class:`numpy.ndarray`
        Velocity in km/s.

    Returns
    -------
    float, :class:`numpy.ndarray`
        Redshift.
    """
    return np.asarray(v, dtype=float) / SPEED_OF_LIGHT
