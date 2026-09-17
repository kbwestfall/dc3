r"""
Shifting the galaxy to the approximate rest frame, by whole pixels only.

The governing rule is that **the galaxy's flux distribution is never
redistributed**.  That rules out resampling onto a shifted grid.  But on a
logarithmic wavelength grid a redshift is a pure *translation*:

.. math::

    \log_{10}\lambda_{\rm rest} = \log_{10}\lambda_{\rm obs} - \log_{10}(1+z)

so shifting by a whole number of pixels is exact.  It is a **relabelling of the
wavelength axis** -- no interpolation, no flux redistribution, and no induced
covariance between pixels.

The sub-pixel remainder is simply left in place.  It is absorbed by the fitted
velocity, which is being solved for anyway, so nothing is lost by declining to
chase it.

What is actually done
---------------------

.. code-block:: text

    n_shift   = round(log10(1 + z_guess) / dloglam)      # integer
    z_applied = 10**(n_shift * dloglam) - 1              # the redshift removed
    V_obs     = V_fit + c ln(1 + z_applied)              # propagated analytically

The last line is exact *because* the internal velocity convention is
:math:`c\ln(1+z)`, which is the one convention that is additive on a logarithmic
grid; see :mod:`~dc3.core.velocity`.  The velocity removed and the velocity
fitted simply add.

.. note::

    **No data moves, and nothing is truncated.**  Shifting by an integer number
    of pixels could be implemented either by rolling the arrays and keeping the
    grid, or by keeping the arrays and relabelling the grid.  The two are
    equivalent, except that rolling discards ``n_shift`` pixels off one end.
    Relabelling is therefore strictly better, and it is what is done here: only
    :attr:`~dc3.spectra.Spectra.log10lam0` changes.

    This also disposes of the instrumental dispersion automatically.  In
    velocity units the dispersion is independent of redshift for each pixel, so
    it must move with the flux and must *not* be rescaled by :math:`(1+z)`.
    Since neither array moves, they stay in correspondence with no work.

Why bother
----------

The region-mask transcription by :math:`(1+z)^{\pm 1}` largely disappears, the
fit window and the initial guess become nearly independent of the configuration,
and the convention lines up with ``ppxf``, which simplifies the cross-check.

.. warning::

    There is deliberately **no interpolating alternative**.  One would violate
    the rule above and introduce exactly the inter-pixel covariance that ``dc3``
    has decided to ignore everywhere else.

.. include:: ../include/links.rst
"""

import numpy as np

from ..pkg.exceptions import DC3Error
from .velocity import log_velocity, redshift_from_log_velocity


__all__ = ['DeRedshift', 'pixel_shift', 'to_rest_frame']


class DeRedshift:
    r"""
    The whole-pixel shift that takes a spectrum to the approximate rest frame.

    Returned by :func:`pixel_shift`.  It records what was removed, so that a
    fitted velocity can always be referred back to the observed frame.

    Parameters
    ----------
    z_guess : float
        The redshift the caller asked to remove.
    n_shift : int
        The whole number of pixels actually shifted.
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.

    Attributes
    ----------
    z_guess : float
        As above.
    n_shift : int
        As above.
    dloglam : float
        As above.
    """

    def __init__(self, z_guess, n_shift, dloglam):
        self.z_guess = float(z_guess)
        self.n_shift = int(n_shift)
        self.dloglam = float(dloglam)

    @property
    def z_applied(self):
        """The redshift actually removed, which the whole-pixel shift fixes exactly."""
        return float(np.power(10.0, self.n_shift * self.dloglam) - 1.0)

    @property
    def velocity_applied(self):
        """The velocity removed, in km/s, in the logarithmic convention."""
        return float(log_velocity(self.z_applied))

    @property
    def velocity_guess(self):
        """The velocity the caller asked to remove, in km/s."""
        return float(log_velocity(self.z_guess))

    @property
    def residual_velocity(self):
        """
        The sub-pixel remainder left for the fit to absorb, in km/s.

        Bounded by half a pixel in magnitude, since the shift is the *nearest*
        whole number of pixels.  It is not an error: the fitted velocity
        measures the total displacement from the relabelled grid, so this is
        recovered rather than lost.
        """
        return self.velocity_guess - self.velocity_applied

    def observed_velocity(self, velocity_fit):
        r"""
        Refer a fitted velocity back to the observed frame.

        Applies :math:`V_{\rm obs} = V_{\rm fit} + c\ln(1+z_{\rm applied})`,
        which is exact because the logarithmic velocity convention is additive.

        Parameters
        ----------
        velocity_fit : float, :class:`numpy.ndarray`
            Velocity measured against the de-redshifted spectrum, in km/s.

        Returns
        -------
        float, :class:`numpy.ndarray`
            The velocity in the observed frame, in km/s.
        """
        return np.asarray(velocity_fit, dtype=float) + self.velocity_applied

    def observed_redshift(self, velocity_fit):
        """
        Refer a fitted velocity back to a redshift in the observed frame.

        Parameters
        ----------
        velocity_fit : float, :class:`numpy.ndarray`
            Velocity measured against the de-redshifted spectrum, in km/s.

        Returns
        -------
        float, :class:`numpy.ndarray`
            The total redshift.
        """
        return redshift_from_log_velocity(self.observed_velocity(velocity_fit))

    def __repr__(self):
        """A short summary of the shift."""
        return (
            f'<{type(self).__name__}: {self.n_shift:+d} pixels, z_applied='
            f'{self.z_applied:.6f} ({self.velocity_applied:+.1f} km/s), '
            f'residual {self.residual_velocity:+.2f} km/s>'
        )


def pixel_shift(z_guess, dloglam):
    r"""
    Determine the whole-pixel shift that best removes a redshift.

    Parameters
    ----------
    z_guess : float
        The redshift to remove.  May be negative, for a blueshift.
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.

    Returns
    -------
    DeRedshift
        The shift, and what it leaves behind.

    Raises
    ------
    DC3Error
        Raised if the redshift is not greater than -1, or the pixel size is not
        positive.

    Notes
    -----
    The shift is a property of the *grid*, so one value serves every spectrum
    sharing it.  A per-spectrum shift is not offered: each spectrum would then
    need its own wavelength grid, which is precisely the structure
    :class:`~dc3.spectra.Spectra` exists to avoid, and the per-spectrum
    differences are what the fitted velocity measures anyway.
    """
    if z_guess <= -1:
        raise DC3Error(f'A redshift must be greater than -1; got {z_guess}.')
    if dloglam <= 0:
        raise DC3Error(f'The logarithmic pixel size must be positive; got {dloglam}.')
    n_shift = int(np.round(np.log10(1.0 + z_guess) / dloglam))
    return DeRedshift(z_guess, n_shift, dloglam)


def to_rest_frame(spectra, shift):
    r"""
    Return a copy of a spectrum set relabelled into the approximate rest frame.

    Only :attr:`~dc3.spectra.Spectra.log10lam0` changes.  The flux, inverse
    variance, mask and instrumental dispersion are carried across untouched,
    which is the whole point: a whole-pixel shift on a logarithmic grid is a
    relabelling, not a resampling.

    Parameters
    ----------
    spectra : :class:`~dc3.spectra.Spectra`
        The observed-frame spectra.
    shift : DeRedshift
        The shift to apply, from :func:`pixel_shift`.

    Returns
    -------
    :class:`~dc3.spectra.Spectra`
        The relabelled spectra.

    Raises
    ------
    DC3Error
        Raised if the shift was computed for a different pixel size, which
        would silently move the spectrum to the wrong place.
    """
    if not np.isclose(shift.dloglam, spectra.dloglam, rtol=1e-12):
        raise DC3Error(
            f'The shift was computed for a pixel size of {shift.dloglam}, but the spectra are '
            f'sampled at {spectra.dloglam}.  Recompute the shift for this grid; applying it as '
            'it stands would move the spectra to the wrong rest wavelengths.'
        )
    out = spectra.copy()
    out.log10lam0 -= shift.n_shift * shift.dloglam
    return out
