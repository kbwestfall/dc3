r"""
Line profiles and the measurement of line widths.

These are the tools for asking what a processing step does to a line-spread
function: build a spectrum of lines of known width, pass it through the step,
and measure the widths that come out.

Which width is measured matters as much as how
----------------------------------------------

A spectrograph records the line-spread function *integrated over each pixel*.
Two widths can be quoted for it, and they differ at the level that matters
here:

- The **post-pixelized** width is what a Gaussian evaluated at the pixel
  centres returns when fit to the recorded data.  It is the more direct
  measurement, but it folds the pixel response into the width.
- The **pre-pixelized** width is the dispersion of the line-spread function
  *before* integration over the pixel.  It is what a Gaussian integrated over
  each pixel returns when fit to the same data, and it is the quantity ``dc3``
  assumes every instrumental dispersion describes.

For a critically sampled spectrum the two differ by 1-10%: Law et al. (2021,
AJ 161, 52; Section 3.2 and Figure 4) find the post-pixelized width broader by
about 4% at one pixel and by about 8% at 0.8 pixels.  To second order the
difference is the variance of the pixel top-hat, :math:`\sigma_{\rm post}^2
\approx \sigma_{\rm pre}^2 + \Delta^2/12`.  Since ``dc3`` works with the
pre-pixelized width throughout, the widths measured here are pre-pixelized:
:func:`fit_line` fits :func:`pixelated_gaussian` by default.  Measuring the
post-pixelized width instead would build the very bias under study into the
instrument used to study it.

All coordinates are in **pixels of the grid the spectrum is sampled on**, with
pixel :math:`i` centred at :math:`x = i` and spanning :math:`[i - 1/2, i +
1/2]`.  On a logarithmic wavelength grid a Gaussian in velocity is a Gaussian
in pixel index, so a width in pixels converts to km/s by multiplying by the
velocity scale.

.. note::

    :func:`pixelated_gaussian` follows ``pixelated_gaussian`` in
    ``mangadap/tests/test_lineprofiles.py`` (BSD 3-Clause); see
    ``licenses/README.rst``.

.. include:: ../include/links.rst
"""

import numpy as np
from scipy import optimize, special

from ..pkg.exceptions import DC3Error


__all__ = [
    'fit_line',
    'gaussian_comb',
    'measure_comb',
    'pixelated_gaussian',
    'sampled_gaussian',
]


def pixelated_gaussian(x, center=0.0, sigma=1.0, flux=1.0):
    r"""
    Return a Gaussian integrated over unit pixels.

    Pixel :math:`i` receives the mean of the profile over :math:`[x_i - 1/2,
    x_i + 1/2]`,

    .. math::

        f_i = \frac{F}{2}\left[
            {\rm erf}\left(\frac{x_i - c + 1/2}{\sqrt{2}\sigma}\right)
          - {\rm erf}\left(\frac{x_i - c - 1/2}{\sqrt{2}\sigma}\right)
        \right],

    which is what a detector records of a Gaussian line-spread function of
    dispersion :math:`\sigma`.  :math:`\sigma` is therefore the
    **pre-pixelized** width.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Pixel coordinates at which to evaluate the profile.
    center : float, optional
        Centre of the line, in pixels.
    sigma : float, optional
        Pre-pixelized dispersion, in pixels.
    flux : float, optional
        Integrated flux of the line.

    Returns
    -------
    :class:`numpy.ndarray`
        The profile at ``x``.
    """
    d = np.asarray(x, dtype=float) - center
    n = np.sqrt(2.0) * sigma
    return flux * (special.erf((d + 0.5) / n) - special.erf((d - 0.5) / n)) / 2.0


def sampled_gaussian(x, center=0.0, sigma=1.0, flux=1.0):
    r"""
    Return a Gaussian evaluated at the pixel centres.

    The model a Gaussian fit that ignores pixel integration assumes.  Fit to
    data recorded by a detector, its :math:`\sigma` is the **post-pixelized**
    width.  Provided to show what that choice gets wrong, not for measurement.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Pixel coordinates at which to evaluate the profile.
    center : float, optional
        Centre of the line, in pixels.
    sigma : float, optional
        Dispersion, in pixels.
    flux : float, optional
        Integrated flux of the line.

    Returns
    -------
    :class:`numpy.ndarray`
        The profile at ``x``.
    """
    d = np.asarray(x, dtype=float) - center
    return flux * np.exp(-0.5 * np.square(d / sigma)) / (np.sqrt(2.0 * np.pi) * sigma)


def gaussian_comb(npix, centers, sigma, flux=1.0):
    """
    Return a spectrum of pixel-integrated Gaussian lines on a zero continuum.

    Each line is evaluated only within ten dispersions of its centre, which is
    exact to double precision and keeps a long comb cheap.

    Parameters
    ----------
    npix : int
        Number of pixels in the spectrum.
    centers : :class:`numpy.ndarray`
        Line centres, in pixels.
    sigma : float, :class:`numpy.ndarray`
        Pre-pixelized dispersion of each line, in pixels; one value for all
        lines, or one per line.
    flux : float, optional
        Integrated flux of every line.

    Returns
    -------
    :class:`numpy.ndarray`
        The spectrum.
    """
    _centers = np.atleast_1d(np.asarray(centers, dtype=float))
    _sigma = np.broadcast_to(np.asarray(sigma, dtype=float), _centers.shape)
    x = np.arange(npix, dtype=float)
    spec = np.zeros(npix, dtype=float)
    for c, s in zip(_centers, _sigma):
        lo = max(0, int(np.floor(c - 10 * s)) - 1)
        hi = min(npix, int(np.ceil(c + 10 * s)) + 2)
        spec[lo:hi] += pixelated_gaussian(x[lo:hi], center=c, sigma=s, flux=flux)
    return spec


def fit_line(y, center, sigma, pixelated=True, window=None, fix_center=False):
    """
    Fit a single Gaussian line and return its flux, centre and width.

    The fit is to noise-free data on a zero continuum, over a window centred on
    the pixel nearest the guessed position.  No continuum term is fit, since one
    would only add a degeneracy the data cannot break.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        The spectrum, sampled at pixel coordinates ``numpy.arange(y.size)``.
    center : float
        Initial guess for the line centre, in pixels, or its fixed value if
        ``fix_center`` is True.
    sigma : float
        Initial guess for the dispersion, in pixels.
    pixelated : bool, optional
        Fit :func:`pixelated_gaussian`, returning the **pre-pixelized** width.
        If False, fit :func:`sampled_gaussian`, returning the
        **post-pixelized** width.
    window : int, optional
        Size of the fitting window, in pixels.  The window extends
        ``window // 2`` pixels either side of the pixel nearest ``center``, so
        it always spans an odd number of pixels, ``2 * (window // 2) + 1``.
        Defaults to sixteen times the guessed dispersion, and never fewer than
        eleven pixels.
    fix_center : bool, optional
        Hold the centre at ``center`` and fit only the flux and dispersion.
        Useful where the centre is known exactly, as for a synthetic line,
        since it removes a parameter the width is correlated with.

    Returns
    -------
    tuple
        The fitted flux, centre and dispersion.  With ``fix_center`` the
        returned centre is the input value.

    Raises
    ------
    DC3Error
        Raised if the window extends beyond the spectrum, or if the fit fails.
    """
    _window = max(int(np.ceil(16.0 * sigma)), 11) if window is None else int(window)
    halfwidth = _window // 2
    mid = int(np.round(center))
    lo = mid - halfwidth
    hi = mid + halfwidth + 1
    if lo < 0 or hi > y.size:
        raise DC3Error(
            f'The fitting window [{lo}, {hi}) for the line at pixel {center:.1f} extends beyond '
            f'the spectrum of {y.size} pixels.'
        )
    x = np.arange(lo, hi, dtype=float)
    data = y[lo:hi]
    model = pixelated_gaussian if pixelated else sampled_gaussian
    flux_guess = np.sum(data)
    flux_scale = max(abs(flux_guess), 1e-12)
    tol = dict(xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if fix_center:
        result = optimize.least_squares(
            _fixed_center_residuals, [flux_guess, sigma], args=(x, data, model, center),
            bounds=([-np.inf, 1e-3 * sigma], [np.inf, 1e3 * sigma]),
            x_scale=[flux_scale, sigma], **tol
        )
        par = (result.x[0], center, result.x[1])
    else:
        result = optimize.least_squares(
            _free_center_residuals, [flux_guess, center, sigma], args=(x, data, model),
            bounds=([-np.inf, lo, 1e-3 * sigma], [np.inf, hi, 1e3 * sigma]),
            x_scale=[flux_scale, 1.0, sigma], **tol
        )
        par = tuple(result.x)
    if not result.success:
        raise DC3Error(f'The fit to the line at pixel {center:.1f} failed: {result.message}')
    return tuple(float(p) for p in par)


def _free_center_residuals(par, x, data, model):
    """The residuals of a single-line model with a free centre; see :func:`fit_line`."""
    return model(x, center=par[1], sigma=par[2], flux=par[0]) - data


def _fixed_center_residuals(par, x, data, model, center):
    """The residuals of a single-line model with a fixed centre; see :func:`fit_line`."""
    return model(x, center=center, sigma=par[1], flux=par[0]) - data


def measure_comb(y, centers, sigma, pixelated=True, window=None, fix_center=False):
    """
    Fit every line in a comb and return the fitted centres and widths.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        The spectrum.
    centers : :class:`numpy.ndarray`
        Initial guesses for the line centres, in pixels, or their fixed values
        if ``fix_center`` is True.
    sigma : float, :class:`numpy.ndarray`
        Initial guesses for the dispersions, in pixels; one value, or one per
        line.
    pixelated : bool, optional
        Measure the pre-pixelized width; see :func:`fit_line`.
    window : int, optional
        Size of each fitting window, in pixels; see :func:`fit_line`.
    fix_center : bool, optional
        Hold each centre at its input value; see :func:`fit_line`.

    Returns
    -------
    tuple
        Two :class:`numpy.ndarray` objects: the fitted centres and the fitted
        dispersions, in pixels.
    """
    _centers = np.atleast_1d(np.asarray(centers, dtype=float))
    _sigma = np.broadcast_to(np.asarray(sigma, dtype=float), _centers.shape)
    fits = np.array([
        fit_line(y, c, s, pixelated=pixelated, window=window, fix_center=fix_center)
        for c, s in zip(_centers, _sigma)
    ])
    return fits[:, 1], fits[:, 2]
