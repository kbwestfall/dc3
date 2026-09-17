r"""
Spectral sampling: the logarithmic wavelength grid, and resampling onto it.

``dc3`` works throughout on a grid uniformly sampled in
:math:`\log_{10}\lambda`.  That is not a convenience: a Doppler shift is a pure
*translation* on such a grid, which is what lets the galaxy be de-redshifted by
an integer pixel shift (:mod:`~dc3.core.deredshift`) and the model be shifted by
a phase ramp in Fourier space, neither of which redistributes flux.  Velocity
and redshift conversions are in :mod:`~dc3.core.velocity`.

.. note::

    :class:`Resample` and the grid helpers that support it are adapted from
    ``mangadap/util/sampling.py`` (BSD 3-Clause); see ``licenses/README.rst``.
    The same code also exists in ``pypeit/core/sampling.py``, but with the
    covariance support commented out for want of a ``Covariance`` class, so the
    ``mangadap`` version is the more complete one and is what is adopted here.

    The substantive change is that the covariance is computed with
    :class:`astropy.nddata.Covariance` rather than ``mangadap``'s own class,
    which has since been upstreamed into ``astropy`` and generalized.  Three
    call sites differ as a result: ``variance()`` became a property, ``full()``
    became ``to_dense()``, and ``impose_triu`` became ``assume_symmetric``.

.. include:: ../include/links.rst
"""

import warnings

import numpy as np
from astropy.nddata import Covariance
from scipy import interpolate

from ..pkg.exceptions import DC3Error
from .velocity import SPEED_OF_LIGHT


__all__ = [
    'Resample',
    'borders_to_centers',
    'centers_to_borders',
    'dloglam_from_velscale',
    'grid_borders',
    'grid_centers',
    'grid_from_wave',
    'grid_npix',
    'log_wavelength_grid',
    'velscale',
]


# ----------------------------------------------------------------------
# The dc3 logarithmic grid
# ----------------------------------------------------------------------
def velscale(dloglam):
    r"""
    Return the velocity scale of a logarithmic grid.

    A grid uniform in :math:`\log_{10}\lambda` is uniform in velocity, with

    .. math::

        \Delta v = c \ln(10)\, \Delta \log_{10}\lambda .

    Parameters
    ----------
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.

    Returns
    -------
    float
        Velocity scale in km/s per pixel.
    """
    return SPEED_OF_LIGHT * np.log(10.0) * dloglam


def dloglam_from_velscale(velocity_scale):
    r"""
    Return the logarithmic pixel size giving a velocity scale.

    The inverse of :func:`velscale`.

    Parameters
    ----------
    velocity_scale : float
        Velocity scale in km/s per pixel.

    Returns
    -------
    float
        Pixel size in :math:`\log_{10}\lambda`.
    """
    return velocity_scale / (SPEED_OF_LIGHT * np.log(10.0))


def log_wavelength_grid(log10lam0, dloglam, npix):
    r"""
    Construct a logarithmically sampled wavelength grid.

    Parameters
    ----------
    log10lam0 : float
        :math:`\log_{10}` of the wavelength of the first pixel's centre.
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.
    npix : int
        Number of pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The wavelengths of the pixel centres.
    """
    return np.power(10.0, log10lam0 + dloglam * np.arange(npix, dtype=float))


def grid_from_wave(wave, rtol=1e-6):
    r"""
    Recover the grid parameters from a wavelength vector, checking its sampling.

    Parameters
    ----------
    wave : :class:`numpy.ndarray`
        Wavelengths of the pixel centres, in ascending order.
    rtol : float, optional
        Relative tolerance on the uniformity of the logarithmic sampling.  The
        test compares the spread of the pixel sizes against their mean.

    Returns
    -------
    tuple
        ``log10lam0`` and ``dloglam``.

    Raises
    ------
    DC3Error
        Raised if the vector is too short, is not ascending and positive, or is
        not uniformly sampled in the logarithm of the wavelength.  ``dc3``
        assumes logarithmic sampling everywhere, so a linear grid is rejected
        here rather than producing subtly wrong velocities later; use
        :class:`Resample` to convert one first.
    """
    _wave = np.atleast_1d(np.asarray(wave, dtype=float))
    if _wave.size < 2:
        raise DC3Error('A wavelength grid must have at least two pixels.')
    if np.any(_wave <= 0):
        raise DC3Error('Wavelengths must be positive.')
    if np.any(np.diff(_wave) <= 0):
        raise DC3Error('Wavelengths must be in ascending order.')

    loglam = np.log10(_wave)
    steps = np.diff(loglam)
    dloglam = float(np.mean(steps))
    if np.any(np.absolute(steps - dloglam) > rtol * dloglam):
        raise DC3Error(
            'Wavelength vector is not uniformly sampled in log10(wavelength).  dc3 requires a '
            'logarithmic grid: a Doppler shift is then a pure translation, which is what allows '
            'the galaxy to be de-redshifted without redistributing its flux.  Resample the '
            'spectrum onto a logarithmic grid before ingesting it.'
        )
    return float(loglam[0]), dloglam


# ======================================================================
# Adapted from mangadap/util/sampling.py; see the module documentation.
# ======================================================================
def grid_npix(rng=None, dx=None, log=False, base=10.0, default=None):
    """
    Determine the number of pixels needed for a given grid.

    Parameters
    ----------
    rng : array-like, optional
        Two-element array with the starting and ending coordinate of the pixel
        centres.  If ``log`` is True this is still the linear coordinate, not
        its logarithm.
    dx : float, optional
        Linear or logarithmic pixel width.
    log : bool, optional
        Bin the range logarithmically.
    base : float, optional
        Base of the logarithm.
    default : int, optional
        Number of pixels returned if either ``rng`` or ``dx`` is not provided.

    Returns
    -------
    tuple
        The number of pixels covering ``rng`` with pixels of width ``dx``, and
        the range adjusted so that the number of pixels is an exact integer.

    Raises
    ------
    DC3Error
        Raised if the range is not a two-element vector.
    """
    if rng is None or dx is None:
        return default, rng
    if len(rng) != 2:
        raise DC3Error('Range must be a 2-element vector.')

    _rng = np.atleast_1d(rng).copy().astype(float)
    npix = (
        int(np.floor(np.diff(np.log(_rng))[0] / np.log(base) / dx) + 1) if log
        else int(np.floor(np.diff(_rng)[0] / dx) + 1)
    )
    _rng[1] = (
        np.power(base, np.log(_rng[0]) / np.log(base) + dx * (npix - 1)) if log
        else _rng[0] + dx * (npix - 1)
    )

    # Guard against numerical precision losing the last pixel
    if (
        (not log and np.isclose(rng[1] - _rng[1], dx))
        or (log and np.isclose((np.log(rng[1]) - np.log(_rng[1])) / np.log(base), dx))
    ):
        npix += 1
        _rng[1] = (
            np.power(base, np.log(_rng[0]) / np.log(base) + dx * (npix - 1)) if log
            else _rng[0] + dx * (npix - 1)
        )

    return npix, _rng


def grid_borders(rng, npix, log=False, base=10.0):
    """
    Determine the bin edges of a grid.

    Parameters
    ----------
    rng : array-like
        Two-element array with the (geometric) centres of the first and last
        pixel.
    npix : int
        Number of pixels.
    log : bool, optional
        The range is logarithmically sampled.
    base : float, optional
        Base of the logarithmic sampling.

    Returns
    -------
    tuple
        The grid borders, of shape ``(npix+1,)``, and the step per grid point.
    """
    if log:
        _rng = np.log(rng) / np.log(base)
        dlogx = np.diff(_rng)[0] / (npix - 1.)
        borders = np.power(base, np.linspace(*(_rng / dlogx + [-0.5, 0.5]), num=npix + 1) * dlogx)
        return borders, dlogx
    dx = np.diff(rng)[0] / (npix - 1.)
    borders = np.linspace(*(np.atleast_1d(rng) / dx + np.array([-0.5, 0.5])), num=npix + 1) * dx
    return borders, dx


def grid_centers(rng, npix, log=False, base=10.0):
    """
    Determine the (geometric) centres of the pixels in a grid.

    Parameters
    ----------
    rng : array-like
        Two-element array with the (geometric) centres of the first and last
        pixel.
    npix : int
        Number of pixels.
    log : bool, optional
        The range is logarithmically sampled.
    base : float, optional
        Base of the logarithmic sampling.

    Returns
    -------
    tuple
        The pixel centres, of shape ``(npix,)``, and the step per grid point.
    """
    if log:
        _rng = np.log(rng) / np.log(base)
        dlogx = np.diff(_rng)[0] / (npix - 1.)
        centers = np.power(base, np.linspace(*(_rng / dlogx), num=npix) * dlogx)
        return centers, dlogx
    dx = np.diff(rng)[0] / (npix - 1.)
    centers = np.linspace(*(np.atleast_1d(rng) / dx), num=npix) * dx
    return centers, dx


def borders_to_centers(borders, log=False):
    """
    Convert a set of bin borders to bin centres.

    The borders need not be regularly spaced.

    Parameters
    ----------
    borders : :class:`numpy.ndarray`
        Borders of adjoining bins.
    log : bool, optional
        Return the geometric centre rather than the linear centre.

    Returns
    -------
    :class:`numpy.ndarray`
        The bin centres.
    """
    return np.sqrt(borders[:-1] * borders[1:]) if log else (borders[:-1] + borders[1:]) / 2.0


def centers_to_borders(x, log=False):
    """
    Convert a set of bin centres to bounding edges.

    The centres need not be regularly spaced.  The outer edges of the first and
    last bins are placed so that those bins are as wide as their neighbours.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        Centres of adjoining bins.
    log : bool, optional
        Adopt geometric rather than linear binning.

    Returns
    -------
    :class:`numpy.ndarray`
        The coordinates of the adjoining bin edges.
    """
    if log:
        dx = np.diff(np.log(x))
        return np.exp(
            np.append(np.log(x[:-1]) - dx / 2, np.log(x[-1]) + np.array([-1, 1]) * dx[-1] / 2)
        )
    dx = np.diff(x)
    return np.append(x[:-1] - dx / 2, x[-1] + np.array([-1, 1]) * dx[-1] / 2)


class Resample:
    r"""
    Resample regularly or irregularly sampled data onto a new grid, by
    integration.

    This is a generalization of :func:`ppxf.ppxf_util.log_rebin`.

    The abscissa (``x``) or the pixel borders (``xBorders``) should be given for
    irregularly sampled data.  For linearly or geometrically sampled data the
    abscissa can instead be generated from ``xRange``.  If ``x``, ``xBorders``
    and ``xRange`` are all None, the coordinates are assumed to be
    ``np.arange(y.shape[-1])``.

    The data are resampled by constructing the borders of the output grid from
    the ``new*`` keywords and integrating the input function between them.
    Output beyond the limits of the input is set to ``ext_value``.

    ``y`` may be 1-D or 2-D; the abscissa is always 1-D.  For 2-D input the
    resampling runs along the last axis.

    The function is assumed to be a step function (``step=True``).  If the
    output grid is much finer than the input, that assumption becomes visible;
    set ``step=False`` to assume linear interpolation between the input points
    instead.

    Parameters
    ----------
    y : :class:`numpy.ndarray`
        Data to resample, 1-D or 2-D.
    e : :class:`numpy.ndarray`, optional
        1-sigma errors on ``y``, with the same shape.
    mask : :class:`numpy.ndarray`, optional
        Boolean mask, True where a value should be ignored, with the same shape
        as ``y``.
    x : :class:`numpy.ndarray`, optional
        Abscissa coordinates of the input pixel centres.
    xRange : array-like, optional
        Two-element range of the input pixel centres.
    xBorders : :class:`numpy.ndarray`, optional
        Borders of the input pixels.
    inLog : bool, optional
        The input grid is geometrically sampled.
    newx : :class:`numpy.ndarray`, optional
        Coordinates of the output pixel centres.
    newRange : array-like, optional
        Two-element range of the output pixel centres.
    newBorders : :class:`numpy.ndarray`, optional
        Borders of the output pixels.
    newpix : int, optional
        Number of output pixels.
    newLog : bool, optional
        Sample the output grid geometrically.
    newdx : float, optional
        Output pixel width.
    base : float, optional
        Base of the logarithm used for geometric sampling.
    ext_value : float, optional
        Value assigned to output pixels beyond the input range.  If None, no
        such assignment is made.
    conserve : bool, optional
        Conserve the integral of the input, rather than its density.
    step : bool, optional
        Treat the input as a step function rather than linearly interpolating
        between the input points.
    covar : bool, optional
        Compute the covariance between output pixels induced by the resampling.
        Only available for step resampling.

    Attributes
    ----------
    x : :class:`numpy.ndarray`
        Coordinates of the input pixel centres.
    xborders : :class:`numpy.ndarray`
        Borders of the input pixels.
    outx : :class:`numpy.ndarray`
        Coordinates of the output pixel centres.
    outborders : :class:`numpy.ndarray`
        Borders of the output pixels.
    outy : :class:`numpy.ndarray`
        The resampled data.
    oute : :class:`numpy.ndarray`, None
        The resampled 1-sigma errors.
    outf : :class:`numpy.ndarray`
        The fraction of each output pixel covered by valid input data.
    covar : :class:`astropy.nddata.Covariance`, None
        The covariance between output pixels, if requested.

    Raises
    ------
    DC3Error
        Raised if the input grid or the output grid is under- or
        over-specified, if ``y`` is not a 1-D or 2-D array, if the shapes of
        the errors or mask disagree with ``y``, or if the covariance is
        requested without step resampling.
    """

    def __init__(
        self, y, e=None, mask=None, x=None, xRange=None, xBorders=None, inLog=False, newx=None,
        newRange=None, newBorders=None, newpix=None, newLog=True, newdx=None, base=10.0,
        ext_value=0.0, conserve=False, step=True, covar=False
    ):
        if np.sum([inp is not None for inp in [x, xRange, xBorders]]) != 1:
            raise DC3Error(
                'One and only one of the x, xRange, and xBorders arguments should be provided.'
            )
        if np.sum([inp is not None for inp in [newx, newRange, newBorders]]) != 1:
            raise DC3Error(
                'One and only one of the newx, newRange, and newBorders arguments should be '
                'provided.'
            )
        if not isinstance(y, np.ndarray):
            raise DC3Error('Input vector must be a numpy.ndarray.')
        if y.ndim > 2:
            raise DC3Error('Input must be a 1D or 2D array.')
        if covar and not step:
            raise DC3Error('Covariance is currently only calculated for step resampling.')

        # Set up the data, errors, and mask.  The mask is copied rather than
        # adopted, because the masks of y and e are merged into it below and
        # doing that in place would modify the caller's array.
        self.y = y.filled(0.0) if isinstance(y, np.ma.MaskedArray) else y.copy()
        self.twod = self.y.ndim == 2
        self.e = (
            None if e is None
            else e.filled(0.0) if isinstance(e, np.ma.MaskedArray) else e.copy()
        )
        self.m = np.zeros(self.y.shape, dtype=bool) if mask is None else np.array(mask, dtype=bool)

        if self.e is not None and self.e.shape != self.y.shape:
            raise DC3Error(
                f'Error array shape {self.e.shape} does not match the data shape {self.y.shape}.'
            )
        if self.m.shape != self.y.shape:
            raise DC3Error(
                f'Mask array shape {self.m.shape} does not match the data shape {self.y.shape}.'
            )

        # Merge in any masks carried by the input arrays themselves
        if isinstance(y, np.ma.MaskedArray):
            self.m |= y.mask
        if e is not None and isinstance(e, np.ma.MaskedArray):
            self.m |= e.mask

        # The input coordinates
        nx = self.y.shape[-1] if x is None and xBorders is None else None
        self.x, self.xborders = self._coordinate_grid(
            x=x, rng=xRange, nx=nx, borders=xBorders, log=inLog, base=base
        )

        # If conserving the integral, the input is integrated over the pixel
        # width, so convert it to a density
        if conserve:
            self.y /= (
                np.diff(self.xborders)[None, :] if self.twod else np.diff(self.xborders)
            )

        # The output coordinates
        nx = (
            self.x.size
            if newx is None and newBorders is None and newpix is None and newdx is None
            else newpix
        )
        self.outx, self.outborders = self._coordinate_grid(
            x=newx, rng=newRange, nx=nx, borders=newBorders, dx=newdx, log=newLog, base=base
        )

        if covar:
            self._resample_with_covariance()
        else:
            self.covar = None
            self.outy = self._resample_step(self.y) if step else self._resample_linear(self.y)
            # The mask and the errors are always resampled as a step function
            self.oute = None if self.e is None else self._resample_step(self.e, quad=True)
            self.outf = (
                self._resample_step(np.logical_not(self.m).astype(int))
                / np.diff(self.outborders)
            )

        # Convert back from a density unless the integral is being conserved
        if not conserve:
            width = np.diff(self.outborders)[None, :] if self.twod else np.diff(self.outborders)
            self.outy /= width
            if self.oute is not None:
                self.oute /= width
                if self.covar is not None:
                    self.covar = self.covar.apply_new_variance(np.square(self.oute.T))

        # Assign the extrapolated regions
        if ext_value is not None:
            indx = (
                (self.outborders[:-1] < self.xborders[0])
                | (self.outborders[1:] > self.xborders[-1])
            )
            if np.sum(indx) > 0:
                self.outy[..., indx] = ext_value
                self.outf[..., indx] = 0.
                if self.oute is not None:
                    self.oute[..., indx] = 0.

    def _resample_with_covariance(self):
        """
        Resample by explicit matrix multiplication, tracking the covariance.

        Resampling mixes neighbouring input pixels into each output pixel, which
        correlates the output even when the input is uncorrelated.  Building the
        operation as a matrix makes that correlation available; see
        :func:`_resample_step_matrix`.
        """
        A = self._resample_step_matrix()
        self.outy = np.dot(A, self.y.T).T
        self.outf = (
            np.dot(A, np.logical_not(self.m.T).astype(int)).T / np.diff(self.outborders)[..., :]
        )
        if self.e is None:
            self.covar = Covariance.from_matrix_multiplication(
                A, np.ones_like(self.x)
            ).apply_new_variance(np.ones_like(self.outx))
            self.oute = None
            return
        if self.twod:
            covar = np.empty(self.y.shape[0], dtype=object)
            for i in range(self.y.shape[0]):
                covar[i] = Covariance.from_matrix_multiplication(
                    A, np.square(self.e[i])
                ).to_dense()
            self.covar = Covariance(covar, assume_symmetric=True)
            self.oute = np.sqrt(self.covar.variance.T)
            return
        self.covar = Covariance.from_matrix_multiplication(A, np.square(self.e))
        self.oute = np.sqrt(self.covar.variance)

    @staticmethod
    def _coordinate_grid(x=None, rng=None, nx=None, dx=None, borders=None, log=False, base=10.0):
        """
        Construct the coordinate grid and its borders from what was provided.

        Parameters
        ----------
        x : :class:`numpy.ndarray`, optional
            Pixel centres.
        rng : array-like, optional
            Two-element range of the pixel centres.
        nx : int, optional
            Number of pixels.
        dx : float, optional
            Pixel width.
        borders : :class:`numpy.ndarray`, optional
            Pixel borders.
        log : bool, optional
            Sample geometrically.
        base : float, optional
            Base of the logarithm.

        Returns
        -------
        tuple
            The pixel centres and the pixel borders.

        Raises
        ------
        DC3Error
            Raised if both ``x`` and ``borders`` are given, or if there is too
            little information to construct the grid.
        """
        if x is not None and borders is not None:
            raise DC3Error(
                'Provide either x or borders, not both; this function does not check that the '
                'two are consistent with one another.'
            )
        if (x is not None or borders is not None) and rng is not None:
            warnings.warn('Provided both x or borders and the range.  Ignoring the range.')
        if x is None and borders is not None:
            return borders_to_centers(borders, log=log), borders
        if x is not None and borders is None:
            return x, centers_to_borders(x, log=log)

        if rng is None and nx is None:
            raise DC3Error('Insufficient input to construct the coordinate grid.')

        if rng is None:
            # A uniform pixel grid
            return np.arange(nx, dtype=float) + 0.5, np.arange(nx + 1, dtype=float)

        if dx is not None and nx is not None:
            warnings.warn(
                'Provided rng, dx, and nx, which over-specifies the grid; rng and nx take '
                'precedence.'
            )
        if nx is not None:
            borders = grid_borders(rng, nx, log=log, base=base)[0]
            return borders_to_centers(borders, log=log), borders

        nx, _rng = grid_npix(rng=rng, dx=dx, log=log, base=base)
        borders = grid_borders(_rng, nx, log=log, base=base)[0]
        return borders_to_centers(borders, log=log), borders

    def _resample_linear(self, v, quad=False):
        """
        Resample a vector, interpolating linearly between the input points.

        Parameters
        ----------
        v : :class:`numpy.ndarray`
            The vector to resample.
        quad : bool, optional
            Sum in quadrature, as required for errors.

        Returns
        -------
        :class:`numpy.ndarray`
            The resampled vector.
        """
        combinedX = np.append(self.outborders, self.x)
        srt = np.argsort(combinedX)
        combinedX = combinedX[srt]

        border = np.ones(combinedX.size, dtype=bool)
        border[self.outborders.size:] = False
        k = np.arange(combinedX.size)[border[srt]]

        if self.twod:
            interp = interpolate.interp1d(
                self.x, v, axis=-1, assume_sorted=True, fill_value='extrapolate'
            )
            combinedY = np.append(interp(self.outborders), v, axis=-1)[:, srt]
            integrand = (combinedY[:, 1:] + combinedY[:, :-1]) * np.diff(combinedX)[None, :] / 2.0
        else:
            interp = interpolate.interp1d(
                self.x, v, assume_sorted=True, fill_value='extrapolate'
            )
            combinedY = np.append(interp(self.outborders), v)[srt]
            integrand = (combinedY[1:] + combinedY[:-1]) * np.diff(combinedX) / 2.0

        if quad:
            integrand = np.square(integrand)

        out = (
            np.add.reduceat(integrand, k[:-1], axis=-1) if k[-1] == combinedX.size - 1
            else np.add.reduceat(integrand, k, axis=-1)[..., :-1]
        )
        return np.sqrt(out) if quad else out

    def _resample_step(self, v, quad=False):
        """
        Resample a vector, treating the input as a step function.

        Parameters
        ----------
        v : :class:`numpy.ndarray`
            The vector to resample.
        quad : bool, optional
            Sum in quadrature, as required for errors.

        Returns
        -------
        :class:`numpy.ndarray`
            The resampled vector.
        """
        # Convert to a step function: repeat each value twice, and each border
        # twice with the outermost two removed
        _v = np.repeat(v, 2, axis=1) if self.twod else np.repeat(v, 2)
        _x = np.repeat(self.xborders, 2)[1:-1]

        # Merge the input coordinates and the output borders
        indx = np.searchsorted(_x, self.outborders)
        combinedX = np.insert(_x, indx, self.outborders)

        v_indx = indx.copy()
        v_indx[indx >= _v.shape[-1]] = -1
        combinedY = (
            np.array([np.insert(__v, indx, __v[v_indx]) for __v in _v]) if self.twod
            else np.insert(_v, indx, _v[v_indx])
        )

        integrand = (
            combinedY[:, 1:] * np.diff(combinedX)[None, :] if self.twod
            else combinedY[1:] * np.diff(combinedX)
        )
        if quad:
            integrand = np.square(integrand)

        border = np.insert(
            np.zeros(_x.size, dtype=bool), indx, np.ones(self.outborders.size, dtype=bool)
        )
        k = np.arange(combinedX.size)[border]

        out = (
            np.add.reduceat(integrand, k[:-1], axis=-1) if k[-1] == combinedX.size - 1
            else np.add.reduceat(integrand, k, axis=-1)[..., :-1]
        )
        return np.sqrt(out) if quad else out

    def _resample_step_matrix(self):
        r"""
        Build the matrix :math:`\mathbf{A}` such that :math:`y = \mathbf{A} x`.

        Here :math:`x` is the input vector and :math:`y` the resampled one.
        Expressing the resampling as a matrix is what makes the induced
        covariance calculable.

        Returns
        -------
        :class:`numpy.ndarray`
            The resampling matrix, of shape ``(nout, nin)``.
        """
        ny = self.outx.size
        nx = self.x.size

        _p = np.repeat(np.arange(self.x.size), 2)
        _x = np.repeat(self.xborders, 2)[1:-1]

        indx = np.searchsorted(_x, self.outborders)
        combinedX = np.insert(_x, indx, self.outborders)

        p_indx = indx.copy()
        p_indx[indx >= _p.shape[-1]] = -1
        combinedP = np.insert(_p, indx, _p[p_indx])

        border = np.insert(
            np.zeros(_x.size, dtype=bool), indx, np.ones(self.outborders.size, dtype=bool)
        )
        nn = np.where(np.logical_not(border))[0][::2]
        k = np.zeros(len(combinedX), dtype=int)
        k[border] = np.arange(np.sum(border))
        k[nn - 1] = k[nn - 2]
        k[nn] = k[nn - 1]
        start, end = np.where(border)[0][[0, -1]]

        # The fraction of each input pixel falling into each output pixel
        fraction = np.diff(combinedX[start:end + 1])
        indx = fraction > 0
        A = np.zeros((ny, nx), dtype=float)
        A[k[start:end][indx], combinedP[start:end][indx]] = fraction[indx]
        return A
