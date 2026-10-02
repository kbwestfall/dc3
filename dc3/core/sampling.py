r"""
Spectral sampling: describing a wavelength grid, and detecting how one is
sampled.

The galaxy data and the prepared templates are always uniformly sampled in
:math:`\log_{10}\lambda`.  That is not a convenience: a Doppler shift is a pure
*translation* on such a grid, which is what lets the galaxy be de-redshifted by
an integer pixel shift (:mod:`~dc3.core.deredshift`) and the model be shifted by
a phase ramp in Fourier space, neither of which redistributes flux.  Template
libraries, however, arrive sampled however their authors chose, so a
:class:`SpectralGrid` can describe linear and irregular sampling as well, and
:func:`sampling_type` determines which a wavelength vector has.

Resampling onto a new grid is in :mod:`~dc3.core.resample`, and velocity and
redshift conversions in :mod:`~dc3.core.velocity`.

.. note::

    :func:`grid_velocity_offset` is adapted from
    ``mangadap.proc.ppxffit.PPXFFit.ppxf_tpl_obj_voff`` (BSD 3-Clause), recast
    in terms of the grid parameters rather than the wavelength vectors, and
    :func:`borders_to_centers` and :func:`centers_to_borders` are taken from
    ``mangadap/util/sampling.py`` (BSD 3-Clause); see ``licenses/README.rst``.

.. include:: ../include/links.rst
"""

import numpy as np

from ..pkg.exceptions import DC3Error
from .velocity import SPEED_OF_LIGHT


__all__ = [
    'SpectralGrid',
    'borders_to_centers',
    'centers_to_borders',
    'dloglam_from_velscale',
    'grid_velocity_offset',
    'log_wavelength_grid',
    'sampling_type',
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


def grid_velocity_offset(log10lam0_tpl, log10lam0_obj, dloglam_obj, velscale_ratio=1):
    r"""
    Return the velocity offset between two logarithmic grids of matched sampling.

    Templates and galaxy spectra need share only their pixel *sampling* -- the
    template pixel exactly ``1/velscale_ratio`` of the galaxy pixel -- not their
    starting wavelength.  Whatever the two starting wavelengths, a pixel lag
    between the spectra is then a fixed velocity plus this offset, which is
    known analytically and so needs no registration of one grid to the other.
    Nor would registering help: a Doppler shift almost never moves a given rest
    wavelength by a whole number of pixels, so the two spectra are offset by a
    fraction of a pixel regardless.

    For a line at rest wavelength :math:`\lambda_r` in the template, and
    Doppler shifted by :math:`V` in the galaxy, the pixel lag :math:`L` between
    the two satisfies

    .. math::

        V = c\,\Delta\ln\lambda\,L + V_{\rm off}, \qquad
        V_{\rm off} = c\,(\ln\lambda_{0,{\rm obj}} - \ln\lambda_{0,{\rm tpl}}),

    where :math:`\Delta\ln\lambda` is the galaxy pixel.  This is the
    ``log_velocity`` convention of :mod:`~dc3.core.velocity`, in which the two
    terms simply add.

    With ``velscale_ratio`` :math:`r > 1` the template is compared after being
    binned down by :math:`r`, so the template's reference point is the centre
    of its first *binned* pixel, :math:`(r-1)/2` template pixels past its first
    pixel, rather than its first pixel.

    Parameters
    ----------
    log10lam0_tpl : float
        :math:`\log_{10}` of the wavelength of the template's first pixel.
    log10lam0_obj : float
        :math:`\log_{10}` of the wavelength of the galaxy's first pixel.
    dloglam_obj : float
        The galaxy pixel size in :math:`\log_{10}\lambda`.  The template pixel
        is taken to be this divided by ``velscale_ratio``.
    velscale_ratio : int, optional
        Template pixels per galaxy pixel.

    Returns
    -------
    float
        :math:`V_{\rm off}` in km/s.  Positive when the galaxy grid starts
        redward of the template's.
    """
    reference = log10lam0_tpl + (velscale_ratio - 1) / 2 * dloglam_obj / velscale_ratio
    return SPEED_OF_LIGHT * np.log(10.0) * (log10lam0_obj - reference)


# ----------------------------------------------------------------------
# Describing and detecting the sampling of a wavelength vector
# ----------------------------------------------------------------------
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


def _validate_wave(wave):
    """
    Return a wavelength vector as a float array, checking it can be a grid.

    Parameters
    ----------
    wave : array-like
        Wavelengths of the pixel centres.

    Returns
    -------
    :class:`numpy.ndarray`
        The wavelengths as a 1-D float array.

    Raises
    ------
    DC3Error
        Raised if the vector has fewer than two elements, or is not positive
        and strictly ascending.
    """
    _wave = np.atleast_1d(np.asarray(wave, dtype=float))
    if _wave.ndim != 1:
        raise DC3Error('A wavelength vector must be one-dimensional.')
    if _wave.size < 2:
        raise DC3Error('A wavelength grid must have at least two pixels.')
    if np.any(_wave <= 0):
        raise DC3Error('Wavelengths must be positive.')
    if np.any(np.diff(_wave) <= 0):
        raise DC3Error('Wavelengths must be in strictly ascending order.')
    return _wave


def _uniform_fit(x):
    """
    Fit a uniform grid to a coordinate vector by least squares.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        The coordinate of each pixel centre: the wavelength for a linear grid,
        its logarithm for a logarithmic one.

    Returns
    -------
    tuple
        The fitted start and step, and the largest departure of ``x`` from the
        fitted grid in units of the step, i.e. in pixels.
    """
    i = np.arange(x.size, dtype=float)
    step, start = np.polyfit(i, x, 1)
    return float(start), float(step), float(np.amax(np.absolute(x - start - step * i)) / step)


def _departure_limit(wave, tol):
    """
    Return the largest departure from a regular grid, in pixels, to accept.

    The larger of ``tol`` and the float32 floor described in
    :func:`sampling_type`.

    Parameters
    ----------
    wave : :class:`numpy.ndarray`
        Wavelengths of the pixel centres, already validated.
    tol : float
        The requested tolerance, in pixels.

    Returns
    -------
    float
        The limit, in pixels.
    """
    return max(tol, np.finfo(np.float32).eps * np.amax(wave[1:] / np.diff(wave)))


def sampling_type(wave, tol=1e-3):
    r"""
    Determine whether a wavelength vector is sampled linearly, logarithmically,
    or neither.

    Each regular description is tested by fitting a uniform grid, in the
    wavelength or in its logarithm, and measuring how far any pixel centre lies
    from it, **in pixels**.  A description is accepted if that departure is
    within the tolerance.

    **The tolerance has a floor set by float32 rounding.**  Wavelength vectors
    are commonly stored in single precision, and a vector read that way and
    converted to double precision still carries the rounding, with no trace of
    it in the dtype.  Rounding to the nearest float32 displaces a wavelength by
    up to half the float32 machine epsilon, :math:`\epsilon_{32}\lambda/2`
    (:math:`\epsilon_{32} = 2^{-23}`), which is
    :math:`\epsilon_{32}\lambda/(2\Delta\lambda)` pixels: about
    :math:`2\times10^{-3}` pixels at :math:`\lambda/\Delta\lambda = 4\times10^4`,
    and :math:`3\times10^{-2}` at :math:`4\times10^5`.  The departure is
    therefore compared against the larger of ``tol`` and twice that,
    :math:`\epsilon_{32}\lambda/\Delta\lambda`, so a regularly gridded vector is
    never declared irregular merely because it was stored in single precision.
    The floor only loosens the test, and only by the uncertainty the stored
    values carry anyway.  It does not blur the distinctions that matter: a
    logarithmic grid tested as linear departs by tens of pixels over any
    realistic range, and a spliced grid drifts further with every pixel past
    the splice.

    A vector passing both tests is called logarithmic.  That happens only when
    the vector spans so small a fraction of its own wavelength that the two
    descriptions agree to within the tolerance everywhere, in which case the
    choice makes no difference at that tolerance, and logarithmic is the
    sampling the rest of ``dc3`` works in.

    Parameters
    ----------
    wave : array-like
        Wavelengths of the pixel centres, strictly ascending.
    tol : float, optional
        The largest acceptable departure from a regular grid, in pixels.  The
        comparison uses the larger of this and the float32 floor described
        above.

    Returns
    -------
    str
        ``'log'``, ``'linear'``, or ``'irregular'``.

    Raises
    ------
    DC3Error
        Raised if the vector cannot describe a grid; see :func:`_validate_wave`.
    """
    _wave = _validate_wave(wave)
    limit = _departure_limit(_wave, tol)
    if _uniform_fit(np.log10(_wave))[2] <= limit:
        return 'log'
    if _uniform_fit(_wave)[2] <= limit:
        return 'linear'
    return 'irregular'


class SpectralGrid:
    r"""
    The wavelength sampling shared by a set of spectra.

    Three kinds of sampling are supported:

    =============  =============================================================
    Kind           Description
    =============  =============================================================
    ``log``        Uniform in :math:`\log_{10}\lambda`: every pixel has the same
                   width in velocity.  The only sampling the galaxy data and the
                   prepared templates may have.
    ``linear``     Uniform in :math:`\lambda`, as most stellar libraries are
                   delivered.
    ``irregular``  Anything else, described pixel by pixel -- for example a
                   library spliced from sections sampled differently.
    =============  =============================================================

    Generally speaking, we recommend building a :class:`SpectralGrid` using
    either :meth:`from_log_spacing`, :meth:`from_linear_spacing`, or
    :meth:`from_vector`, instead of the direct constructor.

    Instances are immutable, so any number of spectrum sets can share one.
    The pixel centres and borders, and the quantities derived from them, are
    computed once, on construction, and returned as read-only arrays.

    **Pixel borders.**  A regular grid's borders follow exactly from its
    parameters, and in its own convention: a logarithmic grid's centres are
    the geometric centres of its pixels, a linear grid's the linear centres.

    An irregular grid is best given its borders explicitly.  If only the pixel
    centres are given, they are taken to be the **linear** centres of their
    pixels, and the borders are derived by :func:`centers_to_borders`: each
    pixel's lower border lies below its centre by half the distance to the
    following centre, and the last pixel is as wide as its neighbour.  Neither
    convention can be known to be right for an irregular grid, whose vector
    does not say how it was built; linear is adopted because it is the clearer
    assumption, and the two differ by only about :math:`1/(8\lambda/\Delta
    \lambda)` of a pixel.  The derived borders are exact wherever the pixel size
    is constant, and accurate to second order where it changes smoothly.
    Where it jumps -- at the splice between two sections of a spliced library
    -- the border just below the last pixel before the splice is misplaced, by
    a quarter of the change in pixel size.  Supply the borders to avoid it.

    Parameters
    ----------
    kind : str
        ``'log'``, ``'linear'`` or ``'irregular'``.
    npix : int
        The number of pixels.
    start, step : float, optional
        For a regular grid, the coordinate of the first pixel centre and the
        pixel size: in :math:`\log_{10}` of angstroms for ``log``, in
        angstroms for ``linear``.
    wave : :class:`numpy.ndarray`, optional
        For an irregular grid, the pixel centres, in angstroms.
    borders : :class:`numpy.ndarray`, optional
        For an irregular grid, the ``npix + 1`` pixel borders, in angstroms.

    Raises
    ------
    DC3Error
        Raised if the parameters do not describe a valid grid of the given
        kind.
    """

    def __init__(self, kind, npix, start=None, step=None, wave=None, borders=None):
        if kind not in ['log', 'linear', 'irregular']:
            raise DC3Error(f'Unknown grid kind {kind!r}; use "log", "linear" or "irregular".')
        self._kind = kind
        self._npix = int(npix)
        if self._npix < 2:
            raise DC3Error('A wavelength grid must have at least two pixels.')

        if kind == 'irregular':
            self._start = self._step = None
            self._wave, self._borders = self._irregular_arrays(wave, borders)
            self._loglam = np.log10(self._wave)
        else:
            if start is None or step is None:
                raise DC3Error(f'A {kind} grid needs both its start and its step.')
            if step <= 0:
                raise DC3Error(f'The pixel size must be positive; got {step}.')
            if kind == 'linear' and start - step / 2 <= 0:
                raise DC3Error('A linear grid must lie entirely at positive wavelengths.')
            self._start, self._step = float(start), float(step)
            # The grid's own coordinate, at the centres and at the borders
            centers = self._start + self._step * np.arange(self._npix, dtype=float)
            edges = self._start + self._step * (np.arange(self._npix + 1, dtype=float) - 0.5)
            if kind == 'log':
                self._wave, self._borders = np.power(10.0, centers), np.power(10.0, edges)
                self._loglam = centers
            else:
                self._wave, self._borders = centers, edges
                self._loglam = np.log10(centers)

        self._pixel_velocity = (
            np.full(self._npix, velscale(self._step)) if kind == 'log'
            else SPEED_OF_LIGHT * np.log(self._borders[1:] / self._borders[:-1])
        )
        # Every array is shared by whoever holds the grid, so none may be
        # modified in place.
        for array in [self._wave, self._borders, self._loglam, self._pixel_velocity]:
            array.flags.writeable = False

    def _irregular_arrays(self, wave, borders):
        """
        Validate, and if necessary complete, the arrays of an irregular grid.

        Parameters
        ----------
        wave : :class:`numpy.ndarray`
            The pixel centres.
        borders : :class:`numpy.ndarray`, None
            The pixel borders, or None to derive them from the centres.

        Returns
        -------
        tuple
            Copies of the centres and the borders, which the caller's arrays
            therefore do not share.

        Raises
        ------
        DC3Error
            Raised if the arrays are inconsistent with each other or with
            ``npix``.
        """
        if wave is None:
            raise DC3Error('An irregular grid needs its pixel centres.')
        # Copied, because the grid makes its arrays read-only, and
        # _validate_wave does not copy a float array it is given
        _wave = _validate_wave(wave).copy()
        if _wave.size != self._npix:
            raise DC3Error(f'Expected {self._npix} pixel centres; got {_wave.size}.')
        if borders is None:
            return _wave, centers_to_borders(_wave, log=False)
        _borders = _validate_wave(borders).copy()
        if _borders.size != self._npix + 1:
            raise DC3Error(f'Expected {self._npix + 1} pixel borders; got {_borders.size}.')
        if np.any(_wave <= _borders[:-1]) or np.any(_wave >= _borders[1:]):
            raise DC3Error('Every pixel centre must lie strictly within its borders.')
        return _wave, _borders

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_log_spacing(cls, log10lam0, dloglam, npix):
        r"""
        Return a grid uniform in :math:`\log_{10}\lambda` based on its defining
        parameters.

        Parameters
        ----------
        log10lam0 : float
            :math:`\log_{10}` of the first pixel centre, in angstroms.
        dloglam : float
            The pixel size in :math:`\log_{10}\lambda`.
        npix : int
            The number of pixels.

        Returns
        -------
        SpectralGrid
            The grid.
        """
        return cls('log', npix, start=log10lam0, step=dloglam)

    @classmethod
    def from_linear_spacing(cls, lam0, dlam, npix):
        """
        Return a grid uniform in wavelength based on its defining parameters.

        Parameters
        ----------
        lam0 : float
            The first pixel centre, in angstroms.
        dlam : float
            The pixel size, in angstroms.
        npix : int
            The number of pixels.

        Returns
        -------
        SpectralGrid
            The grid.
        """
        return cls('linear', npix, start=lam0, step=dlam)

    @classmethod
    def from_vector(cls, wave, borders=None, tol=1e-3):
        """
        Return the grid a wavelength vector describes, determining its kind.

        A vector found to be regular (see :func:`sampling_type`) is replaced by
        the uniform grid fit to it, which departs from the input by no more
        than the tolerance; that is the grid the data were sampled on, and it
        is free of any rounding in the stored values.  Otherwise the vector is
        adopted as an irregular grid, with the given borders, or with borders
        derived from the centres if none are given.

        Borders given for a vector that turns out to be regular must agree with
        the fitted grid's own borders to within the same tolerance.  They are
        not silently discarded: borders that disagree carry information the
        regular description would lose, so they are reported instead.

        Parameters
        ----------
        wave : array-like
            The pixel centres, in angstroms.
        borders : array-like, optional
            The pixel borders, in angstroms.  If the wavelengths are irregularly
            sampled, explicitly providing the coordinates of the pixels borders
            is strongly recommended; see the class documentation.  If
            irregularly gridded and the borders are not provided, they are
            inferred from the wavelength coordinates.
        tol : float, optional
            The tolerance, in pixels, used to determine the sampling type; see
            :func:`sampling_type`.

        Returns
        -------
        SpectralGrid
            The grid.

        Raises
        ------
        DC3Error
            Raised if borders are given for a regularly sampled vector and do
            not agree with the regular grid's borders.
        """
        _wave = _validate_wave(wave)
        kind = sampling_type(_wave, tol=tol)
        if kind == 'irregular':
            return cls('irregular', _wave.size, wave=_wave, borders=borders)
        start, step, _ = _uniform_fit(np.log10(_wave) if kind == 'log' else _wave)
        grid = cls(kind, _wave.size, start=start, step=step)
        if borders is not None:
            _borders = _validate_wave(borders)
            if _borders.size != grid.npix + 1:
                raise DC3Error(f'Expected {grid.npix + 1} pixel borders; got {_borders.size}.')
            # The departure in pixels, in the grid's own coordinate
            if kind == 'log':
                departure = np.absolute(np.log10(_borders) - np.log10(grid.borders)) / step
            else:
                departure = np.absolute(_borders - grid.borders) / step
            limit = _departure_limit(_wave, tol)
            if np.amax(departure) > limit:
                sense = 'logarithmically' if kind == 'log' else 'linearly'
                raise DC3Error(
                    f'The pixel centres are {sense} sampled, but the borders given depart from '
                    f'that grid by up to {np.amax(departure):.3g} pixels, more than the '
                    f'tolerance of {limit:.3g}.  The borders are inconsistent with the centres, '
                    'or describe pixels a regular grid cannot; construct the grid directly as '
                    'irregular to keep them.'
                )
        return grid

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def kind(self):
        """The kind of sampling: ``'log'``, ``'linear'`` or ``'irregular'``."""
        return self._kind

    @property
    def npix(self):
        """The number of pixels."""
        return self._npix

    @property
    def is_log(self):
        r"""Whether the grid is uniform in :math:`\log_{10}\lambda`."""
        return self._kind == 'log'

    @property
    def wave(self):
        """The pixel centres, in angstroms.  Read-only."""
        return self._wave

    @property
    def borders(self):
        """The ``npix + 1`` pixel borders, in angstroms.  Read-only."""
        return self._borders

    @property
    def loglam(self):
        r""":math:`\log_{10}` of the pixel centres.  Read-only."""
        return self._loglam

    @property
    def pixel_velocity(self):
        r"""
        The width of each pixel in velocity, in km/s.  Read-only.

        :math:`c \ln(\lambda_{i+1/2}/\lambda_{i-1/2})`, from the pixel
        borders.  Constant, and equal to :attr:`velscale`, for a logarithmic
        grid; for any other it varies from pixel to pixel.
        """
        return self._pixel_velocity

    @property
    def log10lam0(self):
        r""":math:`\log_{10}` of the first pixel centre.  Logarithmic grids only."""
        self._require_log('log10lam0')
        return self._start

    @property
    def dloglam(self):
        r"""The pixel size in :math:`\log_{10}\lambda`.  Logarithmic grids only."""
        self._require_log('dloglam')
        return self._step

    @property
    def velscale(self):
        """The velocity width of every pixel, in km/s.  Logarithmic grids only."""
        self._require_log('velscale')
        return velscale(self._step)

    def _require_log(self, name):
        """
        Refuse a quantity that exists only for a logarithmic grid.

        Parameters
        ----------
        name : str
            The quantity asked for.

        Raises
        ------
        DC3Error
            Raised if the grid is not logarithmic.
        """
        if self._kind != 'log':
            raise DC3Error(
                f'{name} is defined only for a logarithmically sampled grid; this grid is '
                f'{self._kind}.  Use pixel_velocity for the width of each pixel.'
            )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def breaks(self, tol=0.01):
        r"""
        Find the jumps in pixel size, where sections sampled differently meet.

        A spliced library is regular within each section and jumps in pixel
        size where two meet; downstream, such a jump breaks the assumption
        that the sampling varies smoothly (see
        :func:`~dc3.templates.prepare`).  A boundary between pixels :math:`i`
        and :math:`i+1` is flagged if the ratio of their widths departs from
        one by more than the tolerance,

        .. math::

            \left|\frac{\Delta_{i+1}}{\Delta_i} - 1\right| > {\rm tol},

        with the widths measured in wavelength from :attr:`borders`.  A
        logarithmic or linear grid has no breaks by definition.  An irregular
        grid that varies smoothly does not trip the test: even a linear grid
        described irregularly in log changes pixel size by only about
        :math:`10^{-4}` per pixel, where a splice changes it by tens of per
        cent.

        **The tolerance has a float32 floor.**  Wavelengths stored in single
        precision carry rounding of about :math:`\epsilon_{32}\lambda`, which
        puts noise of about :math:`\epsilon_{32}\lambda/\Delta\lambda` into the
        ratio of adjacent widths -- :math:`6\times10^{-4}` at
        :math:`\lambda/\Delta\lambda = 5000`, but :math:`10^{-2}` at
        :math:`10^5`.  The tolerance applied is therefore the larger of
        ``tol`` and four times that, the factor of four allowing that a ratio of
        two differences of rounded values carries about twice the error of
        one.  See :func:`sampling_type` for the same floor on a regular grid.

        **A splice may flag more than one boundary.**  If the borders were
        derived from the pixel centres (see the class documentation), the one
        just below the last pixel before a splice is misplaced, which changes
        the widths of the two pixels below the splice and spreads the jump over
        the two boundaries below it.  Every boundary that exceeds the
        tolerance is returned, so that nothing is lost; a run of adjacent
        breaks is one splice.  Supplying the borders avoids this.

        Parameters
        ----------
        tol : float, optional
            The largest fractional change in pixel size between neighbours that
            is treated as smooth.

        Returns
        -------
        :class:`numpy.ndarray`
            The index of the first pixel after each flagged boundary, in
            ascending order; empty if there are none.
        """
        if self._kind != 'irregular':
            return np.zeros(0, dtype=int)
        width = np.diff(self._borders)
        limit = max(tol, 4 * np.finfo(np.float32).eps * np.amax(self._wave / width))
        return np.where(np.absolute(width[1:] / width[:-1] - 1) > limit)[0] + 1

    def shifted(self, npix):
        r"""
        Return the grid relabelled by a whole number of pixels.

        Only a logarithmic grid can be shifted this way: a Doppler shift is a
        translation of :math:`\log\lambda`, so on such a grid it moves every
        pixel by the same amount.

        Parameters
        ----------
        npix : int
            The number of pixels to move the grid redward; negative moves it
            blueward.

        Returns
        -------
        SpectralGrid
            The shifted grid.
        """
        self._require_log('shifted')
        return type(self).from_log_spacing(
            self._start + int(npix) * self._step, self._step, self._npix
        )

    def __repr__(self):
        """A short summary of the grid."""
        w = self.wave
        if self._kind == 'log':
            detail = f'dloglam={self._step:.4e}, {self.velscale:.3f} km/s/pix'
        elif self._kind == 'linear':
            detail = f'dlam={self._step:.4g} A'
        else:
            detail = 'irregular'
        return f'<{type(self).__name__}: {self._npix} pixels, {w[0]:.2f}-{w[-1]:.2f} A, {detail}>'
