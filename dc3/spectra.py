r"""
The internal representation of a set of spectra.

This is the object the fit actually works on.  It is deliberately plain: plain
contiguous ``float64`` arrays on a shared logarithmic wavelength grid, with the
units fixed by convention and documented rather than carried.

Why not :class:`specutils.Spectrum`
-----------------------------------

``specutils`` is the right tool at the **I/O boundary** -- it handles the format
zoo, the WCS parsing, the unit attachment and the long tail of instrument
conventions -- and ``dc3`` uses it there.  But its objects carry
:class:`astropy.units.Quantity` arrays, lazy WCS and uncertainty objects whose
per-access overhead is irrelevant when scripting and fatal inside a cost
function evaluated thousands of times per spectrum.

So a spectrum is read with ``specutils`` and then **ingested here once**, after
which ``specutils`` is not touched again inside the fit.  Ingest converts; it
does not subclass.

Conventions, which are fixed and not carried
--------------------------------------------

=============  =====================================================================
Quantity       Convention
=============  =====================================================================
``wave``       Vacuum wavelength in Angstroms, pixel centres
``flux``       Arbitrary, but consistent across a set
``ivar``       Inverse variance of ``flux``, in the inverse square of its units
``idsp``       Instrumental dispersion :math:`\sigma` in km/s
=============  =====================================================================

.. important::

    ``idsp`` is an instrumental **dispersion** in km/s, not a resolving power.
    It is named ``idsp`` rather than ``sres`` deliberately: in ``mangadap``,
    ``sres`` always means :math:`R = \lambda/{\rm FWHM}_\lambda`, and reusing
    that name for a dispersion would invite exactly the confusion the naming is
    meant to avoid.

    **Converting a resolving power to a dispersion is part of ingestion.**  A
    user who has :math:`R` supplies it to the ingest function, which converts it
    with :func:`~dc3.core.resolution.dispersion_from_resolving_power`; nothing
    inside ``dc3`` ever sees :math:`R`.

.. warning::

    ``idsp`` is the **pre-pixelized** instrumental dispersion: the line-spread
    function *before* integration over the spectral channel.  This cannot be
    checked, so it is an input contract.  Supplying a post-pixelized dispersion
    biases every astrophysical dispersion the code reports.  For MaNGA-style
    inputs this means ``PREDISP``, not ``DISP``.

.. todo::

    **Revisit whether uncertainties should be held as inverse variance or as
    1-sigma errors.**  Inverse variance follows the MaNGA and PypeIt
    convention, and suits the places ``dc3`` actually uses it -- weighting is a
    multiplication rather than a division, and an unusable pixel is naturally
    zero rather than infinite.  But a user supplies and reads 1-sigma errors,
    so the choice pushes a conversion to every boundary, and the cost of
    converting in both directions may outweigh what the convention buys.  The
    decision is deferred; only this class and its ingest functions would change.

.. include:: include/links.rst
"""

import numpy as np

from .core import sampling
from .core.bitmask import BitMask, BitMaskArray
from .pkg.exceptions import DC3Error


__all__ = ['SpectrumBitMask', 'SpectrumMask', 'Spectra']


class SpectrumBitMask(BitMask):
    """
    Bits flagging individual spectral pixels.

    The order of these definitions is the datamodel; see
    :mod:`~dc3.core.bitmask`.
    """

    prefix = 'SPCBIT'
    bits = {
        'NODATA': 'Pixel has no data; it is outside the valid wavelength range',
        'INVALID': 'Flux is not finite',
        'NOIVAR': 'Pixel has no usable inverse variance',
        'USER': 'Pixel masked on input by the user',
        'REGION': 'Pixel falls in a masked spectral region',
        'NOIDSP': 'Pixel has no usable instrumental dispersion',
        'UNMATCHED': 'Template resolution could not be matched at this pixel',
    }


class SpectrumMask(BitMaskArray):
    """The per-pixel mask of a :class:`Spectra` set."""

    bitmask = SpectrumBitMask


class Spectra:
    r"""
    A set of spectra sharing one logarithmic wavelength grid.

    A single spectrum is the ``nspec == 1`` case; the arrays are always 2-D
    internally, so that nothing downstream needs to special-case it.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        Flux, of shape ``(nspec, npix)`` or ``(npix,)``.
    log10lam0 : float
        :math:`\log_{10}` of the first pixel's central wavelength.
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.
    ivar : :class:`numpy.ndarray`, optional
        Inverse variance of ``flux``, with the same shape.  **Only the galaxy
        carries errors**; templates are treated as noise-free throughout, which
        is a standing assumption of this method and of ``ppxf``.
    mask : :class:`numpy.ndarray`, :class:`SpectrumMask`, optional
        Per-pixel mask.  A boolean array is read as "True means masked" and
        recorded as the ``USER`` bit.  If None, a zeroed mask is created.
    idsp : :class:`numpy.ndarray`, optional
        Pre-pixelized instrumental dispersion in km/s, either per spectrum with
        the same shape as ``flux``, or one vector of length ``npix`` shared by
        all of them.
    cont : :class:`numpy.ndarray`, optional
        Continuum, with the same shape as ``flux``.

    Attributes
    ----------
    flux : :class:`numpy.ndarray`
        Flux, shape ``(nspec, npix)``.
    ivar : :class:`numpy.ndarray`, None
        Inverse variance, shape ``(nspec, npix)``.
    mask : :class:`SpectrumMask`
        Per-pixel mask, shape ``(nspec, npix)``.
    idsp : :class:`numpy.ndarray`, None
        Instrumental dispersion in km/s, shape ``(nspec, npix)``.
    cont : :class:`numpy.ndarray`, None
        Continuum, shape ``(nspec, npix)``.
    log10lam0 : float
        :math:`\log_{10}` of the first pixel's central wavelength.
    dloglam : float
        Pixel size in :math:`\log_{10}\lambda`.
    nspec : int
        Number of spectra.
    npix : int
        Number of pixels in each spectrum.
    """

    def __init__(self, flux, log10lam0, dloglam, ivar=None, mask=None, idsp=None, cont=None):
        self.flux = self._as_2d(flux, 'flux')
        self.nspec, self.npix = self.flux.shape
        if dloglam <= 0:
            raise DC3Error(f'The logarithmic pixel size must be positive; got {dloglam}.')
        self.log10lam0 = float(log10lam0)
        self.dloglam = float(dloglam)

        self.ivar = None if ivar is None else self._as_2d(ivar, 'ivar', match=True)
        self.cont = None if cont is None else self._as_2d(cont, 'cont', match=True)
        self.idsp = None if idsp is None else self._broadcast_idsp(idsp)
        self.mask = self._ingest_mask(mask)

        self._flag_invalid()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _as_2d(self, array, name, match=False):
        """
        Coerce an input array to a contiguous 2-D ``float64`` array.

        Parameters
        ----------
        array : array-like
            The array to coerce.
        name : str
            Name used in error messages.
        match : bool, optional
            Require the shape to match :attr:`flux`.

        Returns
        -------
        :class:`numpy.ndarray`
            The coerced array.

        Raises
        ------
        DC3Error
            Raised if the array has the wrong dimensionality or shape.
        """
        _array = np.ascontiguousarray(np.atleast_2d(np.asarray(array, dtype=float)))
        if _array.ndim != 2:
            raise DC3Error(f'{name} must be 1-D or 2-D; got {_array.ndim} dimensions.')
        if match and _array.shape != self.flux.shape:
            raise DC3Error(
                f'{name} has shape {_array.shape}, but flux has shape {self.flux.shape}.'
            )
        return _array

    def _broadcast_idsp(self, idsp):
        """
        Coerce the instrumental dispersion, allowing one vector for all spectra.

        Parameters
        ----------
        idsp : array-like
            Instrumental dispersion, of shape ``(npix,)`` or matching
            :attr:`flux`.

        Returns
        -------
        :class:`numpy.ndarray`
            The dispersion, shape ``(nspec, npix)``.

        Raises
        ------
        DC3Error
            Raised if the shape is neither of the two allowed.
        """
        _idsp = np.asarray(idsp, dtype=float)
        if _idsp.ndim == 1 and _idsp.size == self.npix:
            return np.ascontiguousarray(np.tile(_idsp, (self.nspec, 1)))
        _idsp = self._as_2d(_idsp, 'idsp')
        if _idsp.shape != self.flux.shape:
            raise DC3Error(
                f'idsp has shape {_idsp.shape}; expected {self.flux.shape} or ({self.npix},).'
            )
        return _idsp

    def _ingest_mask(self, mask):
        """
        Build the per-pixel mask from the input.

        Parameters
        ----------
        mask : None, :class:`SpectrumMask`, array-like
            The input mask.  A boolean array means "True is masked" and is
            recorded as the ``USER`` bit; an integer array is adopted as raw bit
            values.

        Returns
        -------
        :class:`SpectrumMask`
            The mask.

        Raises
        ------
        DC3Error
            Raised if the shape does not match :attr:`flux`.
        """
        if mask is None:
            return SpectrumMask(self.flux.shape)
        if isinstance(mask, SpectrumMask):
            if mask.shape != self.flux.shape:
                raise DC3Error(
                    f'mask has shape {mask.shape}, but flux has shape {self.flux.shape}.'
                )
            return mask

        _mask = np.atleast_2d(np.asarray(mask))
        if _mask.shape != self.flux.shape:
            raise DC3Error(f'mask has shape {_mask.shape}, but flux has shape {self.flux.shape}.')
        if _mask.dtype == bool:
            out = SpectrumMask(self.flux.shape)
            out.turn_on('USER', select=_mask)
            return out
        return SpectrumMask(np.ascontiguousarray(_mask.astype(SpectrumBitMask.minimum_dtype())))

    def _flag_invalid(self):
        """
        Flag pixels that cannot be used, whatever the input mask says.

        A non-finite flux, a non-positive or non-finite inverse variance, and a
        non-positive instrumental dispersion are all unusable regardless of what
        the caller flagged, so they are detected here rather than trusted to the
        input.
        """
        self.mask.turn_on('INVALID', select=np.logical_not(np.isfinite(self.flux)))
        if self.ivar is not None:
            bad = np.logical_or(np.logical_not(np.isfinite(self.ivar)), self.ivar <= 0)
            self.mask.turn_on('NOIVAR', select=bad)
        if self.idsp is not None:
            bad = np.logical_or(np.logical_not(np.isfinite(self.idsp)), self.idsp <= 0)
            self.mask.turn_on('NOIDSP', select=bad)

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_wave(cls, wave, flux, **kwargs):
        """
        Construct from an explicit wavelength vector.

        The vector is checked for logarithmic sampling and reduced to the two
        grid parameters; it is not stored.

        Parameters
        ----------
        wave : :class:`numpy.ndarray`
            Wavelengths of the pixel centres.
        flux : :class:`numpy.ndarray`
            Flux.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        Spectra
            The spectrum set.

        Raises
        ------
        DC3Error
            Raised if the wavelength vector is not logarithmically sampled, or
            its length does not match the flux.
        """
        log10lam0, dloglam = sampling.grid_from_wave(wave)
        _flux = np.atleast_2d(np.asarray(flux, dtype=float))
        if _flux.shape[-1] != np.asarray(wave).size:
            raise DC3Error(
                f'flux has {_flux.shape[-1]} pixels but wave has {np.asarray(wave).size}.'
            )
        return cls(flux, log10lam0, dloglam, **kwargs)

    # ------------------------------------------------------------------
    # The wavelength axis
    # ------------------------------------------------------------------
    @property
    def wave(self):
        """
        The wavelengths of the pixel centres.

        Computed from the grid parameters on each access rather than stored, so
        that the grid cannot drift out of step with the arrays.
        """
        return sampling.log_wavelength_grid(self.log10lam0, self.dloglam, self.npix)

    @property
    def loglam(self):
        """The base-10 logarithm of the pixel centres."""
        return self.log10lam0 + self.dloglam * np.arange(self.npix, dtype=float)

    @property
    def velscale(self):
        """The velocity scale of the grid, in km/s per pixel."""
        return sampling.velscale(self.dloglam)

    @property
    def shape(self):
        """The shape of the flux array."""
        return self.flux.shape

    def __len__(self):
        """The number of spectra in the set."""
        return self.nspec

    def __repr__(self):
        """A short summary of the set."""
        return (
            f'<{type(self).__name__}: {self.nspec} spectra x {self.npix} pixels, '
            f'{self.wave[0]:.1f}-{self.wave[-1]:.1f} A, {self.velscale:.2f} km/s/pix>'
        )

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    @property
    def gpm(self):
        """
        The good-pixel mask: True where a pixel is usable.

        This is the complement of "any bit set", so a pixel is good only if
        nothing at all is flagged against it.
        """
        return np.logical_not(self.mask.flagged())

    def nvalid(self):
        """
        Return the number of usable pixels in each spectrum.

        Returns
        -------
        :class:`numpy.ndarray`
            Counts, of length ``nspec``.
        """
        return np.sum(self.gpm, axis=1)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def mean(self):
        """
        Return the mean flux of each spectrum over its **unmasked** pixels.

        .. important::

            Averaging over unmasked pixels only is load-bearing, not a detail.
            The mean is subtracted from each spectrum before correlation, and
            including masked pixels in it would bias the zero point of the
            cross-correlation function by an amount that depends on how much was
            masked -- coupling the kinematics to the masking.

        Returns
        -------
        :class:`numpy.ndarray`
            Mean flux, of length ``nspec``.  A spectrum with no usable pixels
            gives zero.
        """
        gpm = self.gpm
        n = np.sum(gpm, axis=1)
        total = np.sum(np.where(gpm, self.flux, 0.0), axis=1)
        return np.divide(total, n, out=np.zeros(self.nspec, dtype=float), where=n > 0)

    def snr(self):
        """
        Return the median signal-to-noise ratio per pixel of each spectrum.

        Returns
        -------
        :class:`numpy.ndarray`
            Median signal-to-noise ratio, of length ``nspec``.  A spectrum with
            no usable pixels, or a set with no inverse variance, gives zero.

        Notes
        -----
        The median is used rather than the mean because a few high-weight pixels
        would otherwise dominate.
        """
        if self.ivar is None:
            return np.zeros(self.nspec, dtype=float)
        gpm = self.gpm
        snr = np.zeros(self.nspec, dtype=float)
        ratio = self.flux * np.sqrt(np.where(gpm, self.ivar, 0.0))
        for i in range(self.nspec):
            if np.any(gpm[i]):
                snr[i] = np.median(ratio[i][gpm[i]])
        return snr

    # ------------------------------------------------------------------
    # Manipulation
    # ------------------------------------------------------------------
    def copy(self):
        """
        Return an independent copy.

        Returns
        -------
        Spectra
            A copy sharing nothing with this set.
        """
        return type(self)(
            self.flux.copy(), self.log10lam0, self.dloglam,
            ivar=None if self.ivar is None else self.ivar.copy(),
            mask=SpectrumMask(self.mask.mask.copy()),
            idsp=None if self.idsp is None else self.idsp.copy(),
            cont=None if self.cont is None else self.cont.copy(),
        )

    def __getitem__(self, index):
        """
        Select a subset of the spectra, keeping the wavelength grid.

        Parameters
        ----------
        index : int, slice, :class:`numpy.ndarray`
            Index into the spectrum axis.

        Returns
        -------
        Spectra
            The selected spectra.
        """
        select = [index] if isinstance(index, (int, np.integer)) else index
        return type(self)(
            self.flux[select], self.log10lam0, self.dloglam,
            ivar=None if self.ivar is None else self.ivar[select],
            mask=SpectrumMask(np.atleast_2d(self.mask.mask[select]).copy()),
            idsp=None if self.idsp is None else self.idsp[select],
            cont=None if self.cont is None else self.cont[select],
        )
