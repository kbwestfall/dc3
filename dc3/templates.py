r"""
Stellar template libraries, and the pipeline that prepares them for fitting.

Preparation runs **once per execution**, never per spectrum, and is never part
of the cost function.  It has two steps:

===== =========================================================================
Step  Operation
===== =========================================================================
1     Resolution matching to a **fiducial** galaxy resolution, leaving a
      constant instrumental offset ``dvar_inst``
2     Resampling to the galaxy's sampling, at an integer ``velscale_ratio``
===== =========================================================================

Both steps are approximate, and the module says so rather than implying that
matching makes the velocity-dispersion corrections go away.

**Step 1 is approximate by construction, for two reasons.**  One matching
operation serves every template against a *fiducial* galaxy resolution (see
:func:`~dc3.spectra.Spectra.fiducial_resolution`), and unless every galaxy
spectrum has the same resolution -- they will not -- the fiducial matches no
individual spectrum exactly.  Less obviously, matching is also
**redshift-dependent**: in velocity units the instrumental resolution is
independent of redshift for each pixel, but the pixel *wavelength* is not, so
the fiducial is a fiducial resolution **and** a fiducial redshift.

This is a deliberate trade.  Moving preparation out of the fitting loop buys a
large speed-up and a ``dvar_inst`` that is known rather than re-derived, at the
cost of per-spectrum registration.  The residual is second order for a sample
spanning a narrow range of redshift and grows with the span.

**Step 2 changes the effective template resolution too.**  The array being
resampled already carries integration over its own native pixels, and rebinning
adds the resampling kernel on top; the excess is of order
:math:`\Delta^2_{\rm tpl}/12` and cannot be removed without deconvolution.  No
analytic term is added for it here, because under the pre-pixelized input
contract that would double-count.  It is measured instead, by pushing lines of
known width through this pipeline -- a separate deliverable.

.. note::

    This is a considerable simplification of
    ``mangadap.proc.templatelibrary.TemplateLibrary``, for reasons specific to
    ``dc3``.

    That class holds its state as an in-memory FITS HDU list, and is
    constructed against the datacube whose resolution and sampling the
    preparation targets.  Here a library *is* a :class:`~dc3.spectra.Spectra`,
    the pipeline is a function, and the galaxy spectra are passed to it, so
    neither the library nor its preparation is bound to a particular
    observation.

    It also threads a velocity offset through the preparation, using it to map
    the galaxy's resolution -- measured at *observed* wavelengths -- onto the
    rest wavelengths of the templates.  ``dc3`` shifts the galaxy to the
    approximate rest frame first, by whole pixels (:mod:`~dc3.core.deredshift`),
    so by the time preparation runs the two are already in the same frame and
    no offset needs carrying.

    Reading raw library files from disk is **not** implemented here.  The plan
    puts file I/O at the ``specutils`` boundary, which is not yet written, so a
    reader built now would be designed against an interface that does not exist.

.. include:: include/links.rst
"""

import hashlib
import warnings

import numpy as np

from .core import resolution, sampling
from .pkg.exceptions import DC3Error
from .spectra import Spectra, SpectrumMask


__all__ = ['PreparedTemplates', 'TemplateLibrary', 'prepare', 'preparation_key']


class TemplateLibrary(Spectra):
    """
    A set of stellar templates sharing one logarithmic wavelength grid.

    A :class:`~dc3.spectra.Spectra` that additionally carries a name and
    enforces the two things that make a set usable as templates.  Subclassing
    rather than wrapping keeps ``library.flux``, ``library.wave`` and the rest
    directly available, which is how the fitting machinery uses them.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        Template flux.
    log10lam0 : float
        As for :class:`~dc3.spectra.Spectra`.
    dloglam : float
        As for :class:`~dc3.spectra.Spectra`.
    key : str, optional
        A short name identifying the library.  It goes into the preparation
        cache key, so two libraries that differ must not share one.
    **kwargs
        Passed to :class:`~dc3.spectra.Spectra`.

    Raises
    ------
    DC3Error
        Raised if the templates carry no instrumental dispersion, or if they
        carry errors.
    """

    def __init__(self, flux, log10lam0, dloglam, key='unnamed', **kwargs):
        super().__init__(flux, log10lam0, dloglam, **kwargs)
        if self.idsp is None:
            raise DC3Error(
                'The templates carry no instrumental dispersion, so there is nothing to match '
                'their resolution from.  Supply idsp, converting from a resolving power with '
                'dc3.core.resolution.dispersion_from_resolving_power if necessary.'
            )
        if self.ivar is not None:
            raise DC3Error(
                'The templates carry errors.  Template spectra are treated as noise-free '
                'throughout dc3, as they are in ppxf: only the galaxy contributes to the '
                'covariance of the cross-correlation function.  Supplying errors here suggests '
                'the galaxy and template arguments may have been exchanged.'
            )
        self.key = str(key)

    @classmethod
    def from_spectra(cls, spectra, key='unnamed'):
        """
        Build a library from an existing spectrum set.

        Parameters
        ----------
        spectra : :class:`~dc3.spectra.Spectra`
            The templates.
        key : str, optional
            A short name identifying the library.

        Returns
        -------
        TemplateLibrary
            The library.
        """
        return cls(
            spectra.flux, spectra.log10lam0, spectra.dloglam, key=key,
            mask=SpectrumMask(spectra.mask.mask.copy()), idsp=spectra.idsp, cont=spectra.cont,
        )

    def _derived_kwargs(self):
        """Carry the library name through a copy or a selection."""
        return {'key': self.key}

    @property
    def ntpl(self):
        """The number of templates in the library."""
        return self.nspec

    def __repr__(self):
        """A short summary of the library."""
        return (
            f'<{type(self).__name__} {self.key!r}: {self.ntpl} templates x {self.npix} pixels, '
            f'{self.wave[0]:.1f}-{self.wave[-1]:.1f} A>'
        )


class PreparedTemplates(Spectra):
    r"""
    Templates prepared for fitting, and the offset preparation left behind.

    A :class:`~dc3.spectra.Spectra` on the output grid, carrying the provenance
    of its own preparation.  The offset is a property of the preparation rather
    than of any individual template, so it survives selecting a subset -- which
    is what the multi-template path does when it discards templates of zero
    weight.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The prepared flux.
    log10lam0 : float
        As for :class:`~dc3.spectra.Spectra`.
    dloglam : float
        As for :class:`~dc3.spectra.Spectra`.
    match : :class:`~dc3.core.resolution.ResolutionMatch`
        The matching computed on the templates' native grid.
    velscale_ratio : int, optional
        Prepared-template pixels per galaxy pixel.
    key : str, optional
        The cache key of this preparation; see :func:`preparation_key`.
    **kwargs
        Passed to :class:`~dc3.spectra.Spectra`.
    """

    def __init__(
        self, flux, log10lam0, dloglam, match=None, velscale_ratio=1, key='', **kwargs
    ):
        super().__init__(flux, log10lam0, dloglam, **kwargs)
        self.match = match
        self.velscale_ratio = int(velscale_ratio)
        self.key = str(key)

    def _derived_kwargs(self):
        """Carry the preparation's provenance through a copy or a selection."""
        return {'match': self.match, 'velscale_ratio': self.velscale_ratio, 'key': self.key}

    @property
    def dvar_inst(self):
        r"""The signed instrumental variance, in :math:`({\rm km/s})^2`."""
        return self.match.dvar_inst

    def astrophysical_variance(self, sigma_obs):
        r"""
        Convert a fitted dispersion to the astrophysical one.

        Delegates to the underlying
        :class:`~dc3.core.resolution.ResolutionMatch`; see there for why a
        signed variance is returned rather than a dispersion.

        Parameters
        ----------
        sigma_obs : float, :class:`numpy.ndarray`
            The fitted observed dispersion, in km/s.

        Returns
        -------
        float, :class:`numpy.ndarray`
            The astrophysical variance, in :math:`({\rm km/s})^2`.
        """
        return self.match.astrophysical_variance(sigma_obs)

    def __repr__(self):
        """A short summary of the prepared library."""
        return (
            f'<{type(self).__name__}: {self.nspec} templates x {self.npix} pixels, '
            f'{self.velscale:.2f} km/s/pix (ratio {self.velscale_ratio}), '
            f'dvar_inst={self.dvar_inst:+.3f} (km/s)^2>'
        )


def preparation_key(
    library_key, fiducial_idsp, velscale, velscale_ratio, epsilon_sigma, sigma_floor, oversample
):
    """
    Return a key identifying a prepared library.

    The key covers everything that changes the prepared product, so that a
    cached result can be reused exactly when it is still valid.  Because
    preparation no longer depends on any individual galaxy spectrum, the key is
    a property of the *run* rather than of a galaxy.

    Parameters
    ----------
    library_key : str
        Identifies the library.
    fiducial_idsp : :class:`numpy.ndarray`
        The fiducial galaxy resolution matched to.
    velscale : float
        The galaxy's velocity scale, in km/s per pixel.
    velscale_ratio : int
        Prepared-template pixels per galaxy pixel.
    epsilon_sigma : float
        The minimum preparation kernel, in pixels.
    sigma_floor : float
        The largest permitted pedestal, in km/s.
    oversample : int
        Oversampling of the convolution's internal grid.

    Returns
    -------
    str
        A hexadecimal digest.
    """
    digest = hashlib.sha256()
    digest.update(str(library_key).encode())
    # The fiducial resolution is a vector, so it is hashed by content rather
    # than by identity; two runs with numerically identical vectors share a key.
    digest.update(np.ascontiguousarray(fiducial_idsp, dtype=float).tobytes())
    for value in [velscale, velscale_ratio, epsilon_sigma, sigma_floor, oversample]:
        digest.update(repr(float(value)).encode())
    return digest.hexdigest()[:16]


def _fiducial_on_template_grid(template_wave, galaxy_wave, fiducial_idsp):
    """
    Interpolate the fiducial resolution onto the templates' wavelength grid.

    Templates routinely extend beyond the galaxy's wavelength range, where there
    is no measured galaxy resolution.  The fiducial is held at its nearest edge
    value there rather than extrapolated: a linear extrapolation of a resolution
    curve can go negative, and a constant is the weakest assumption available.

    Parameters
    ----------
    template_wave : :class:`numpy.ndarray`
        Template wavelengths.
    galaxy_wave : :class:`numpy.ndarray`
        Galaxy wavelengths.
    fiducial_idsp : :class:`numpy.ndarray`
        The fiducial resolution on the galaxy grid.

    Returns
    -------
    tuple
        The fiducial on the template grid, and a boolean array flagging the
        pixels that fall outside the galaxy's range.
    """
    outside = (template_wave < galaxy_wave[0]) | (template_wave > galaxy_wave[-1])
    # np.interp holds the end values outside the range, which is what is wanted
    return np.interp(template_wave, galaxy_wave, fiducial_idsp), outside


def prepare(library, galaxy, velscale_ratio=1, epsilon_sigma=0.1, sigma_floor=0.0,
            mask_unmatched_idsp=False, varsmooth_oversample=1, fiducial_method='median'):
    r"""
    Run the two-step preparation pipeline.

    The keyword arguments are :class:`~dc3.par.dc3par.TemplatePar`, one for one
    and with the same defaults, so the intended call is

    .. code-block:: python

        prepared = templates.prepare(library, galaxy, **par.template.to_kwargs())

    The agreement between the two is checked by the test suite rather than
    asserted here, since a disagreement is a coding error rather than something
    a user can cause.

    Parameters
    ----------
    library : TemplateLibrary
        The raw templates.
    galaxy : :class:`~dc3.spectra.Spectra`
        The galaxy spectra, which supply the fiducial resolution and the
        sampling to match.  They are read, never altered.
    velscale_ratio : int, optional
        Integer number of prepared-template pixels per galaxy pixel.
    epsilon_sigma : float, optional
        Target for the minimum dispersion of the preparation kernel, in pixels.
    sigma_floor : float, optional
        Largest pedestal, in km/s, allowed to accommodate template regions of
        lower resolution than the galaxy.
    mask_unmatched_idsp : bool, optional
        Mask template regions that cannot be brought to the target resolution.
    varsmooth_oversample : int, optional
        Oversampling of the internal stretched grid used by the variable-sigma
        convolution.
    fiducial_method : str, optional
        How the galaxy set is reduced to one resolution; see
        :func:`~dc3.spectra.Spectra.fiducial_resolution`.

    Returns
    -------
    PreparedTemplates
        The prepared templates and the offset left behind.

    Raises
    ------
    DC3Error
        Raised if the galaxy carries no instrumental dispersion, or if the
        templates and the galaxy do not overlap in wavelength.

    Warns
    -----
    UserWarning
        Issued if part of the template range falls outside the galaxy's, where
        the fiducial resolution has to be held constant.
    """
    fiducial = galaxy.fiducial_resolution(method=fiducial_method)

    if library.wave[-1] < galaxy.wave[0] or library.wave[0] > galaxy.wave[-1]:
        raise DC3Error(
            f'The templates ({library.wave[0]:.1f}-{library.wave[-1]:.1f} A) and the galaxy '
            f'({galaxy.wave[0]:.1f}-{galaxy.wave[-1]:.1f} A) do not overlap in wavelength.  '
            'Check that the galaxy has been de-redshifted and that the two use the same '
            'wavelength convention.'
        )

    target, outside = _fiducial_on_template_grid(library.wave, galaxy.wave, fiducial)
    if np.any(outside):
        warnings.warn(
            f'{np.sum(outside)} of {outside.size} template pixels fall outside the galaxy '
            'wavelength range, where the fiducial resolution is held at its nearest measured '
            'value.  The resolution matching there rests on that assumption rather than on a '
            'measurement.'
        )

    # --- Step 1: match the resolution on the templates' native grid ---------
    match = resolution.match_resolution(
        library.idsp[0], target, library.velscale,
        epsilon_sigma=epsilon_sigma, sigma_floor=sigma_floor
    )
    matched_flux = np.atleast_2d(resolution.apply_kernel(
        library.loglam, library.flux, match, oversample=varsmooth_oversample
    ))
    # By construction of the matching, the prepared resolution is the target
    # offset by a constant; see dc3.core.resolution.
    matched_idsp = np.sqrt(np.square(target) - match.dvar_inst)

    # --- Step 2: resample onto the galaxy's sampling -----------------------
    resampled = sampling.Resample(
        matched_flux, x=library.wave, newRange=[library.wave[0], library.wave[-1]],
        newdx=galaxy.dloglam / velscale_ratio, newLog=True
    )
    out_flux = np.atleast_2d(resampled.outy)
    log10lam0, dloglam = sampling.grid_from_wave(resampled.outx)

    # The instrumental dispersion is a property of each wavelength, not an
    # integrated quantity, so it is interpolated rather than resampled.
    out_idsp = np.interp(resampled.outx, library.wave, matched_idsp)

    mask = SpectrumMask(out_flux.shape)
    # outf is the fraction of each output pixel covered by valid input, so a
    # value of zero means the pixel drew on nothing.
    mask.turn_on('NODATA', select=np.broadcast_to(resampled.outf <= 0, out_flux.shape))
    if mask_unmatched_idsp and np.any(match.unmatched):
        unmatched = np.interp(resampled.outx, library.wave, match.unmatched.astype(float)) > 0.5
        mask.turn_on('UNMATCHED', select=np.broadcast_to(unmatched, out_flux.shape))

    return PreparedTemplates(
        out_flux, log10lam0, dloglam, match=match, velscale_ratio=velscale_ratio,
        key=preparation_key(
            library.key, fiducial, galaxy.velscale, velscale_ratio, epsilon_sigma,
            sigma_floor, varsmooth_oversample
        ),
        mask=mask, idsp=out_idsp,
    )
