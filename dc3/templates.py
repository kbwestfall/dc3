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
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from .core import resolution, sampling
# NOTE: dc3.par.parset only -- never dc3.par.dc3par, and never the dc3.par
# namespace.  The parameter sets below are nested into DC3Par by dc3.par.dc3par,
# so an import in that direction would close a cycle.  See dc3/par/__init__.py.
from .par.parset import ParSet, document_parameters
from .pkg.exceptions import DC3Error
from .spectra import Spectra, SpectrumMask


__all__ = [
    'PreparedTemplates',
    'TemplateLibrary',
    'TemplateLibraryPar',
    'TemplatePar',
    'prepare',
    'preparation_key',
]


class TemplateLibraryPar(ParSet):
    """
    Definition of a stellar template library as it is stored on disk.

    This says *what* the library is and how to read it.  It is deliberately
    separate from :class:`TemplatePar`, which says how the library is *prepared*
    for fitting: the two change independently.  A library is a fixed property of
    an installation, declared once and reused, whereas the preparation follows
    from the galaxy data being fit.

    Modelled on ``mangadap.proc.templatelibrary.TemplateLibraryDef``, which
    solves the same problem.

    .. note::

        The reader that consumes these parameters is not yet written; see the
        implementation record.  They are declared now because they fix the
        configuration surface that :func:`prepare` sits behind.
    """

    default_key = 'library'
    card_prefix = 'LIB'
    api_doc = ':class:`~dc3.templates.TemplateLibraryPar`'
    default_comment = 'Definition of the stellar template library.'

    key: Annotated[str | None, Field(
        default=None,
        description='Keyword identifying the library.  It names the library in the output and '
                    'is part of the key under which a prepared library is cached.'
    )]
    file_search: Annotated[str | None, Field(
        default=None,
        description='Search pattern, relative to the library root, matching the one-dimensional '
                    'FITS spectra that make up the library.'
    )]
    fwhm: Annotated[float | None, Field(
        default=None, gt=0.0,
        description='FWHM of the resolution element, in angstroms, taken as constant with '
                    'wavelength.  Superseded by resolution_ext where that is given.  This is '
                    'converted to an instrumental dispersion in km/s on ingest, since that is '
                    'the internal convention.'
    )]
    resolution_ext: Annotated[str | None, Field(
        default=None,
        description='Name of the extension holding the spectral resolution, R = lambda/dlambda, '
                    'as a function of wavelength.  Supersedes fwhm.  Converted to an '
                    'instrumental dispersion in km/s on ingest.'
    )]
    in_vacuum: Annotated[bool, Field(
        default=False,
        description='The library wavelengths are vacuum wavelengths.  If False they are air '
                    'wavelengths and are converted on ingest.'
    )]
    wave_limit: Annotated[list[float] | None, Field(
        default=None,
        description='Two-element lower and upper wavelength limit, in angstroms, outside which '
                    'the library spectra are not valid.  Omit it to use the full range of each '
                    'spectrum; an individual end cannot be left unbounded.'
    )]
    lower_flux_limit: Annotated[float | None, Field(
        default=None,
        description='Smallest valid flux.  Pixels below this are masked, which is how libraries '
                    'that pad their spectra with zeros are handled.'
    )]
    log10: Annotated[bool, Field(
        default=False,
        description='The library spectra are already sampled logarithmically in wavelength.  If '
                    'False they are resampled on ingest.'
    )]

    @model_validator(mode='after')
    def _check_wave_limit(self):
        """
        Check that the wavelength limit is a valid two-element range.

        Returns
        -------
        TemplateLibraryPar
            The validated parameter set.

        Raises
        ------
        ValueError
            Raised if the limit does not have two elements, or is not ordered.
        """
        if self.wave_limit is None:
            return self
        if len(self.wave_limit) != 2:
            raise ValueError(
                f'wave_limit must have exactly two elements; got {len(self.wave_limit)}.'
            )
        if self.wave_limit[0] >= self.wave_limit[1]:
            raise ValueError(
                f'wave_limit must be ordered; got {self.wave_limit}.'
            )
        return self


class TemplatePar(ParSet):
    """
    Parameters governing template preparation.

    Template preparation runs **once per execution**, never per spectrum, and is
    never part of the cost function.  It has two steps: resolution matching to a
    fiducial galaxy resolution, offset by a constant instrumental variance
    ``dvar_inst``; then resampling to the galaxy's sampling at an integer
    ``velscale_ratio``.

    These are the keyword arguments of :func:`prepare`, one for one, so the two
    are called as ``prepare(library, galaxy, **par.to_kwargs())`` and cannot
    drift apart without a test failing.  Their descriptions are what that
    function's docstring is generated from.  *What* library is prepared is
    declared separately, by :class:`TemplateLibraryPar`.
    """

    default_key = 'template'
    card_prefix = 'TPL'
    api_doc = ':class:`~dc3.templates.TemplatePar`'
    default_comment = 'Template preparation, run once per execution.'

    velscale_ratio: Annotated[int | Literal['auto'], Field(
        default=1,
        description='Integer number of prepared-template pixels per galaxy pixel.  Oversampling '
                    'the template keeps its line-spread function Nyquist-sampled on its own '
                    'grid, which relaxes one of the two bounds on the instrumental offset.  Use '
                    '"auto" for the smallest ratio at which the FWHM of the prepared '
                    'line-spread function spans at least two pixels at every wavelength.'
    )]
    epsilon_sigma: Annotated[float, Field(
        default=0.1, ge=0.1,
        description='Target for the *minimum* dispersion of the preparation kernel, in pixels.  '
                    'The floor of 0.1 is not arbitrary: ppxf_util.varsmooth silently clips its '
                    'kernel to 0.1 pixels, bounding the coordinate stretch of its algorithm, '
                    'which diverges as the kernel width goes to zero.  A smaller value here '
                    'would mean the code believed it applied a narrower kernel than it did, '
                    'making dvar_inst wrong by the difference and biasing the astrophysical '
                    'dispersion low.'
    )]
    sigma_floor: Annotated[float, Field(
        default=0.0, ge=0.0,
        description='Largest pedestal, in km/s, allowed to accommodate template regions of '
                    '*lower* resolution than the galaxy.  This sets the most negative dvar_inst, '
                    'and hence the floor it imposes on the measurable astrophysical dispersion.'
    )]
    mask_unmatched_idsp: Annotated[bool, Field(
        default=False,
        description='Mask template regions that cannot be brought to the target resolution.  '
                    'If False, such regions are retained and the resolution mismatch is '
                    'reported rather than hidden.'
    )]
    varsmooth_oversample: Annotated[int, Field(
        default=1, ge=1,
        description='Oversampling of the *internal* stretched grid used by the variable-sigma '
                    'convolution, which reduces its interpolation error.  This is a different '
                    'knob from velscale_ratio, which oversamples the *output* grid; the two '
                    'address different error terms and should not be conflated.'
    )]
    fiducial_method: Annotated[Literal['median', 'min', 'max'], Field(
        default='median',
        description='How the galaxy set is reduced to the single fiducial resolution the '
                    'templates are matched to.  Unless every galaxy spectrum has the same '
                    'resolution, the fiducial matches none of them exactly; "min" takes the '
                    'highest resolution present and so makes dvar_inst most negative, "max" '
                    'the lowest and so most positive.'
    )]

    @field_validator('velscale_ratio')
    @classmethod
    def _check_velscale_ratio(cls, value):
        """
        Check that an explicit ``velscale_ratio`` is at least one.

        A ``ge`` constraint cannot be attached to a field that also admits
        ``'auto'``, so the bound is checked here.

        Parameters
        ----------
        value : int, str
            The value to check.

        Returns
        -------
        int, str
            The value, unchanged.

        Raises
        ------
        ValueError
            Raised if an integer ratio is less than one.
        """
        if value != 'auto' and value < 1:
            raise ValueError(f'velscale_ratio must be at least 1, or "auto"; got {value}.')
        return value


class TemplateLibrary(Spectra):
    """
    A set of stellar templates sharing one logarithmic wavelength grid.

    A :class:`~dc3.spectra.Spectra` that additionally carries a name and
    enforces that the templates are noise-free.  Subclassing rather than
    wrapping keeps ``library.flux``, ``library.wave`` and the rest directly
    available, which is how the fitting machinery uses them.

    A library without an instrumental dispersion is accepted.  It cannot be
    resolution-matched, so :func:`prepare` uses it at its native resolution and
    warns that the reported dispersions are uncorrected.

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
        Raised if the templates carry errors.
    """

    def __init__(self, flux, log10lam0, dloglam, key='unnamed', **kwargs):
        super().__init__(flux, log10lam0, dloglam, **kwargs)
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

    def velocity_offset(self, galaxy):
        r"""
        Return the velocity offset between these templates and a galaxy grid.

        The prepared templates share the galaxy's pixel *sampling*, up to
        :attr:`velscale_ratio`, but not its starting wavelength; the offset
        this leaves is a constant velocity, handled analytically.  See
        :func:`~dc3.core.sampling.grid_velocity_offset`.

        Parameters
        ----------
        galaxy : :class:`~dc3.spectra.Spectra`
            The galaxy spectra to be fit.

        Returns
        -------
        float
            The offset in km/s, to be added to a velocity measured from a
            pixel lag.

        Raises
        ------
        DC3Error
            Raised if the galaxy's sampling is not exactly
            :attr:`velscale_ratio` times the templates', in which case no
            constant offset relates the two grids.
        """
        if not np.isclose(galaxy.dloglam, self.dloglam * self.velscale_ratio, rtol=1e-10):
            raise DC3Error(
                f'The galaxy pixel ({galaxy.dloglam:.6e} in log10 wavelength) is not '
                f'{self.velscale_ratio} times the template pixel ({self.dloglam:.6e}), so the two '
                'grids are not related by a constant velocity offset.  Prepare the templates '
                'against this galaxy.'
            )
        return sampling.grid_velocity_offset(
            self.log10lam0, galaxy.log10lam0, galaxy.dloglam, velscale_ratio=self.velscale_ratio
        )

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
    fiducial_idsp : :class:`numpy.ndarray`, None
        The fiducial galaxy resolution matched to, or None if no matching was
        performed.  None gives a key distinct from every matched preparation,
        so an unmatched product can never be served from the cache in place of
        a matched one, or vice versa.
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
    # Each branch is tagged, so an unmatched preparation cannot share a key with
    # a matched one.
    if fiducial_idsp is None:
        digest.update(b'unmatched')
    else:
        # The fiducial resolution is a vector, so it is hashed by content rather
        # than by identity; two runs with numerically identical vectors share a
        # key.
        digest.update(b'matched')
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


@document_parameters(TemplatePar)
def prepare(library, galaxy, velscale_ratio=1, epsilon_sigma=0.1, sigma_floor=0.0,
            mask_unmatched_idsp=False, varsmooth_oversample=1, fiducial_method='median'):
    r"""
    Run the two-step preparation pipeline.

    The keyword arguments are :class:`TemplatePar`, one for one
    and with the same defaults, so the intended call is

    .. code-block:: python

        prepared = templates.prepare(library, galaxy, **par.template.to_kwargs())

    Their descriptions below are generated from that parameter set rather than
    written here, so the text cannot drift from the one a user reads in the
    configuration file.  That the keywords and defaults themselves agree is
    checked by the test suite, since a disagreement is a coding error rather
    than something a user can cause.

    Parameters
    ----------
    library : TemplateLibrary
        The raw templates.
    galaxy : :class:`~dc3.spectra.Spectra`
        The galaxy spectra, which supply the fiducial resolution and the
        sampling to match.  They are read, never altered.
    {parameters}

    Returns
    -------
    PreparedTemplates
        The prepared templates and the offset left behind.

    Raises
    ------
    DC3Error
        Raised if the templates and the galaxy do not overlap in wavelength.

    Warns
    -----
    UserWarning
        Issued in any of these cases:

        - The templates or the galaxy carry no instrumental dispersion.  Step 1
          is then skipped: the templates are resampled at their native
          resolution, ``dvar_inst`` is zero, and the reported dispersions are
          uncorrected for any difference in resolution.
        - A supplied dispersion is smaller than pixel integration alone would
          produce, which most likely means the vector is wrong; see
          :func:`~dc3.core.resolution.check_pixelization`.
        - Part of the template range falls outside the galaxy's, where the
          fiducial resolution has to be held constant.
        - An explicit ``velscale_ratio`` leaves the prepared templates
          undersampled; see :func:`~dc3.core.resolution.minimum_velscale_ratio`.
        - ``velscale_ratio`` is ``'auto'`` but the templates carry no
          dispersion to choose it from, in which case a ratio of 1 is used.
    """
    if library.wave[-1] < galaxy.wave[0] or library.wave[0] > galaxy.wave[-1]:
        raise DC3Error(
            f'The templates ({library.wave[0]:.1f}-{library.wave[-1]:.1f} A) and the galaxy '
            f'({galaxy.wave[0]:.1f}-{galaxy.wave[-1]:.1f} A) do not overlap in wavelength.  '
            'Check that the galaxy has been de-redshifted and that the two use the same '
            'wavelength convention.'
        )

    # Soft diagnostics on the supplied dispersions, each against its own
    # sampling, since that is the pixel whose integration sets the bound.
    if library.idsp is not None:
        resolution.check_pixelization(library.idsp, library.velscale, label='templates')
    if galaxy.idsp is not None:
        resolution.check_pixelization(galaxy.idsp, galaxy.velscale, label='galaxy spectra')

    # --- Step 1: match the resolution on the templates' native grid ---------
    if library.idsp is None or galaxy.idsp is None:
        missing = [
            name for name, spec in [('templates', library), ('galaxy spectra', galaxy)]
            if spec.idsp is None
        ]
        warnings.warn(
            f'The {" and the ".join(missing)} carry no instrumental dispersion, so resolution '
            'matching is skipped: the templates are used at their native resolution and '
            'dvar_inst is set to zero.  The reported dispersions are therefore UNCORRECTED for '
            'any difference in resolution between the templates and the galaxy.'
        )
        # No fiducial was matched to, which the cache key must record.
        fiducial = None
        match = resolution.ResolutionMatch.identity(
            library.npix, library.velscale, epsilon_sigma=epsilon_sigma
        )
        matched_flux = library.flux
        # The templates keep whatever dispersion they carry, which may be none.
        matched_idsp = None if library.idsp is None else library.idsp[0]
    else:
        fiducial = galaxy.fiducial_resolution(method=fiducial_method)
        target, outside = _fiducial_on_template_grid(library.wave, galaxy.wave, fiducial)
        if np.any(outside):
            warnings.warn(
                f'{np.sum(outside)} of {outside.size} template pixels fall outside the galaxy '
                'wavelength range, where the fiducial resolution is held at its nearest '
                'measured value.  The resolution matching there rests on that assumption '
                'rather than on a measurement.'
            )
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

    # --- Choose the output sampling, and check it --------------------------
    # The line-spread function the output grid must carry is the one Step 1
    # produced, so this is decided only once Step 1 has run.
    if velscale_ratio == 'auto':
        if matched_idsp is None:
            warnings.warn(
                'velscale_ratio = "auto" needs the instrumental dispersion of the templates, '
                'which they do not carry.  Using a ratio of 1.'
            )
            _velscale_ratio = 1
        else:
            _velscale_ratio = resolution.minimum_velscale_ratio(matched_idsp, galaxy.velscale)
    else:
        _velscale_ratio = velscale_ratio
        if matched_idsp is not None:
            needed = resolution.minimum_velscale_ratio(matched_idsp, galaxy.velscale)
            if _velscale_ratio < needed:
                warnings.warn(
                    f'At velscale_ratio = {_velscale_ratio} the prepared templates are not '
                    'Nyquist-sampled: the FWHM of their line-spread function spans fewer than '
                    'two pixels somewhere in the range, so resampling degrades their '
                    f'effective resolution.  Use velscale_ratio = {needed} or more, or "auto".'
                )

    # --- Step 2: resample onto the galaxy's sampling -----------------------
    resampled = sampling.Resample(
        matched_flux, x=library.wave, newRange=[library.wave[0], library.wave[-1]],
        newdx=galaxy.dloglam / _velscale_ratio, newLog=True
    )
    out_flux = np.atleast_2d(resampled.outy)
    # Only the starting wavelength is taken from the resampled grid.  The pixel
    # size is required to be exactly the galaxy's divided by the ratio -- that
    # is what makes the offset between the two grids a constant velocity -- so
    # it is set directly rather than recovered with round-off from the output
    # wavelengths.
    log10lam0, _ = sampling.grid_from_wave(resampled.outx)
    dloglam = galaxy.dloglam / _velscale_ratio

    # The instrumental dispersion is a property of each wavelength, not an
    # integrated quantity, so it is interpolated rather than resampled.
    out_idsp = (
        None if matched_idsp is None
        else np.interp(resampled.outx, library.wave, matched_idsp)
    )

    mask = SpectrumMask(out_flux.shape)
    # outf is the fraction of each output pixel covered by valid input, so a
    # value of zero means the pixel drew on nothing.
    mask.turn_on('NODATA', select=np.broadcast_to(resampled.outf <= 0, out_flux.shape))
    if mask_unmatched_idsp and np.any(match.unmatched):
        unmatched = np.interp(resampled.outx, library.wave, match.unmatched.astype(float)) > 0.5
        mask.turn_on('UNMATCHED', select=np.broadcast_to(unmatched, out_flux.shape))

    return PreparedTemplates(
        out_flux, log10lam0, dloglam, match=match, velscale_ratio=_velscale_ratio,
        key=preparation_key(
            library.key, fiducial, galaxy.velscale, _velscale_ratio, epsilon_sigma,
            sigma_floor, varsmooth_oversample
        ),
        mask=mask, idsp=out_idsp,
    )
