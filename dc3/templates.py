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

.. include:: ../include/links.rst
"""

import hashlib
import warnings
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from . import log
from .core import resample, resolution, sampling
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
    sampling_tol: Annotated[float, Field(
        default=1e-3, gt=0.0,
        description='Tolerance, in pixels, used to determine how the library is sampled: the '
                    'largest departure of a pixel centre from a uniform grid, in wavelength or '
                    'in its logarithm, for the library still to be treated as that uniform grid.  '
                    'The comparison is against the larger of this and the rounding carried by '
                    'wavelengths stored in single precision; see '
                    ':func:`~dc3.core.sampling.sampling_type`.  The library is not resampled on '
                    'ingest, whatever its sampling: it is resampled once, onto the galaxy grid, '
                    'during preparation.'
    )]
    sampling_jump_tol: Annotated[float, Field(
        default=0.01, gt=0.0,
        description='Largest fractional change in pixel size between neighbouring pixels that '
                    'is treated as smooth.  A larger change marks a splice between sections of '
                    'an irregularly sampled library, around which the prepared templates are '
                    'masked.  The comparison is against the larger of this and the noise that '
                    'single-precision wavelengths put into the ratio of neighbouring pixel '
                    'sizes; see :meth:`~dc3.core.sampling.SpectralGrid.breaks`.  A regularly '
                    'sampled library has no splices.'
    )]
    idsp_jump_tol: Annotated[float, Field(
        default=0.01, gt=0.0,
        description='Largest fractional change in instrumental dispersion between neighbouring '
                    'pixels that is treated as smooth.  A larger change marks a jump in the '
                    'library resolution, around which the prepared templates are masked.  A '
                    'change spread over several pixels, each below this, is treated as smooth '
                    'however large its total; see :func:`~dc3.core.resolution.idsp_breaks`.'
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
        default=2, ge=2,
        description='Oversampling of the *internal* stretched grid used by the variable-sigma '
                    'convolution, which reduces its interpolation error.  At least 2: at 1, a '
                    'uniform kernel is applied exactly but one that varies even slightly '
                    'broadens the templates by up to a third of a pixel squared in variance, so '
                    'the result would depend on whether the kernel happened to be uniform; see '
                    ':func:`~dc3.core.resolution.apply_kernel`.  This is a different knob from '
                    'velscale_ratio, which oversamples the *output* grid; the two address '
                    'different error terms and should not be conflated.'
    )]
    correct_lsf_excess: Annotated[bool, Field(
        default=True,
        description='Account for the variance that the two steps of preparation add to the '
                    'line-spread function beyond what they are asked to: the variable-sigma '
                    'convolution\'s own interpolation, and the resampling\'s treatment of each '
                    'pixel as a step.  The matching kernel is then chosen so that the prepared '
                    'templates reach the target resolution, in the second moment, and the '
                    'dispersion reported where they cannot is the one they actually carry.  If '
                    'False, the kernel alone is assumed to change the resolution, as in '
                    'mangadap, and the prepared templates are broader than reported by about a '
                    'third of a native template pixel squared in variance.'
    )]
    resample_excess_method: Annotated[Literal['expectation', 'local'], Field(
        default='local',
        description='How the variance the resampling adds is predicted, when correct_lsf_excess '
                    'is True.  "expectation" takes its average over all offsets between the two '
                    'grids, a sixth of the native template pixel squared at every wavelength.  '
                    '"local" computes it from the offsets the two grids actually have, averaged '
                    'over the output pixels a line spans; it is exact where the grids align, '
                    'where the expectation is not.'
    )]
    convolution_mask_growth: Annotated[float, Field(
        default=3.0, gt=0.0,
        description='How far the prepared templates are masked beyond each region of the '
                    'template library that the resolution-matching convolution cannot treat '
                    'correctly, in units of the dispersion of the matching kernel there (the '
                    'largest across the region and its two neighbouring pixels), and never less '
                    'than one native template pixel.  The convolution spreads each pixel over '
                    'the kernel footprint, so the region is grown by it.  It applies to three '
                    'kinds of region, each flagged with its own bit: (1) a jump in the library '
                    'sampling, found with the library\'s sampling_jump_tol, flagged SAMP_JUMP; '
                    '(2) a jump in the library resolution, found with its idsp_jump_tol, flagged '
                    'RES_JUMP; and (3) a run of pixels masked in the library itself, for '
                    'example a gap between spliced sections, flagged TPL_MASKED.'
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
    A set of stellar templates sharing one wavelength grid.

    A :class:`~dc3.spectra.Spectra` that additionally carries a name and
    enforces that the templates are noise-free.  Subclassing rather than
    wrapping keeps ``library.flux``, ``library.wave`` and the rest directly
    available, which is how the fitting machinery uses them.

    The grid may be of any kind -- logarithmic, linear or irregular -- and is
    kept as delivered.  The library is resampled once, onto the galaxy's
    sampling, by :func:`prepare`; resampling it on ingest as well would add a
    second resampling, and each one broadens the effective resolution.

    A library without an instrumental dispersion is accepted.  It cannot be
    resolution-matched, so :func:`prepare` uses it at its native resolution and
    warns that the reported dispersions are uncorrected.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        Template flux.
    grid : :class:`~dc3.core.sampling.SpectralGrid`
        The library's wavelength grid, of any kind.
    key : str, optional
        A short name identifying the library.  It goes into the preparation
        cache key, so two libraries that differ must not share one.
    sampling_jump_tol : float, optional
        The tolerance used to find splices in the sampling; see
        :attr:`sampling_breaks` and :class:`TemplateLibraryPar`.
    idsp_jump_tol : float, optional
        The tolerance used to find jumps in the resolution; see
        :attr:`idsp_breaks` and :class:`TemplateLibraryPar`.
    **kwargs
        Passed to :class:`~dc3.spectra.Spectra`.

    Raises
    ------
    DC3Error
        Raised if the templates carry errors, or a tolerance is not positive.
    """

    def __init__(
        self, flux, grid, key='unnamed', sampling_jump_tol=0.01, idsp_jump_tol=0.01, **kwargs
    ):
        super().__init__(flux, grid, **kwargs)
        if self.ivar is not None:
            raise DC3Error(
                'The templates carry errors.  Template spectra are treated as noise-free '
                'throughout dc3, as they are in ppxf: only the galaxy contributes to the '
                'covariance of the cross-correlation function.  Supplying errors here suggests '
                'the galaxy and template arguments may have been exchanged.'
            )
        self.key = str(key)
        if sampling_jump_tol <= 0 or idsp_jump_tol <= 0:
            raise DC3Error(
                'The tolerances on jumps in sampling and resolution must be positive; got '
                f'{sampling_jump_tol} and {idsp_jump_tol}.'
            )
        self.sampling_jump_tol = float(sampling_jump_tol)
        self.idsp_jump_tol = float(idsp_jump_tol)

    @classmethod
    def from_spectra(cls, spectra, key='unnamed', **kwargs):
        """
        Build a library from an existing spectrum set.

        Parameters
        ----------
        spectra : :class:`~dc3.spectra.Spectra`
            The templates.
        key : str, optional
            A short name identifying the library.
        **kwargs
            The jump tolerances, passed to the constructor.

        Returns
        -------
        TemplateLibrary
            The library.
        """
        return cls(
            spectra.flux, spectra.grid, key=key,
            mask=SpectrumMask(spectra.mask.mask.copy()), idsp=spectra.idsp, cont=spectra.cont,
            **kwargs
        )

    def _derived_kwargs(self):
        """Carry the library name and tolerances through a copy or a selection."""
        return {
            'key': self.key, 'sampling_jump_tol': self.sampling_jump_tol,
            'idsp_jump_tol': self.idsp_jump_tol
        }

    @property
    def sampling_breaks(self):
        """
        The splices in the library's sampling.

        The index of the first pixel after each boundary across which the
        pixel size jumps by more than :attr:`sampling_jump_tol`; see
        :meth:`~dc3.core.sampling.SpectralGrid.breaks`.  Empty for a regularly
        sampled library.
        """
        return self.grid.breaks(tol=self.sampling_jump_tol)

    @property
    def idsp_breaks(self):
        """
        The jumps in the library's instrumental dispersion.

        The index of the first pixel after each boundary across which the
        dispersion jumps by more than :attr:`idsp_jump_tol`; see
        :func:`~dc3.core.resolution.idsp_breaks`.  The dispersion of the first
        template is used, as it is for the resolution matching, which assumes
        every template shares it.  Empty if the library carries no dispersion.
        """
        if self.idsp is None:
            return np.zeros(0, dtype=int)
        return resolution.idsp_breaks(self.idsp[0], tol=self.idsp_jump_tol)

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
    grid : :class:`~dc3.core.sampling.SpectralGrid`
        The output grid: logarithmic, with a pixel size exactly the galaxy's
        divided by ``velscale_ratio``.
    match : :class:`~dc3.core.resolution.ResolutionMatch`
        The matching computed on the templates' native grid.
    velscale_ratio : int, optional
        Prepared-template pixels per galaxy pixel.
    key : str, optional
        The cache key of this preparation; see :func:`preparation_key`.
    **kwargs
        Passed to :class:`~dc3.spectra.Spectra`.
    """

    def __init__(self, flux, grid, match=None, velscale_ratio=1, key='', **kwargs):
        super().__init__(flux, grid, **kwargs)
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
        galaxy : :class:`~dc3.spectra.GalaxySpectra`
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
    library_key, fiducial_idsp, velscale, velscale_ratio, epsilon_sigma, sigma_floor, oversample,
    convolution_mask_growth=3.0, sampling_jump_tol=0.01, idsp_jump_tol=0.01,
    correct_lsf_excess=True, resample_excess_method='local'
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
    convolution_mask_growth : float, optional
        How far the masks around jumps and masked pixels in the library are
        grown, in kernel dispersions.
    sampling_jump_tol, idsp_jump_tol : float, optional
        The library's tolerances on jumps in sampling and resolution, which set
        where those bands fall.
    correct_lsf_excess : bool, optional
        Whether the excess variance of the two steps is corrected for.
    resample_excess_method : str, optional
        How the resampling's excess is predicted.  It changes the product only
        when ``correct_lsf_excess`` is True, so it enters the key only then.

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
    for value in [
        velscale, velscale_ratio, epsilon_sigma, sigma_floor, oversample,
        convolution_mask_growth, sampling_jump_tol, idsp_jump_tol
    ]:
        digest.update(repr(float(value)).encode())
    digest.update(
        f'corrected-{resample_excess_method}'.encode() if correct_lsf_excess else b'uncorrected'
    )
    return digest.hexdigest()[:16]


def _count_jumps(breaks):
    """
    Count the jumps a set of breaks describes.

    One jump may flag a run of adjacent boundaries -- see
    :meth:`~dc3.core.sampling.SpectralGrid.breaks` -- so a run counts once.

    Parameters
    ----------
    breaks : :class:`numpy.ndarray`
        Ascending indices of flagged boundaries.

    Returns
    -------
    int
        The number of runs of adjacent indices.
    """
    return 0 if len(breaks) == 0 else 1 + int(np.sum(np.diff(breaks) > 1))


def _check_splices(sampling_breaks, sampling_jump_tol):
    """
    Warn if a library appears to be spliced from implausibly many sections.

    Spliced libraries join a handful of sections, so a library that the
    sampling breaks divide into **more than 5 segments** is more likely to have
    noisy wavelengths than to be spliced that many times, and every apparent
    splice masks a guard band of the prepared templates.  The limit of 5 is a
    judgement rather than a measurement, and is fixed here rather than
    configurable until a library shows it needs to be otherwise.  Only jumps in
    the sampling are counted, since the remedy is ``sampling_jump_tol``.

    Parameters
    ----------
    sampling_breaks : :class:`numpy.ndarray`
        The library's sampling breaks; see
        :attr:`TemplateLibrary.sampling_breaks`.
    sampling_jump_tol : float
        The tolerance they were found with, for the message.

    Returns
    -------
    int
        The number of segments, one more than the number of splices.
    """
    nsegments = _count_jumps(sampling_breaks) + 1
    if nsegments > 5:
        warnings.warn(
            f'The template sampling breaks into {nsegments} segments at sampling_jump_tol = '
            f'{sampling_jump_tol}.  Spliced libraries join only a handful of sections, so this '
            'more likely means the pixel sizes are noisy than that the library is spliced that '
            'many times; each apparent splice masks part of the prepared templates.  Increase '
            'sampling_jump_tol.'
        )
    return nsegments


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


def _segment_bounds(breaks, npix):
    """
    Return the first and last-plus-one native pixel of each segment.

    Parameters
    ----------
    breaks : :class:`numpy.ndarray`
        Ascending indices of the first pixel after each break.
    npix : int
        The number of native pixels.

    Returns
    -------
    tuple
        Two integer arrays, the start and the end of each segment.
    """
    return np.concatenate([[0], breaks]), np.concatenate([breaks, [npix]])


def _segment_overlap(out_borders, borders, breaks):
    """
    Determine which native segments each output pixel draws on.

    Parameters
    ----------
    out_borders : :class:`numpy.ndarray`
        The output pixel borders, in angstroms.
    borders : :class:`numpy.ndarray`
        The native pixel borders, in angstroms.
    breaks : :class:`numpy.ndarray`
        Ascending indices of the first native pixel after each break.

    Returns
    -------
    tuple
        A boolean array of shape ``(nsegment, nout)``, True where an output
        pixel overlaps a segment; and the segment covering the greater part of
        each output pixel.  An output pixel that overlaps no segment, beyond
        the native range, is assigned the nearest.
    """
    edges = borders[np.concatenate([[0], breaks, [borders.size - 1]])]
    lo, hi = out_borders[:-1], out_borders[1:]
    overlap = np.clip(
        np.minimum(hi[None, :], edges[1:, None]) - np.maximum(lo[None, :], edges[:-1, None]),
        0.0, None
    )
    centres = (lo + hi) / 2
    nearest = np.searchsorted(edges[1:-1], centres, side='right')
    majority = np.where(np.any(overlap > 0, axis=0), np.argmax(overlap, axis=0), nearest)
    return overlap > 0, majority


def _interp_within_segments(out_wave, wave, values, breaks, majority):
    """
    Interpolate a per-pixel quantity onto the output grid without crossing a break.

    Each output pixel takes the value interpolated within the segment
    covering the greater part of it, held at that segment's end value beyond
    its last pixel centre.  A value is never interpolated across a jump, which
    would give a pixel straddling it a value neither side has.

    Parameters
    ----------
    out_wave : :class:`numpy.ndarray`
        The output pixel centres.
    wave : :class:`numpy.ndarray`
        The native pixel centres.
    values : :class:`numpy.ndarray`
        The quantity at each native pixel.
    breaks : :class:`numpy.ndarray`
        Ascending indices of the first native pixel after each break.
    majority : :class:`numpy.ndarray`
        The segment assigned to each output pixel; see
        :func:`_segment_overlap`.

    Returns
    -------
    :class:`numpy.ndarray`
        The quantity at each output pixel.
    """
    out = np.empty(out_wave.size, dtype=float)
    for s, (start, end) in enumerate(zip(*_segment_bounds(breaks, wave.size))):
        select = majority == s
        out[select] = np.interp(out_wave[select], wave[start:end], values[start:end])
    return out


def _flag_within_segments(out_wave, wave, flag, breaks, overlap):
    """
    Carry a per-pixel flag onto the output grid without crossing a break.

    Within each segment the flag is interpolated and thresholded at one half.
    An output pixel straddling a break takes the flag from **every** segment
    it overlaps, so it is set if it is set on either side.

    Parameters
    ----------
    out_wave : :class:`numpy.ndarray`
        The output pixel centres.
    wave : :class:`numpy.ndarray`
        The native pixel centres.
    flag : :class:`numpy.ndarray`
        The boolean flag at each native pixel.
    breaks : :class:`numpy.ndarray`
        Ascending indices of the first native pixel after each break.
    overlap : :class:`numpy.ndarray`
        Boolean, ``(nsegment, nout)``; see :func:`_segment_overlap`.

    Returns
    -------
    :class:`numpy.ndarray`
        The flag at each output pixel.
    """
    out = np.zeros(out_wave.size, dtype=bool)
    _flag = np.asarray(flag, dtype=float)
    for s, (start, end) in enumerate(zip(*_segment_bounds(breaks, wave.size))):
        select = overlap[s]
        out[select] |= np.interp(out_wave[select], wave[start:end], _flag[start:end]) > 0.5
    return out


def _grow_regions(out_borders, borders, starts, ends, kernel_pixels, growth):
    """
    Flag the output pixels within the grown footprint of any native region.

    Each region runs from native border ``starts[i]`` to native border
    ``ends[i]``, and is grown by ``growth`` times the local kernel dispersion
    either side, in native pixels, and never by less than one native pixel.
    The local kernel is the largest across the region's pixels and the pixel
    either side of it.  Where no resolution matching was performed there is no
    kernel and no convolution, so the growth is the one-pixel minimum, which
    still covers an output pixel that straddles the region's edge.

    Parameters
    ----------
    out_borders : :class:`numpy.ndarray`
        The output pixel borders, in angstroms.
    borders : :class:`numpy.ndarray`
        The native pixel borders, in angstroms.
    starts, ends : :class:`numpy.ndarray`
        The first and last native border of each region.  A break at border
        ``b``, between native pixels ``b - 1`` and ``b``, is the region from
        ``b`` to ``b``; a run of native pixels ``i`` to ``j - 1`` is the region
        from ``i`` to ``j``.
    kernel_pixels : :class:`numpy.ndarray`, None
        The matching kernel's dispersion at each native pixel, in native
        pixels, or None if no matching was performed.
    growth : float
        How far each region is grown, in kernel dispersions.

    Returns
    -------
    :class:`numpy.ndarray`
        Boolean, True for each output pixel that overlaps a grown region.
    """
    flagged = np.zeros(out_borders.size - 1, dtype=bool)
    if len(starts) == 0:
        return flagged
    npix = borders.size - 1
    index = np.arange(borders.size, dtype=float)
    for start, end in zip(starts, ends):
        if kernel_pixels is None:
            sigma = 0.0
        else:
            sigma = np.amax(kernel_pixels[max(start - 1, 0):min(end + 1, npix)])
        halfwidth = max(growth * sigma, 1.0)
        # Fractional border index to wavelength; the region is clipped to the grid
        lo = np.interp(start - halfwidth, index, borders)
        hi = np.interp(end + halfwidth, index, borders)
        flagged |= (out_borders[:-1] < hi) & (out_borders[1:] > lo)
    return flagged


def _resample(flux, library, galaxy, velscale_ratio):
    """
    Resample library spectra onto the galaxy's sampling, at a velocity-scale ratio.

    The input pixels are given by their borders, which the library's grid holds
    exactly in its own convention whatever its kind; ``inLog`` only sets how
    :class:`~dc3.core.resample.Resample` recovers the centres from them, which
    the resampling itself does not use.  The output grid depends on the grids
    alone, never on the flux.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The flux on the library's native grid.
    library : :class:`TemplateLibrary`
        The library, which supplies the native grid.
    galaxy : :class:`~dc3.spectra.GalaxySpectra`
        The galaxy, which supplies the sampling.
    velscale_ratio : int
        Output pixels per galaxy pixel.

    Returns
    -------
    :class:`~dc3.core.resample.Resample`
        The resampled spectra and the output grid.
    """
    return resample.Resample(
        flux, xBorders=library.grid.borders, inLog=library.grid.is_log,
        newRange=[library.wave[0], library.wave[-1]],
        newdx=galaxy.dloglam / velscale_ratio, newLog=True
    )


def _resample_variance(library, out_borders, out_velscale, idsp, method):
    r"""
    Predict the variance resampling adds to the line-spread function.

    :class:`~dc3.core.resample.Resample` treats each input pixel as a step, and
    an output border that falls a fraction :math:`\phi` of the way into a
    native pixel of width :math:`\Delta_{\rm tpl}` adds
    :math:`\phi(1 - \phi)\,\Delta_{\rm tpl}^2` of variance to the lines it
    splits; see that class.

    - ``'expectation'``: the average over a uniform offset between the grids,
      :math:`\Delta_{\rm tpl}^2/6` at every wavelength.
    - ``'local'``: :math:`\phi(1 - \phi)` at each output border, from the
      offsets the grids actually have, averaged over the borders a line spans.
      The average is weighted by a Gaussian of the line's own dispersion in
      output pixels, since each border's contribution is in proportion to the
      line flux it splits.

    Parameters
    ----------
    library : :class:`TemplateLibrary`
        The library, on its native grid.
    out_borders : :class:`numpy.ndarray`
        The borders of the output pixels, in angstroms.
    out_velscale : float
        The width of an output pixel, in km/s.
    idsp : :class:`numpy.ndarray`
        The dispersion of the prepared lines at each native pixel, in km/s,
        which sets the span of the average.
    method : str
        ``'expectation'`` or ``'local'``.

    Returns
    -------
    :class:`numpy.ndarray`
        The predicted variance at each native pixel, in :math:`({\rm km/s})^2`.
    """
    pixel2 = np.square(library.pixel_velocity)
    if method == 'expectation':
        return np.full(library.npix, 1 / 6) * pixel2

    # The fraction of the way into its native pixel at which each output border
    # falls.  A border beyond the native range splits nothing, and is left out
    # of the average.
    borders = library.grid.borders
    inside = (out_borders >= borders[0]) & (out_borders <= borders[-1])
    index = np.interp(out_borders, borders, np.arange(borders.size, dtype=float))
    phase = index - np.floor(index)
    split = phase * (1 - phase)

    # Averaged over the borders each line spans, weighted by its profile
    width = np.interp(out_borders, library.wave, idsp) / out_velscale
    position = np.arange(out_borders.size, dtype=float)
    average = np.empty(out_borders.size, dtype=float)
    for j in range(out_borders.size):
        half = int(np.ceil(4 * width[j]))
        lo, hi = max(j - half, 0), min(j + half + 1, out_borders.size)
        weight = np.exp(-0.5 * np.square((position[lo:hi] - j) / width[j])) * inside[lo:hi]
        average[j] = np.sum(weight * split[lo:hi]) / np.sum(weight)
    return np.interp(library.wave, out_borders, average) * pixel2


def _masked_runs(gpm):
    """
    Return the runs of masked pixels in one spectrum.

    Parameters
    ----------
    gpm : :class:`numpy.ndarray`
        The good-pixel mask of one spectrum.

    Returns
    -------
    tuple
        Two integer arrays: the first masked pixel of each run, and the pixel
        after its last, which are also the native borders that bound it.
    """
    edges = np.diff(np.concatenate([[0], np.logical_not(gpm).astype(int), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


@document_parameters(TemplatePar)
def prepare(
    library, galaxy, velscale_ratio=1, epsilon_sigma=0.1, sigma_floor=0.0,
    mask_unmatched_idsp=False, varsmooth_oversample=2, correct_lsf_excess=True,
    resample_excess_method='local', convolution_mask_growth=3.0, fiducial_method='median'
):
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
        The raw templates, on a grid of any kind.
    galaxy : :class:`~dc3.spectra.GalaxySpectra`
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
        - The library's sampling breaks it into more than 5 segments, which
          more likely means its wavelengths are noisy than that it is spliced
          that many times; see :attr:`TemplateLibrary.sampling_breaks`.

    Notes
    -----
    **Correcting for the excess of both steps.**  Each step broadens the
    line-spread function beyond what it is asked to, by about a sixth of the
    native template pixel squared in variance: the variable-dispersion
    convolution through its own interpolation (see
    :func:`~dc3.core.resolution.varsmooth_excess`), and the resampling by
    treating each pixel as a step (see :class:`~dc3.core.resample.Resample`).
    With ``correct_lsf_excess``, both are accounted for, so that the prepared
    templates carry the target resolution in the second moment:

    - The variance resampling will add is predicted, at each native pixel, on
      the output grid it will use, by ``resample_excess_method``, and
      subtracted from the target passed to the matching.
    - The matching chooses the kernel whose variance, with the convolution's
      own excess, makes up the difference; see
      :func:`~dc3.core.resolution.match_resolution`.

    The dispersion reported is what the templates carry: the target offset by
    ``dvar_inst`` where the matching succeeds, and, where it cannot -- where
    the templates, with the excess of both steps, are already broader than the
    target -- their own dispersion with that excess.  Without the correction,
    the kernel alone is assumed to change the resolution, as in ``mangadap``,
    and the target offset by ``dvar_inst`` is reported everywhere.

    **Jumps in the library.**  A library spliced from sections sampled, or
    observed, differently is treated as piecewise smooth: smooth within each
    segment, with breaks where the pixel size
    (:attr:`TemplateLibrary.sampling_breaks`) or the instrumental dispersion
    (:attr:`TemplateLibrary.idsp_breaks`) jumps.  Within a segment everything
    above holds.  At a break it does not, and the affected pixels are masked
    rather than modelled:

    - Step 1 convolves in pixel space, so a kernel whose footprint crosses a
      sampling break treats pixels of two sizes as equal; and at a resolution
      break the coordinate stretch of the variable-dispersion convolution
      changes abruptly.  Each break is therefore masked, grown by
      ``convolution_mask_growth`` kernel dispersions either side, and flagged
      ``SAMP_JUMP`` or ``RES_JUMP`` according to its kind; both if it is both.
    - Step 2 resamples the whole spectrum at once from the library's pixel
      borders, and an output pixel straddling a break mixes the two segments.
      It lies within the grown mask; its instrumental dispersion is taken
      from the segment covering more of it, and ``UNMATCHED`` is set if it is
      set on either side.  Elsewhere the dispersion and ``UNMATCHED`` are
      interpolated within each segment, never across a break.

    Deliberately **not** guarded: changes below the tolerances, which are
    treated as smooth however large their total; the galaxy, which must be
    logarithmically sampled and enters only through the fiducial resolution;
    and the second-order error of convolving in pixel space where the pixel
    size changes smoothly within a segment.  No correction is made for the
    effective kernel at a break; its pixels are only masked.

    **Masked pixels in the library.**  A library's own mask is not carried
    through the convolution, which spreads every pixel over the kernel's
    footprint whether it is masked or not.  The masked pixels are therefore
    convolved as they are, and each run of them, in each template, is masked in
    the prepared templates, grown by ``convolution_mask_growth`` kernel
    dispersions either side, and flagged ``TPL_MASKED``.  This is what a gap
    between spliced sections, padded with zeros, needs.  Their flux must be
    finite: a non-finite value anywhere would make the whole convolved
    spectrum non-finite.  Non-finite flux is to be rejected when a library is
    read, and is not handled here.
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
        resolution.check_pixelization(library.idsp, library.pixel_velocity, label='templates')
    if galaxy.idsp is not None:
        resolution.check_pixelization(galaxy.idsp, galaxy.velscale, label='galaxy spectra')

    # Jumps in the library's sampling and resolution, around which the
    # smoothness the two steps assume does not hold
    sampling_breaks = library.sampling_breaks
    idsp_breaks = library.idsp_breaks
    nsegments = _check_splices(sampling_breaks, library.sampling_jump_tol)
    log.info(
        f'Template library {library.key!r}: {nsegments} sampling segment(s), '
        f'{_count_jumps(idsp_breaks)} jump(s) in resolution.'
    )

    # --- The target resolution ----------------------------------------------
    target = None
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
        # The templates keep whatever dispersion they carry, which may be none.
        nominal_idsp = None if library.idsp is None else library.idsp[0]
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
        # The resolution the prepared templates nominally carry, the target
        # offset by a constant, which sets the output sampling.  The match that
        # gives the offset is repeated below, with any correction, and its
        # warnings with it.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nominal = resolution.match_resolution(
                library.idsp[0], target, library.pixel_velocity,
                epsilon_sigma=epsilon_sigma, sigma_floor=sigma_floor
            )
        nominal_idsp = np.sqrt(np.square(target) - nominal.dvar_inst)

    # --- Choose the output sampling, and check it --------------------------
    if velscale_ratio == 'auto':
        if nominal_idsp is None:
            warnings.warn(
                'velscale_ratio = "auto" needs the instrumental dispersion of the templates, '
                'which they do not carry.  Using a ratio of 1.'
            )
            _velscale_ratio = 1
        else:
            _velscale_ratio = resolution.minimum_velscale_ratio(nominal_idsp, galaxy.velscale)
    else:
        _velscale_ratio = velscale_ratio
        if nominal_idsp is not None:
            needed = resolution.minimum_velscale_ratio(nominal_idsp, galaxy.velscale)
            if _velscale_ratio < needed:
                warnings.warn(
                    f'At velscale_ratio = {_velscale_ratio} the prepared templates are not '
                    'Nyquist-sampled: the FWHM of their line-spread function spans fewer than '
                    'two pixels somewhere in the range, so resampling degrades their '
                    f'effective resolution.  Use velscale_ratio = {needed} or more, or "auto".'
                )

    # --- The variance Step 2 will add ---------------------------------------
    # Predicted on the output grid Step 2 will use, which does not depend on
    # the flux, so it is known before Step 1 runs.
    resample_variance = None
    if correct_lsf_excess and nominal_idsp is not None:
        out_borders = _resample(
            np.zeros(library.npix), library, galaxy, _velscale_ratio
        ).outborders
        resample_variance = _resample_variance(
            library, out_borders, galaxy.velscale / _velscale_ratio, nominal_idsp,
            resample_excess_method
        )

    # --- Step 1: match the resolution on the templates' native grid ---------
    if target is None:
        match = resolution.ResolutionMatch.identity(
            library.npix, library.pixel_velocity, epsilon_sigma=epsilon_sigma
        )
        matched_flux = library.flux
        matched_idsp = nominal_idsp
        if resample_variance is not None:
            # Without a matching there is no kernel to correct, but the
            # dispersion reported is still what the templates carry
            matched_idsp = np.sqrt(np.square(nominal_idsp) + resample_variance)
    elif correct_lsf_excess:
        # The target less what Step 2 will add, so that Step 2 brings the
        # templates to it.  Where Step 2 alone exceeds the target, no kernel can
        # reach it; a vanishing target makes those pixels unmatched.
        reduced = np.square(target) - resample_variance
        idsp_to = np.sqrt(np.maximum(reduced, np.square(1e-6 * target)))
        match = resolution.match_resolution(
            library.idsp[0], idsp_to, library.pixel_velocity, epsilon_sigma=epsilon_sigma,
            sigma_floor=sigma_floor, varsmooth_oversample=varsmooth_oversample
        )
        matched_flux = np.atleast_2d(resolution.apply_kernel(
            library.flux, match, oversample=varsmooth_oversample
        ))
        # What the templates carry after both steps: the target offset by a
        # constant where the matching succeeds, and more where it cannot
        matched_idsp = np.sqrt(np.square(match.achieved) + resample_variance)
    else:
        match = resolution.match_resolution(
            library.idsp[0], target, library.pixel_velocity,
            epsilon_sigma=epsilon_sigma, sigma_floor=sigma_floor
        )
        matched_flux = np.atleast_2d(resolution.apply_kernel(
            library.flux, match, oversample=varsmooth_oversample
        ))
        # The kernel alone assumed to change the resolution, as in mangadap:
        # the target offset by a constant, everywhere.
        matched_idsp = np.sqrt(np.square(target) - match.dvar_inst)

    # --- Step 2: resample onto the galaxy's sampling -----------------------
    resampled = _resample(matched_flux, library, galaxy, _velscale_ratio)
    out_flux = np.atleast_2d(resampled.outy)
    # Only the starting wavelength is taken from the resampled grid.  The pixel
    # size is required to be exactly the galaxy's divided by the ratio -- that
    # is what makes the offset between the two grids a constant velocity -- so
    # it is set directly rather than recovered with round-off from the output
    # wavelengths.
    out_grid = sampling.SpectralGrid.from_log_spacing(
        np.log10(resampled.outx[0]), galaxy.dloglam / _velscale_ratio, resampled.outx.size
    )

    # The instrumental dispersion is a property of each wavelength, not an
    # integrated quantity, so it is interpolated rather than resampled -- and
    # never across a jump in the sampling or the resolution, where it would
    # give a pixel straddling the jump a value neither side has.
    segment_breaks = np.union1d(sampling_breaks, idsp_breaks)
    overlap, majority = _segment_overlap(
        resampled.outborders, library.grid.borders, segment_breaks
    )
    out_idsp = (
        None if matched_idsp is None
        else _interp_within_segments(
            resampled.outx, library.wave, matched_idsp, segment_breaks, majority
        )
    )

    mask = SpectrumMask(out_flux.shape)
    # outf is the fraction of each output pixel covered by valid input, so a
    # value of zero means the pixel drew on nothing.
    mask.turn_on('NODATA', select=np.broadcast_to(resampled.outf <= 0, out_flux.shape))
    if mask_unmatched_idsp and np.any(match.unmatched):
        unmatched = _flag_within_segments(
            resampled.outx, library.wave, match.unmatched, segment_breaks, overlap
        )
        mask.turn_on('UNMATCHED', select=np.broadcast_to(unmatched, out_flux.shape))

    # The regions the convolution cannot treat correctly, grown by the kernel's
    # footprint: each jump, where the smoothness Steps 1 and 2 assume does not
    # hold, and each run of pixels masked in the library.  Always applied, unlike
    # UNMATCHED.
    for flag, breaks in [('SAMP_JUMP', sampling_breaks), ('RES_JUMP', idsp_breaks)]:
        grown = _grow_regions(
            resampled.outborders, library.grid.borders, breaks, breaks,
            match.kernel_sigma_pixels, convolution_mask_growth
        )
        mask.turn_on(flag, select=np.broadcast_to(grown, out_flux.shape))
    # The library's mask is per template, so its runs are grown one template
    # at a time
    library_gpm = library.gpm
    tpl_masked = np.zeros(out_flux.shape, dtype=bool)
    for i in range(library.nspec):
        starts, ends = _masked_runs(library_gpm[i])
        tpl_masked[i] = _grow_regions(
            resampled.outborders, library.grid.borders, starts, ends,
            match.kernel_sigma_pixels, convolution_mask_growth
        )
    mask.turn_on('TPL_MASKED', select=tpl_masked)

    return PreparedTemplates(
        out_flux, out_grid, match=match, velscale_ratio=_velscale_ratio,
        key=preparation_key(
            library.key, fiducial, galaxy.velscale, _velscale_ratio, epsilon_sigma,
            sigma_floor, varsmooth_oversample, convolution_mask_growth=convolution_mask_growth,
            sampling_jump_tol=library.sampling_jump_tol, idsp_jump_tol=library.idsp_jump_tol,
            correct_lsf_excess=correct_lsf_excess, resample_excess_method=resample_excess_method
        ),
        mask=mask, idsp=out_idsp,
    )
