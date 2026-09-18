"""
The ``dc3`` parameter sets.

These are the parameters that govern a ``dc3`` run.  They are declared by hand,
with full types, options and descriptions, because ``dc3`` owns them; parameters
belonging to a wrapped third-party function are declared instead as a
:class:`~dc3.par.funcpar.FuncPar`.

**A parameter set is declared in the module whose code it configures**, not
here: :class:`~dc3.templates.TemplatePar` lives in :mod:`dc3.templates` beside
:func:`~dc3.templates.prepare`, which it is expanded over.  This module holds
the top-level :class:`DC3Par` that nests them, so it necessarily imports every
module that declares one.  The arrangement is ``mangadap``'s rather than
PypeIt's, and it is what lets a module that wraps a third-party function declare
the corresponding :class:`~dc3.par.funcpar.FuncPar` without a central file
having to import that dependency.  The sets that remain below are those whose
module is not yet written; each moves out as its phase lands.

Relative to the C++ implementation, this exposes a good deal that used to be
**hardwired** in ``DC3_express.cpp`` and therefore unreachable: the apodization
window, the sub-Nyquist convolution floor, the fit-window factor, and the
Levenberg-Marquardt parameter scales.  ``doc/develop.txt`` flags several of
these as "NEED TO REVISIT" -- notably the fit-window width, found empirically
optimal at 1.0-1.7 CC-peak FWHM but defaulted to 2.0 -- which cannot be revisited
while they are compiled in.

Where the C++ encoded two meanings in one number, the port splits them: the
sign of ``mvdiff`` used to select its units, and a negative ``miter`` used to
mean "iterate to convergence".  Both are now separate, explicit parameters.
Backwards compatibility with the old parameter files is not a goal, and the old
format survives only as a test-fixture ingest path.

.. include:: ../include/links.rst
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .parset import ParSet
from ..templates import TemplateLibraryPar, TemplatePar


__all__ = [
    'ConvolvePar',
    'CorrelatePar',
    'MaskPar',
    'WindowPar',
    'ContinuumPar',
    'FitPar',
    'QAPar',
    'DC3Par',
]


class ConvolvePar(ParSet):
    """
    Parameters governing convolution of the template by the broadening function.

    Replaces the ``cnvlvpar`` struct of the C++ implementation.
    """

    default_key = 'convolve'
    card_prefix = 'CNV'
    api_doc = ':class:`~dc3.par.dc3par.ConvolvePar`'

    padtype: Annotated[Literal['reflect', 'wrap', 'nearest', 'constant'], Field(
        default='reflect',
        description='How the spectrum is extended beyond its ends before convolution.'
    )]
    pad_value: Annotated[float, Field(
        default=1.0,
        description='Value used to pad the spectrum when padtype is "constant".  Ignored '
                    'otherwise.  This was called "contin" in the C++ parameter file.'
    )]
    min_sigma: Annotated[float, Field(
        default=0.85, gt=0.0,
        description='Smallest kernel dispersion, in pixels, that can be convolved directly.  '
                    'Below this the spectrum is block-replicated, convolved, and averaged back '
                    'down.  The default of 0.85 px corresponds to a FWHM of 2 px, i.e. Nyquist '
                    'sampling of the kernel.  Hardwired in the C++ implementation.'
    )]
    max_block: Annotated[int, Field(
        default=8, ge=1,
        description='Maximum block-replication factor used for sub-Nyquist kernels.  Together '
                    'with min_sigma this sets a hard floor on the measurable dispersion at '
                    'min_sigma/max_block pixels; below it the fitted dispersion is set to zero.'
    )]
    taper: Annotated[int, Field(
        default=0, ge=0,
        description='Number of spectral pixels zeroed at each edge before correlation.'
    )]


class CorrelatePar(ParSet):
    """
    Parameters governing construction of the cross-correlation function.
    """

    default_key = 'correlate'
    card_prefix = 'XCOR'
    api_doc = ':class:`~dc3.par.dc3par.CorrelatePar`'

    apodization: Annotated[
        Literal['none', 'cosine', 'hann', 'bartlett', 'blackman', 'hamming'], Field(
            default='none',
            description='Window applied to each unmasked block before the transform.  The '
                        'default of "none" is an empirical result recorded in the original '
                        'doc/develop.txt: for spectra with few lines, apodizing is worse than '
                        'not, and otherwise the choice of window matters little.'
        )
    ]
    cosine_percent: Annotated[float, Field(
        default=2.0, ge=0.0, le=100.0,
        description='Percentage of each block tapered by the cosine-bell window.  Ignored '
                    'unless apodization is "cosine".'
    )]
    length_factor: Annotated[float, Field(
        default=2.2, ge=2.0,
        description='Multiplier setting the padded transform length from the common wavelength '
                    'range.  The factor of two is required to treat one spectrum as the '
                    'response function for the other; the extra 10% is deliberate padding.'
    )]
    exact_cpp_length: Annotated[bool, Field(
        default=False,
        description='Reproduce the C++ transform length exactly, rather than rounding up to a '
                    'fast FFT length.  Rounding moves the velocity grid, since its zero point '
                    'is set by the transform length, so this exists for comparison against '
                    'archived C++ products and for nothing else.'
    )]
    censor_sigma: Annotated[float, Field(
        default=6.0, gt=0.0,
        description='Number of broadening sigmas trimmed from each end of the common '
                    'wavelength range, to keep convolution wrap-around out of the correlation.'
    )]


class MaskPar(ParSet):
    """
    Parameters governing spectral masking.
    """

    default_key = 'mask'
    card_prefix = 'MSK'
    api_doc = ':class:`~dc3.par.dc3par.MaskPar`'

    regions: Annotated[str | None, Field(
        default=None,
        description='Path to the table of spectral regions to mask, each given in either the '
                    'galaxy or the template frame.'
    )]
    grow_sigma: Annotated[float, Field(
        default=2.0, ge=0.0,
        description='Number of broadening sigmas by which a masked region grows or shrinks '
                    'when transcribed between the galaxy and template frames.  The region '
                    'width changes by twice this, times the broadening dispersion.'
    )]


class WindowPar(ParSet):
    """
    Parameters governing the fitting window on the cross-correlation function.
    """

    default_key = 'window'
    card_prefix = 'WIN'
    api_doc = ':class:`~dc3.par.dc3par.WindowPar`'

    type: Annotated[Literal['xzero', 'pmin', 'nfwhm', 'fixv', 'nvsig', 'full'], Field(
        default='nfwhm',
        description='How the fitting window is derived.  "nfwhm" spans a multiple of the '
                    'CC-peak FWHM; "nvsig" tracks the fitted dispersion; "fixv" uses a fixed '
                    'velocity range; "xzero" and "pmin" run to the zero crossings or minima '
                    'bracketing the peak; "full" uses the whole function.'
    )]
    nfwhm: Annotated[float, Field(
        default=2.0, gt=0.0,
        description='Width of the fitting window in CC-peak FWHM, used when type is "nfwhm".  '
                    'The original experiments found 1.0-1.7 optimal and the value is flagged '
                    'for revision in doc/develop.txt, but 2.0 is what the published results '
                    'used, so it remains the default until re-derived.'
    )]
    winfac: Annotated[float, Field(
        default=2.5, gt=0.0,
        description='Multiplier giving the wider window over which the covariance is built, '
                    'relative to the fitting window, when type is "nvsig".'
    )]
    fixed_range: Annotated[tuple[float, float] | None, Field(
        default=None,
        description='Velocity range, in km/s, used when type is "fixv".'
    )]
    min_width: Annotated[int, Field(
        default=30, ge=1,
        description='Smallest allowed window, in pixels.  This matters for noisy data, where '
                    'the derived window can otherwise collapse to a few pixels.'
    )]
    smooth_bins: Annotated[int, Field(
        default=0, ge=0,
        description='Size of the smoothing applied to the cross-correlation function before '
                    'the window is derived from it.  Zero disables smoothing.'
    )]


class ContinuumPar(ParSet):
    """
    Parameters governing the continuum added to the broadened template.

    The continuum is fit to the difference between the galaxy and the broadened
    template and **added to the template**, not subtracted from the
    cross-correlation function.
    """

    default_key = 'continuum'
    card_prefix = 'CONT'
    api_doc = ':class:`~dc3.par.dc3par.ContinuumPar`'

    basis: Annotated[Literal['legendre', 'bspline'], Field(
        default='legendre',
        description='Functional basis for the continuum.  "legendre" reproduces the published '
                    'algorithm; "bspline" is offered because the continuum here is an '
                    'instrumental residual rather than a physical one, and a low-order global '
                    'polynomial models it poorly over a wide wavelength range.'
    )]
    order: Annotated[int, Field(
        default=0, ge=0,
        description='Order of the Legendre polynomial, or equivalently the stiffness of the '
                    'spline.  Zero disables continuum fitting entirely.  Note that the '
                    'constant term is always held fixed, because the spectrum means are '
                    'subtracted before correlation and a free zeroth order destabilizes the fit.'
    )]
    iterations: Annotated[int, Field(
        default=0, ge=0,
        description='Maximum number of continuum-fitting iterations.  Zero disables continuum '
                    'fitting entirely.'
    )]
    reject_low: Annotated[float, Field(
        default=3.0, ge=0.0,
        description='Lower sigma-clipping threshold used when fitting the continuum.'
    )]
    reject_high: Annotated[float, Field(
        default=3.0, ge=0.0,
        description='Upper sigma-clipping threshold used when fitting the continuum.'
    )]
    reject_iterations: Annotated[int, Field(
        default=10, ge=0,
        description='Maximum number of rejection iterations within a single continuum fit.'
    )]

    @model_validator(mode='after')
    def _order_and_iterations_agree(self):
        """
        Reconcile the two ways of disabling the continuum.

        The C++ implementation silently forced each of these to zero when the
        other was, which meant a configuration could assert a continuum order
        that was never used.  Here the inconsistency is reported instead.
        """
        if (self.order == 0) != (self.iterations == 0):
            raise ValueError(
                'continuum.order and continuum.iterations must be zero together or nonzero '
                f'together; got order={self.order} and iterations={self.iterations}.  Zero in '
                'either one disables continuum fitting, so this configuration is ambiguous.'
            )
        return self


class FitPar(ParSet):
    """
    Parameters governing the kinematic fit.

    The three iteration tiers of the published algorithm map onto these
    parameters as: ``restarts`` is tier 1; ``mask_iterations`` and
    ``mask_vdiff`` are tier 2; the continuum tier is
    :class:`ContinuumPar`.
    """

    default_key = 'fit'
    card_prefix = 'FIT'
    api_doc = ':class:`~dc3.par.dc3par.FitPar`'

    method: Annotated[Literal['lsq', 'de', 'mcmc'], Field(
        default='lsq',
        description='Optimizer or sampler used for the parameter estimation.  All three share '
                    'one objective function.  "lsq" is least squares and is the production '
                    'path; "de" is differential evolution, a global optimizer that replaces '
                    'the randomized-restart workaround; "mcmc" samples the posterior and '
                    'returns errors from it rather than from the precision matrix.'
    )]
    moments: Annotated[int, Field(
        default=2, ge=2, le=6,
        description='Number of line-of-sight velocity distribution moments to fit.  Two is '
                    'velocity and dispersion; more adds Gauss-Hermite terms.  Higher moments '
                    'need a signal-to-noise ratio near the upper limit of what this method '
                    'targets, and are only interpretable when the prepared templates are '
                    'matched in resolution.'
    )]
    solve_norm: Annotated[bool, Field(
        default=True,
        description='Solve the model amplitude analytically rather than fitting it.  This '
                    'removes a parameter from the nonlinear problem and measured about 25% '
                    'faster in the archived production run.  Ignored when more than one '
                    'template is used, since the template weights absorb the amplitude.'
    )]
    restarts: Annotated[int, Field(
        default=5, ge=1,
        description='Number of improved solutions sought from randomized restarts, guarding '
                    'against local minima.  Five times this many attempts are made.'
    )]
    max_nfev: Annotated[int, Field(
        default=50, ge=1,
        description='Maximum number of objective-function evaluations per optimizer call.'
    )]
    mask_iterations: Annotated[int | None, Field(
        default=0, ge=0,
        description='Number of mask-transcription iterations.  Set to null to iterate until '
                    'the mask and fit velocities agree to within mask_vdiff instead of a fixed '
                    'count.  The C++ implementation encoded that choice as a negative count.'
    )]
    mask_vdiff: Annotated[float, Field(
        default=1.0, ge=0.0,
        description='Convergence criterion for the mask-transcription tier: the fit is '
                    'converged when the fitted and mask velocities agree to within this.'
    )]
    mask_vdiff_unit: Annotated[Literal['km/s', 'pixel'], Field(
        default='km/s',
        description='Unit in which mask_vdiff is expressed.  The C++ implementation encoded '
                    'this in the *sign* of mask_vdiff, which is split out here.'
    )]
    sigma_max: Annotated[float, Field(
        default=600.0, gt=0.0,
        description='Upper clamp on the fitted observed dispersion, in km/s.'
    )]
    x_scale: Annotated[list[float], Field(
        default=[1.0, 100.0, 100.0],
        description='Characteristic scale of each fitted parameter -- amplitude, velocity and '
                    'dispersion -- used to condition the optimizer.  These are the values '
                    'given in the published description of the algorithm.'
    )]
    constraints: Annotated[str | None, Field(
        default=None,
        description='Path to a table fixing kinematic parameters, or selecting a subset of '
                    'spectra to fit.  One row per spectrum; missing rows are fit normally.'
    )]


class QAPar(ParSet):
    """
    Parameters governing quality-assessment plots.
    """

    default_key = 'qa'
    card_prefix = 'QA'
    api_doc = ':class:`~dc3.par.dc3par.QAPar`'

    level: Annotated[Literal['none', 'summary', 'standard', 'full'], Field(
        default='standard',
        description='How much is plotted.  "none" is the production default for large runs: '
                    'plotting was 41% of the wall clock in the archived production campaign, '
                    'and any figure can be regenerated afterwards from the results file, so '
                    'turning plots off costs nothing.'
    )]


class DC3Par(ParSet):
    """
    The top-level ``dc3`` parameter set.
    """

    default_key = 'dc3'
    card_prefix = 'DC3'
    api_doc = ':class:`~dc3.par.dc3par.DC3Par`'
    default_comment = 'Parameters for a dc3 run.'

    library: Annotated[TemplateLibraryPar, Field(
        default_factory=TemplateLibraryPar, description='The stellar template library to use.'
    )]
    template: Annotated[TemplatePar, Field(
        default_factory=TemplatePar, description='Template preparation.'
    )]
    convolve: Annotated[ConvolvePar, Field(
        default_factory=ConvolvePar, description='Convolution by the broadening function.'
    )]
    correlate: Annotated[CorrelatePar, Field(
        default_factory=CorrelatePar, description='Cross-correlation construction.'
    )]
    mask: Annotated[MaskPar, Field(
        default_factory=MaskPar, description='Spectral masking.'
    )]
    window: Annotated[WindowPar, Field(
        default_factory=WindowPar, description='The fitting window.'
    )]
    continuum: Annotated[ContinuumPar, Field(
        default_factory=ContinuumPar, description='The added continuum.'
    )]
    fit: Annotated[FitPar, Field(
        default_factory=FitPar, description='The kinematic fit.'
    )]
    qa: Annotated[QAPar, Field(
        default_factory=QAPar, description='Quality-assessment plots.'
    )]
    output_dir: Annotated[Path, Field(
        default_factory=Path.cwd,
        description='Directory into which output products are written.'
    )]
    ncpu: Annotated[int, Field(
        default=1, ge=1,
        description='Number of processes used to fit spectra in parallel.  Fitting is '
                    'embarrassingly parallel over spectra.'
    )]

    @model_validator(mode='after')
    def _moments_require_matched_resolution(self):
        """
        Gauss-Hermite moments require matched template resolution.

        The Gauss-Hermite parameterization is defined relative to the fitted
        Gaussian, so the higher moments are only interpretable when the prepared
        templates carry no instrumental pedestal.  This fails validation rather
        than warning, because a warning here produces numbers that look fine and
        are not.
        """
        if self.fit.moments > 2 and self.template.sigma_floor > 0.0:
            raise ValueError(
                f'fit.moments = {self.fit.moments} requires matched template resolution, but '
                f'template.sigma_floor = {self.template.sigma_floor} permits a non-zero '
                'instrumental offset.  Gauss-Hermite moments are defined relative to the '
                'fitted Gaussian and are not interpretable with a pedestal.'
            )
        return self
