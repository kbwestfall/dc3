r"""
Matching the spectral resolution of the templates to the galaxy data.

This implements the two rules that govern resolution in ``dc3``:

1. **The galaxy is data and is never altered.**  Only the templates are
   prepared.
2. **Matching is preparation, never part of the cost function.**  A
   wavelength-dependent convolution is far more expensive than a
   wavelength-independent one, so inside the fit the broadening kernel is a
   single scalar :math:`\sigma`.  Everything wavelength-dependent happens once,
   up front.

A third follows from the second: **no deconvolution, ever**.  It amplifies
noise, so the preparation kernel must be real at every wavelength.  That is a
constraint on the construction below, not a check applied afterwards.

Dispersion, not resolving power
-------------------------------

``dc3`` works in instrumental **dispersion** :math:`\sigma` in km/s throughout.
A resolving power :math:`R` supplied by a user is converted at ingest, by
:func:`dispersion_from_resolving_power`; nothing downstream ever sees
:math:`R`.

The signed instrumental variance
--------------------------------

The relation between the fitted and astrophysical dispersions is carried as a
**signed variance** in :math:`({\rm km/s})^2`, with :math:`\sigma_{T}'` the
template resolution *after* preparation:

.. math::

    {\rm dvar\_inst} \equiv \sigma_G^2 - \sigma_{T}'^2
    \qquad
    \sigma_{\rm obs}^2 = \sigma_\ast^2 + {\rm dvar\_inst}

A signed variance rather than a signed dispersion keeps one convention across
both regimes, removes the square roots from the bookkeeping, and avoids an
imaginary intermediate when the quantity is negative.

============  ==========================  =====================================
dvar_inst     Template resolution         Consequence
============  ==========================  =====================================
:math:`> 0`   higher than the galaxy      :math:`\sigma_{\rm obs}` is held away
                                          from the :math:`\sigma \to 0`
                                          boundary
:math:`= 0`   equal to the galaxy         no pedestal either way
:math:`< 0`   lower than the galaxy       a **floor** on the measurable
                                          :math:`\sigma_\ast`; a warning is
                                          issued
============  ==========================  =====================================

Both signs are legitimate.  A survey whose templates are of higher resolution
than its data can take a positive offset; one that observed its templates with
the *same* instrument usually cannot, and must accept a floor.  The code tracks
the sign, reports it, and warns when it is negative -- it never silently returns
:math:`\sigma = 0`.

When matching does not happen
-----------------------------

Matching needs both resolution vectors.  If either the templates or the galaxy
carry none, there is nothing to match, and preparation proceeds without it: the
templates are used at their native resolution and ``dvar_inst`` is zero.  That
zero is a *statement of ignorance*, not a measurement of equality, so a warning
says the reported dispersions are uncorrected for any resolution difference.
:meth:`ResolutionMatch.identity` represents the case, and
:attr:`ResolutionMatch.performed` distinguishes it from a genuine match that
happened to leave no offset.

There is deliberately no switch to decline matching when both vectors *are*
available.  An unmatched, wavelength-dependent resolution difference cannot be
represented by the single scalar dispersion of the forward model.

How this differs from ``mangadap``
----------------------------------

The reference implementation is
``mangadap.util.resolution.SpectralResolution.GaussianKernelDifference``, which
implements the same W19 Appendix A matching.  This is a considerable
simplification of it, for reasons that are all specific to ``dc3``:

- **Dispersion, not resolving power.**  ``mangadap`` stores :math:`R` and
  converts to :math:`\sigma` at every step; here the conversion happens once, at
  ingest, so it disappears from the matching entirely.
- **One coordinate system, not three.**  ``mangadap`` moves between variance in
  :math:`\unicode{x212B}^2`, in :math:`({\rm km/s})^2` and in pixels, with a
  linear/logarithmic branch in each conversion.  ``dc3`` is always
  logarithmically sampled, so km/s to pixels is a division by one scalar and
  the wavelength-coordinate system is never needed.
- **A returned result, not mutated state.**  ``GaussianKernelDifference``
  writes four attributes onto the object, which a later call reads and a
  further call mutates again; calling them out of order raises.  Here the
  calculation is a function returning :class:`ResolutionMatch`.
- **A signed variance, not a signed dispersion.**  ``mangadap`` encodes a signed
  :math:`\sigma` as :math:`\sigma^2/\sqrt{|\sigma^2|}` so that negative values
  survive.  Working in variance throughout means the square root is taken once,
  where the kernel is known to be real.
- **No fudge factor.**  ``mangadap`` multiplies its offset by 1.01 so that the
  pixel exactly at the minimum is not masked.  Here the minimum kernel is
  constructed to equal ``epsilon_sigma`` exactly, so that boundary case cannot
  arise.
- **One two-sided target, not a flag over two options.**  ``mangadap``'s
  ``no_offset`` selects between trimming and a constant offset, and its offset
  is one-sided -- ``min(0.0, ...)`` makes a *positive* pedestal unreachable.
  See :func:`match_resolution`.
- **No convolution machinery.**  ``mangadap`` carries its own variable-sigma
  convolution, built from a banded matrix of *sampled* Gaussians.  ``dc3``
  convolves with :func:`ppxf.ppxf_util.varsmooth`, which stretches the
  coordinate so the kernel becomes constant and then convolves against the
  *analytic* Gaussian transform.  ``mangadap``'s routine is retained only as an
  independent cross-check.

.. include:: ../include/links.rst
"""

import warnings

import numpy as np
from ppxf import ppxf_util

from ..pkg.exceptions import DC3CodingError, DC3ResolutionError
from .velocity import SPEED_OF_LIGHT


__all__ = [
    'SIGMA_TO_FWHM',
    'VARSMOOTH_MIN_SIG',
    'ResolutionMatch',
    'apply_kernel',
    'dispersion_from_resolving_power',
    'match_resolution',
    'resolving_power_from_dispersion',
]


SIGMA_TO_FWHM = np.sqrt(8 * np.log(2))
r"""The ratio :math:`{\rm FWHM}/\sigma = \sqrt{8\ln 2}` for a Gaussian."""


VARSMOOTH_MIN_SIG = 0.1
r"""
The smallest kernel dispersion, in pixels, that
:func:`ppxf.ppxf_util.varsmooth` applies accurately.

``varsmooth`` clips its kernel to this internally (``sig = sig.clip(0.1)``).
The clip is **not** in the published algorithm and is not exported as API, so it
is recorded here and asserted behaviourally by the test suite: an upstream
change must fail loudly rather than silently biasing ``dvar_inst``.

The reason for the clip is the coordinate stretch, not the convolution.  The
algorithm rescales the abscissa by :math:`\sigma_{\rm max}/\sigma` so that a
variable-:math:`\sigma` kernel becomes constant, and that stretch diverges as
:math:`\sigma \to 0`; the clip bounds the number of resampled points.

.. warning::

    **Below the clip the behaviour is worse than the clip alone implies.**  A
    request under 0.1 pixels is not honoured, and neither is it raised to 0.1:
    measured against a resolved Gaussian, *any* request below the clip yields a
    realised kernel of about **0.71 pixels** -- some seven times the clip
    itself.

    The cause is not the clip, and not undersampling.  ``varsmooth`` sizes its
    internal stretched grid as ``n = ceil(xs[-1] - xs[0])``, where
    ``xs = cumsum(sig_max/sig)``.  Because ``sig_max/sig >= 1`` by construction,
    that span is at least ``N - 1``, with equality **if and only if** ``sig`` is
    exactly uniform -- in which case ``ceil`` returns ``N - 1``, one sample
    short of the input, and the interpolation onto that shortened grid and back
    broadens the result.  The clip merely *causes* exact uniformity, by
    replacing every sub-0.1 value with the same literal, and so exposes the
    off-by-one.

    It is not confined to the clip: a genuinely uniform kernel on a grid whose
    ``numpy.gradient`` is exact triggers it at any width.  Requesting a uniform
    0.5-pixel kernel on ``x = numpy.arange(n)`` yields 0.87 pixels; perturbing
    one interior element of ``sig_x`` by one part in :math:`10^{12}` restores
    the correct 0.500.  See ``test_resolution.py``, which pins both regimes.

    ``dc3`` is insulated on both counts.  The sub-clip path is closed by
    :func:`match_resolution` refusing an ``epsilon_sigma`` below the clip, so
    ``sig`` is never clipped at all.  The uniform-kernel path stays open in
    principle but does not fire on a realistic logarithmic grid, where
    ``numpy.gradient`` carries floating-point noise -- which is luck rather than
    design, hence the regression test.
"""


def dispersion_from_resolving_power(resolving_power):
    r"""
    Convert a resolving power to an instrumental dispersion in km/s.

    With :math:`R = \lambda/{\rm FWHM}_\lambda` and a Gaussian line-spread
    function,

    .. math::

        \sigma_{\rm inst} = \frac{c}{\sqrt{8 \ln 2}\ R}.

    This is the conversion that belongs at **ingest**: a user who has :math:`R`
    converts here, and nothing inside ``dc3`` handles a resolving power.

    Parameters
    ----------
    resolving_power : float, :class:`numpy.ndarray`
        The resolving power :math:`R`.

    Returns
    -------
    float, :class:`numpy.ndarray`
        The instrumental dispersion in km/s.

    Raises
    ------
    DC3ResolutionError
        Raised if the resolving power is not positive.
    """
    _r = np.asarray(resolving_power, dtype=float)
    if np.any(_r <= 0):
        raise DC3ResolutionError('The resolving power must be positive.')
    return SPEED_OF_LIGHT / (SIGMA_TO_FWHM * _r)


def resolving_power_from_dispersion(dispersion):
    r"""
    Convert an instrumental dispersion in km/s to a resolving power.

    The inverse of :func:`dispersion_from_resolving_power`.  Provided for
    reporting, since resolving power is the more familiar way to quote an
    instrument's performance; it is not used internally.

    Parameters
    ----------
    dispersion : float, :class:`numpy.ndarray`
        The instrumental dispersion in km/s.

    Returns
    -------
    float, :class:`numpy.ndarray`
        The resolving power :math:`R`.

    Raises
    ------
    DC3ResolutionError
        Raised if the dispersion is not positive.
    """
    _d = np.asarray(dispersion, dtype=float)
    if np.any(_d <= 0):
        raise DC3ResolutionError('The instrumental dispersion must be positive.')
    return SPEED_OF_LIGHT / (SIGMA_TO_FWHM * _d)


class ResolutionMatch:
    r"""
    The kernel that prepares a template, and the offset it leaves behind.

    Returned by :func:`match_resolution`.  Everything needed to apply the
    matching and to interpret its result afterwards is here; nothing is left on
    the inputs.

    Parameters
    ----------
    kernel_sigma : :class:`numpy.ndarray`, None
        Dispersion of the Gaussian convolution kernel at each pixel, in km/s.
        Real and positive everywhere by construction.  None if no matching was
        performed; see :meth:`not_performed`.
    dvar_inst : float
        The signed instrumental variance left after matching, in
        :math:`({\rm km/s})^2`.  See the module documentation.
    unmatched : :class:`numpy.ndarray`
        Boolean, True where the target resolution could not be reached without
        deconvolution.  All False unless ``sigma_floor`` constrained the offset.
    velscale : float
        Velocity scale of the grid, in km/s per pixel.
    epsilon_sigma : float
        The requested minimum kernel dispersion, in pixels.

    Attributes
    ----------
    kernel_sigma : :class:`numpy.ndarray`, None
        As above.
    dvar_inst : float
        As above.
    unmatched : :class:`numpy.ndarray`
        As above.
    velscale : float
        As above.
    epsilon_sigma : float
        As above.
    """

    def __init__(self, kernel_sigma, dvar_inst, unmatched, velscale, epsilon_sigma):
        self.kernel_sigma = kernel_sigma
        self.dvar_inst = float(dvar_inst)
        self.unmatched = unmatched
        self.velscale = float(velscale)
        self.epsilon_sigma = float(epsilon_sigma)

    @classmethod
    def identity(cls, npix, velscale, epsilon_sigma=VARSMOOTH_MIN_SIG):
        """
        Represent a preparation in which no resolution matching was done.

        The identity preparation: the spectra are left at their native
        resolution.  Used when either resolution vector is missing.  There is
        no kernel, the offset is zero, and no pixel is flagged as unmatched --
        nothing was attempted, so nothing failed.

        Parameters
        ----------
        npix : int
            Number of pixels in the spectra that were not matched.
        velscale : float
            Velocity scale of their grid, in km/s per pixel.
        epsilon_sigma : float, optional
            Recorded for provenance only; it had no effect.

        Returns
        -------
        ResolutionMatch
            A match with :attr:`performed` False and ``dvar_inst`` zero.
        """
        return cls(None, 0.0, np.zeros(npix, dtype=bool), velscale, epsilon_sigma)

    @property
    def performed(self):
        """
        Whether resolution matching was actually done.

        Distinguishes a zero ``dvar_inst`` that was *measured* from one that is
        zero only because there was nothing to match.  The two mean different
        things for a reported dispersion: the first is corrected, the second is
        not.
        """
        return self.kernel_sigma is not None

    @property
    def kernel_sigma_pixels(self):
        """
        The kernel dispersion in pixels, which is what ``varsmooth`` clips.

        None if no matching was performed.
        """
        return None if self.kernel_sigma is None else self.kernel_sigma / self.velscale

    @property
    def sigma_floor(self):
        r"""
        The floor this offset imposes on the measurable :math:`\sigma_\ast`.

        Zero unless :attr:`dvar_inst` is negative, in which case it is
        :math:`\sqrt{|{\rm dvar\_inst}|}`: no astrophysical dispersion below it
        can be measured, because the prepared template is already that broad.
        """
        return 0.0 if self.dvar_inst >= 0 else float(np.sqrt(-self.dvar_inst))

    @property
    def n_unmatched(self):
        """The number of pixels where the target resolution could not be reached."""
        return int(np.sum(self.unmatched))

    def astrophysical_variance(self, sigma_obs):
        r"""
        Convert a fitted dispersion to the astrophysical one.

        Applies :math:`\sigma_\ast^2 = \sigma_{\rm obs}^2 - {\rm dvar\_inst}`.
        The **variance** is returned, signed, rather than a dispersion: a fitted
        dispersion below the floor gives a negative astrophysical variance, and
        reporting that honestly is more useful than clipping it to zero, which
        is what the original C++ did and which made an unmeasurable dispersion
        indistinguishable from a genuine one.

        Parameters
        ----------
        sigma_obs : float, :class:`numpy.ndarray`
            The fitted observed dispersion, in km/s.

        Returns
        -------
        float, :class:`numpy.ndarray`
            The astrophysical variance, in :math:`({\rm km/s})^2`.  May be
            negative.
        """
        return np.square(np.asarray(sigma_obs, dtype=float)) - self.dvar_inst

    def __repr__(self):
        """A short summary of the match."""
        if not self.performed:
            return f'<{type(self).__name__}: not performed, dvar_inst=0 (uncorrected)>'
        if self.dvar_inst > 0:
            sense = 'template at higher resolution'
        elif self.dvar_inst < 0:
            sense = 'template at lower resolution'
        else:
            sense = 'resolutions equal'
        return (
            f'<{type(self).__name__}: dvar_inst={self.dvar_inst:+.3f} (km/s)^2 [{sense}], '
            f'kernel {np.amin(self.kernel_sigma):.2f}-{np.amax(self.kernel_sigma):.2f} km/s, '
            f'{self.n_unmatched} unmatched pixels>'
        )


def match_resolution(
    idsp_from, idsp_to, velscale, epsilon_sigma=VARSMOOTH_MIN_SIG, sigma_floor=0.0
):
    r"""
    Compute the kernel that brings one resolution to another, and the offset.

    Both inputs are the **pre-pixelized** instrumental dispersion in km/s,
    sampled on the same logarithmic grid.

    The construction
    ----------------

    The kernel that would match the two resolutions exactly has variance

    .. math::

        {\rm res\_match}(\lambda) = \sigma_{\rm to}^2(\lambda)
                                  - \sigma_{\rm from}^2(\lambda)

    which is negative wherever the spectrum being prepared is already of
    *lower* resolution than the target, and therefore unusable as it stands.
    Rather than trimming or masking those regions, a single constant
    :math:`\delta^2` is subtracted from the whole vector:

    .. math::

        \delta^2 &= \min_\lambda {\rm res\_match} - \epsilon_\sigma^2 \\
        {\rm kernel}(\lambda) &= \sqrt{{\rm res\_match}(\lambda) - \delta^2} \\
        {\rm dvar\_inst} &= \delta^2

    Two things follow immediately, and are the reason for this form.  The
    kernel is **real everywhere**, since
    :math:`{\rm res\_match} - \delta^2 \geq \epsilon_\sigma^2 > 0` by
    construction -- no deconvolution, and no special case to detect.  And the
    smallest kernel is **exactly** :math:`\epsilon_\sigma`, so no fudge factor
    is needed to keep the extremal pixel from being masked.

    :math:`\delta^2` is **signed**, which is the substantive departure from
    ``mangadap``: its offset is :math:`\min(0, \ldots)` and so can only ever
    lower the resolution of the prepared spectrum.  Allowing a positive value
    makes the regime where the template is left at *higher* resolution than the
    galaxy reachable, which is what holds the fitted dispersion away from zero.

    Parameters
    ----------
    idsp_from : :class:`numpy.ndarray`
        Instrumental dispersion of the spectrum being prepared, in km/s.
    idsp_to : :class:`numpy.ndarray`
        Instrumental dispersion to match, in km/s, on the same grid.
    velscale : float
        Velocity scale of the grid, in km/s per pixel.
    epsilon_sigma : float, optional
        Target for the *minimum* kernel dispersion, in pixels.  Defaults to
        :data:`VARSMOOTH_MIN_SIG`, below which the convolution silently clips;
        see that constant.
    sigma_floor : float, optional
        The largest pedestal permitted, in km/s, when the offset would
        otherwise be negative.  Zero forbids a negative offset altogether.
        Where the constraint bites, the affected pixels cannot be matched and
        are flagged.

    Returns
    -------
    ResolutionMatch
        The kernel and the resulting offset.

    Raises
    ------
    DC3ResolutionError
        Raised if the inputs disagree in shape, are not positive, or if
        ``epsilon_sigma`` is below what the convolution can apply.

    Warns
    -----
    UserWarning
        Issued if the resulting offset is negative, since it imposes a floor on
        the measurable astrophysical dispersion, or if any pixel could not be
        matched.
    """
    _from = np.atleast_1d(np.asarray(idsp_from, dtype=float))
    _to = np.atleast_1d(np.asarray(idsp_to, dtype=float))
    if _from.shape != _to.shape:
        raise DC3ResolutionError(
            f'The two dispersion vectors must have the same shape; got {_from.shape} and '
            f'{_to.shape}.  Interpolate the target onto the grid of the spectrum being prepared '
            'before calling this.'
        )
    if np.any(_from <= 0) or np.any(_to <= 0):
        raise DC3ResolutionError('Instrumental dispersions must be positive.')
    if epsilon_sigma < VARSMOOTH_MIN_SIG:
        raise DC3ResolutionError(
            f'epsilon_sigma of {epsilon_sigma} pixels is below the {VARSMOOTH_MIN_SIG} pixels '
            'that the convolution applies accurately.  Below that, the realised kernel is '
            'roughly 0.71 pixels whatever is asked for, so the code would believe it had '
            'applied a far narrower kernel than it did, making dvar_inst wrong by the '
            'difference and biasing the astrophysical dispersion low.'
        )

    # The exactly-matching kernel, squared.  Negative where the spectrum being
    # prepared is already of lower resolution than the target.
    res_match = np.square(_to) - np.square(_from)
    epsilon_kms2 = np.square(epsilon_sigma * velscale)

    # The signed constant offset.  Subtracting it makes the kernel real
    # everywhere and its minimum exactly epsilon_sigma.
    dvar_inst = np.amin(res_match) - epsilon_kms2

    # A negative offset leaves the prepared spectrum at lower resolution than
    # the target, which is permitted only up to sigma_floor.
    unmatched = np.zeros(res_match.shape, dtype=bool)
    if dvar_inst < -np.square(sigma_floor):
        dvar_inst = -np.square(sigma_floor)
        # With the offset clamped, the kernel is no longer real everywhere;
        # those pixels cannot be matched without deconvolution.
        unmatched = res_match - dvar_inst < epsilon_kms2

    kernel_sigma = np.sqrt(np.clip(res_match - dvar_inst, epsilon_kms2, None))

    if dvar_inst < 0:
        warnings.warn(
            'The prepared spectrum is left at lower resolution than the target, by '
            f'{np.sqrt(-dvar_inst):.2f} km/s (dvar_inst = {dvar_inst:.3f} (km/s)^2).  This '
            'imposes a floor of that size on the measurable astrophysical dispersion: no '
            'smaller dispersion can be recovered, because the template is already that broad.'
        )
    if np.any(unmatched):
        warnings.warn(
            f'{np.sum(unmatched)} of {unmatched.size} pixels cannot be matched to the target '
            f'resolution within sigma_floor = {sigma_floor} km/s.  Their kernel is held at '
            'epsilon_sigma, so the resolution there is not what the bookkeeping assumes.  '
            'Raise sigma_floor to accept a larger pedestal, or mask these regions.'
        )

    return ResolutionMatch(kernel_sigma, dvar_inst, unmatched, velscale, epsilon_sigma)


def apply_kernel(loglam, flux, match, oversample=1):
    r"""
    Convolve a spectrum with a variable-dispersion Gaussian kernel.

    Wraps :func:`ppxf.ppxf_util.varsmooth`, keeping the ``ppxf`` import in one
    place.  The coordinate passed is :math:`\log_{10}\lambda`, for a reason
    worth stating: the algorithm converts the kernel dispersion to pixels using
    a *centred finite-difference* gradient of the coordinate rather than a
    declared pixel size, which is exact only where the sampling is uniform.  On
    a logarithmic grid it is.

    Parameters
    ----------
    loglam : :class:`numpy.ndarray`
        :math:`\log_{10}` of the wavelength at each pixel.
    flux : :class:`numpy.ndarray`
        Spectrum to convolve, of shape ``(npix,)`` or ``(nspec, npix)``.
    match : ResolutionMatch
        The kernel to apply, from :func:`match_resolution`.
    oversample : int, optional
        Oversampling of the *internal* stretched grid used by the convolution,
        which reduces its interpolation error.  This is a different knob from
        ``velscale_ratio``, which oversamples the *output* grid; the two
        address different error terms.

    Returns
    -------
    :class:`numpy.ndarray`
        The convolved spectrum, with the same shape as ``flux``.

    Raises
    ------
    DC3CodingError
        Raised if ``match`` records that no matching was performed.  There is
        no kernel to apply, and the caller should have skipped this step.
    """
    if not match.performed:
        raise DC3CodingError(
            'apply_kernel was called with a ResolutionMatch that was not performed; there is '
            'no kernel to apply.  Check ResolutionMatch.performed before convolving.'
        )
    # varsmooth needs the kernel in the units of the coordinate it is given.
    # On a log10 grid, d(log10 lambda) = dv / (c ln 10).
    sigma_loglam = match.kernel_sigma / (SPEED_OF_LIGHT * np.log(10.0))

    _flux = np.asarray(flux, dtype=float)
    if _flux.ndim == 1:
        return ppxf_util.varsmooth(loglam, _flux, sigma_loglam, oversample=oversample)
    return np.array([
        ppxf_util.varsmooth(loglam, f, sigma_loglam, oversample=oversample) for f in _flux
    ])
