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
  linear/logarithmic branch in each conversion.  Here km/s to pixels is a
  division by each pixel's width in velocity, taken from the pixel borders by
  :attr:`~dc3.core.sampling.SpectralGrid.pixel_velocity` whatever the kind of
  grid, so the wavelength-coordinate system is never needed.
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
    'check_pixelization',
    'dispersion_from_resolving_power',
    'idsp_breaks',
    'match_resolution',
    'minimum_velscale_ratio',
    'resolving_power_from_dispersion',
    'varsmooth_excess',
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
    0.5-pixel kernel on ``x = numpy.arange(n)`` yields 0.87 pixels.  Breaking
    the uniformity restores the correct 0.500, but only if the perturbation
    survives the rounding of the sum that sets the grid length: one part in
    :math:`10^{12}` suffices for a few thousand pixels and is lost by twenty
    thousand.  See ``test_resolution.py``, which pins both regimes.

    ``dc3`` is insulated on both counts, by design.  The sub-clip path is
    closed by :func:`match_resolution` refusing an ``epsilon_sigma`` below the
    clip.  The uniform-kernel path is closed by :func:`apply_kernel`, which
    always breaks the uniformity of the kernel it passes, by an amount scaled
    to the stretched grid so that it survives that rounding without
    overshooting; see ``_break_kernel_uniformity``.  The defect has been reported upstream,
    and the workaround is to be removed once a fixed ``ppxf`` is released.
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


def idsp_breaks(idsp, tol=0.01):
    r"""
    Find the jumps in an instrumental dispersion vector.

    A library spliced from sections observed at different resolutions has a
    dispersion that jumps where two meet, and the variable-dispersion
    convolution of Step 1 is not correct within a kernel's width of such a
    jump (see :func:`~dc3.templates.prepare`).  A boundary between pixels
    :math:`i` and :math:`i+1` is flagged if the dispersion changes across it by
    more than the tolerance, as a fraction of the smaller of the two,

    .. math::

        \frac{|\sigma_{i+1} - \sigma_i|}{\min(\sigma_i, \sigma_{i+1})} > {\rm tol}.

    A smooth resolution vector changes by about :math:`10^{-4}` per pixel, and
    a measured, noisy one by more; the default of one per cent sits above that
    and well below the jumps that matter.  A change spread over several pixels,
    each below the tolerance, is treated as smooth however large its total.

    Every flagged boundary is returned, so a jump spread over two or three
    pixels -- by interpolation of the vector onto the library's grid, for
    example -- appears as a run of adjacent breaks.

    Parameters
    ----------
    idsp : :class:`numpy.ndarray`
        The instrumental dispersion, in km/s, one value per pixel.
    tol : float, optional
        The largest fractional change between neighbours that is treated as
        smooth.

    Returns
    -------
    :class:`numpy.ndarray`
        The index of the first pixel after each flagged boundary, in ascending
        order; empty if there are none.

    Raises
    ------
    DC3ResolutionError
        Raised if the dispersion is not a one-dimensional vector of positive
        values.
    """
    _idsp = np.asarray(idsp, dtype=float)
    if _idsp.ndim != 1:
        raise DC3ResolutionError('The instrumental dispersion must be a one-dimensional vector.')
    if np.any(_idsp <= 0):
        raise DC3ResolutionError('The instrumental dispersion must be positive.')
    change = np.absolute(np.diff(_idsp)) / np.minimum(_idsp[1:], _idsp[:-1])
    return np.where(change > tol)[0] + 1


def varsmooth_excess(k, oversample, dynamic_range):
    r"""
    Return the variance :func:`ppxf.ppxf_util.varsmooth` adds beyond its kernel.

    ``varsmooth`` stretches the coordinate so that a variable kernel becomes
    constant, interpolating the spectrum linearly onto the stretched grid, and
    back after convolving.  For a kernel that varies, the second moment of a
    convolved line exceeds the quadrature sum of its own and the kernel's by

    .. math::

        E_1 = \frac{1}{6} - \frac{1}{\pi^2}\sum_{p \geq 1}
              \frac{e^{-2\pi^2p^2k^2}}{p^2} + \frac{1}{6(mD)^2}

    pixels squared, with :math:`k` the local kernel dispersion in pixels,
    :math:`m` the oversampling and :math:`D = k_{\rm max}/k` the widest kernel
    in the spectrum relative to the local one.

    - The first term is the variance of linear interpolation, a triangle one
      pixel wide on either side.
    - The sum is the part of it the kernel does not resolve; it takes the
      excess to zero as :math:`k \to 0`, where interpolating back at the
      original nodes returns the original samples.
    - The last term is the return interpolation, from a stretched grid whose
      spacing is :math:`1/(mD)` pixels.

    It is exact for the second moment of a resolved line, and does not hold
    for an exactly uniform kernel, whose stretched grid lines up with the
    pixels; :func:`apply_kernel` never passes one.  It is derived and
    characterized in the developer documentation on template preparation
    (Characterization 2).

    Parameters
    ----------
    k : float, :class:`numpy.ndarray`
        The local kernel dispersion, in pixels, as ``varsmooth`` applies it:
        at least :data:`VARSMOOTH_MIN_SIG`.
    oversample : int
        The oversampling passed to ``varsmooth``, :math:`m`.
    dynamic_range : float, :class:`numpy.ndarray`
        The local dynamic range, :math:`D = k_{\rm max}/k`.

    Returns
    -------
    :class:`numpy.ndarray`
        The excess variance, in pixels squared, with the broadcast shape of
        ``k`` and ``dynamic_range``.
    """
    _k = np.asarray(k, dtype=float)
    # The sum converges fast: by p = 100 its terms are below 1e-4 of the first
    p = np.arange(1, 101, dtype=float).reshape((-1,) + (1,) * _k.ndim)
    aliased = np.sum(np.exp(-2 * np.square(np.pi * p * _k)) / np.square(p), axis=0)
    return (
        1 / 6 - aliased / np.pi ** 2
        + 1 / (6 * np.square(oversample * np.asarray(dynamic_range, dtype=float)))
    )


def minimum_velscale_ratio(idsp, velscale):
    r"""
    Return the smallest oversampling that keeps a line-spread function Nyquist-sampled.

    **The criterion** is that the FWHM of the line-spread function spans at
    least two pixels.  For a Gaussian that is a dispersion of

    .. math::

        \sigma_{\rm min} = \frac{2}{\sqrt{8\ln 2}} \approx 0.849\ {\rm pixels}.

    A prepared template of instrumental dispersion :math:`\sigma(\lambda)`,
    resampled onto a grid of :math:`\Delta v/r` km/s per pixel, meets it
    wherever :math:`\sigma r/\Delta v \geq \sigma_{\rm min}`.  This returns the
    smallest integer :math:`r \geq 1` for which that holds at *every*
    wavelength, since the criterion has to be met where the line-spread
    function is narrowest.

    This is numerically the same threshold as the sub-Nyquist convolution
    floor, ``ConvolvePar.min_sigma = 0.85``, below which the broadening kernel
    is block-replicated.  That is not a coincidence: it is one criterion applied
    to two different objects, the prepared template here and the fitted
    broadening function there.

    Parameters
    ----------
    idsp : :class:`numpy.ndarray`
        The instrumental dispersion of the prepared template, in km/s.
    velscale : float
        The velocity scale of the galaxy spectra, in km/s per pixel.

    Returns
    -------
    int
        The smallest ``velscale_ratio`` meeting the criterion.

    Raises
    ------
    DC3ResolutionError
        Raised if the dispersion is not positive everywhere.
    """
    narrowest = np.amin(np.asarray(idsp, dtype=float))
    if narrowest <= 0:
        raise DC3ResolutionError('The instrumental dispersion must be positive.')
    sigma_min = 2.0 / SIGMA_TO_FWHM
    return max(1, int(np.ceil(sigma_min * velscale / narrowest)))


def check_pixelization(idsp, velscale, label='spectra'):
    r"""
    Warn if an instrumental dispersion is smaller than pixelization alone would give.

    A soft diagnostic, not a check: it warns and never fails.  The dispersion
    vectors ``dc3`` is given are assumed to be **pre-pixelized**, describing the
    line-spread function before integration over a pixel, and nothing in the
    vector itself can confirm that.  What *can* be tested is a lower bound.

    **The metric.**  Integrating over a pixel of width :math:`\Delta` convolves
    the spectrum with a top-hat of that width, whose variance is

    .. math::

        \int_{-\Delta/2}^{+\Delta/2} \frac{x^2}{\Delta}\,dx = \frac{\Delta^2}{12},

    so any dispersion measured *after* pixelization is at least
    :math:`\Delta/\sqrt{12} \approx 0.289` pixels.  A supplied value below that
    is therefore **not** a post-pixelized dispersion.  Nor is it a plausible
    pre-pixelized one for a spectrum sampled well enough to measure kinematics
    from.  It most likely means the vector is simply wrong, and the commonest
    way to get there is a unit error: a dispersion in angstroms supplied where
    km/s is expected is smaller by a factor of order :math:`c/\lambda \sim 60`.

    .. note::

        This cannot detect the error the pre-pixelized contract warns about --
        a post-pixelized vector (MaNGA's ``DISP`` rather than ``PREDISP``)
        supplied in its place.  That error makes the vector *larger*, by
        :math:`\Delta^2/12` in quadrature, and no threshold on the vector alone
        distinguishes it from a genuinely broader line-spread function.

    Parameters
    ----------
    idsp : :class:`numpy.ndarray`
        The instrumental dispersion, in km/s, of shape ``(npix,)`` or
        ``(nspec, npix)``.
    velscale : float, :class:`numpy.ndarray`
        The width of each pixel of the spectra it describes, in km/s: a scalar
        for a logarithmic grid, or one value per pixel for any other (see
        :attr:`~dc3.core.sampling.SpectralGrid.pixel_velocity`).
    label : str, optional
        What the dispersion belongs to, for the warning message.

    Returns
    -------
    bool
        True if the dispersion is at least :math:`\Delta/\sqrt{12}` everywhere,
        False if a warning was issued.
    """
    floor = np.asarray(velscale, dtype=float) / np.sqrt(12.0)
    below = np.asarray(idsp, dtype=float) < floor
    if not np.any(below):
        return True
    if floor.ndim == 0:
        bound = (
            f'{floor:.2f} km/s, the dispersion that pixel integration alone produces at '
            f'{float(velscale):.2f} km/s per pixel,'
        )
    else:
        bound = (
            f'{np.amin(floor):.2f}-{np.amax(floor):.2f} km/s, the dispersion that pixel '
            'integration alone produces at the width of each pixel,'
        )
    warnings.warn(
        f'The instrumental dispersion of the {label} falls below {bound} in {np.sum(below)} '
        f'of {below.size} pixels.  A dispersion that small is not plausible for spectra '
        'sampled well enough to measure kinematics from, and most likely indicates an error '
        'in the vector -- check in particular that it is in km/s and not angstroms.'
    )
    return False


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
        performed; see :meth:`identity`.
    dvar_inst : float
        The signed instrumental variance left after matching, in
        :math:`({\rm km/s})^2`.  See the module documentation.
    unmatched : :class:`numpy.ndarray`
        Boolean, True where the target resolution could not be reached without
        deconvolution.  All False unless ``sigma_floor`` constrained the offset.
    velscale : float, :class:`numpy.ndarray`
        The width of each pixel of the grid, in km/s: a scalar for a
        logarithmic grid, or one value per pixel.
    epsilon_sigma : float
        The requested minimum kernel dispersion, in pixels.
    achieved : :class:`numpy.ndarray`, optional
        The instrumental dispersion the matched spectrum carries at each pixel,
        in km/s, as the model of the matching predicts it.  Wherever the
        matching succeeds this is the target less ``dvar_inst``; where it
        cannot, it is what the floor kernel actually leaves.  None if no
        matching was performed.

    Attributes
    ----------
    kernel_sigma : :class:`numpy.ndarray`, None
        As above.
    dvar_inst : float
        As above.
    unmatched : :class:`numpy.ndarray`
        As above.
    velscale : float, :class:`numpy.ndarray`
        As above: a float for a scalar, otherwise a float array.
    epsilon_sigma : float
        As above.
    achieved : :class:`numpy.ndarray`, None
        As above.
    """

    def __init__(
        self, kernel_sigma, dvar_inst, unmatched, velscale, epsilon_sigma, achieved=None
    ):
        self.kernel_sigma = kernel_sigma
        self.dvar_inst = float(dvar_inst)
        self.unmatched = unmatched
        _velscale = np.asarray(velscale, dtype=float)
        self.velscale = float(_velscale) if _velscale.ndim == 0 else _velscale
        self.epsilon_sigma = float(epsilon_sigma)
        self.achieved = achieved

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
        velscale : float, :class:`numpy.ndarray`
            The width of each pixel of their grid, in km/s; a scalar for a
            logarithmic grid.
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

        Each pixel's kernel is divided by that pixel's own width in velocity.
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
    idsp_from, idsp_to, velscale, epsilon_sigma=VARSMOOTH_MIN_SIG, sigma_floor=0.0,
    varsmooth_oversample=None
):
    r"""
    Compute the kernel that brings one resolution to another, and the offset.

    Both inputs are the **pre-pixelized** instrumental dispersion in km/s,
    sampled on the same grid, which may be of any kind.  The resolution is
    matched in the second moment of the line-spread function.

    **The construction.**  The kernel that would match the two resolutions
    exactly has variance

    .. math::

        {\rm res\_match}(\lambda) = \sigma_{\rm to}^2(\lambda)
                                  - \sigma_{\rm from}^2(\lambda)

    which is negative wherever the spectrum being prepared is already of
    *lower* resolution than the target, and therefore unusable as it stands.
    Rather than trimming or masking those regions, a single constant
    :math:`\delta^2` is subtracted from the whole vector.  With
    :math:`v_{\rm pix}(\lambda)` the width of each pixel in km/s, so that
    :math:`\epsilon_\sigma v_{\rm pix}` is the target minimum kernel in km/s,

    .. math::

        \delta^2 &= \min_\lambda \left[{\rm res\_match}(\lambda)
                    - (\epsilon_\sigma v_{\rm pix}(\lambda))^2\right] \\
        {\rm kernel}(\lambda) &= \sqrt{{\rm res\_match}(\lambda) - \delta^2} \\
        {\rm dvar\_inst} &= \delta^2

    Two things follow immediately, and are the reason for this form.  The
    kernel is **real everywhere**, since
    :math:`{\rm res\_match} - \delta^2 \geq (\epsilon_\sigma v_{\rm pix})^2 > 0`
    by construction -- no deconvolution, and no special case to detect.  And
    the kernel is at least :math:`\epsilon_\sigma` **pixels** wide at every
    pixel, and **exactly** that at one, so no fudge factor is needed to keep the
    extremal pixel from being masked.  On a logarithmic grid
    :math:`v_{\rm pix}` is a constant and the minimum is simply that of
    :math:`{\rm res\_match}`; on any other grid the extremal pixel is the one
    where the matching kernel is narrowest *in pixels*, which need not be where
    it is narrowest in km/s.

    :math:`\delta^2` is **signed**, which is the substantive departure from
    ``mangadap``: its offset is :math:`\min(0, \ldots)` and so can only ever
    lower the resolution of the prepared spectrum.  Allowing a positive value
    makes the regime where the template is left at *higher* resolution than the
    galaxy reachable, which is what holds the fitted dispersion away from zero.

    **Achieving the target, not only requesting it.**  The convolution adds
    :math:`E_1(k)` pixels squared beyond its kernel (see
    :func:`varsmooth_excess`).  If ``varsmooth_oversample`` is given, the
    kernel is chosen so that the variance the convolution *applies*,
    :math:`g(k) = k^2 + E_1(k)` pixels squared, rather than :math:`k^2`, makes
    up the difference.  :math:`g` increases monotonically, so it is inverted by
    interpolating a table of it; it depends on the widest kernel through
    :math:`D`, so the kernel is solved for iteratively.  The smallest variance
    the matching can add then becomes :math:`g(\epsilon_\sigma)` pixels squared
    rather than :math:`\epsilon_\sigma^2`, which enters the offset and, where
    ``sigma_floor`` constrains it, the unmatched pixels.

    Whether or not it is corrected for, :attr:`ResolutionMatch.achieved` gives
    the resolution the convolved spectrum carries under the model used: the
    target less ``dvar_inst`` wherever the matching succeeds, and what the
    floor kernel leaves where it cannot.  Anything done to the spectrum after
    the convolution is the caller's to account for, in ``idsp_to``.

    Parameters
    ----------
    idsp_from : :class:`numpy.ndarray`
        Instrumental dispersion of the spectrum being prepared, in km/s.
    idsp_to : :class:`numpy.ndarray`
        Instrumental dispersion to match, in km/s, on the same grid.
    velscale : float, :class:`numpy.ndarray`
        The width of each pixel of the grid, in km/s: a scalar for a
        logarithmic grid, or one value per pixel for any other (see
        :attr:`~dc3.core.sampling.SpectralGrid.pixel_velocity`).
    epsilon_sigma : float, optional
        Target for the *minimum* kernel dispersion, in pixels.  Defaults to
        :data:`VARSMOOTH_MIN_SIG`, below which the convolution silently clips;
        see that constant.
    sigma_floor : float, optional
        The largest pedestal permitted, in km/s, when the offset would
        otherwise be negative.  Zero forbids a negative offset altogether.
        Where the constraint bites, the affected pixels cannot be matched and
        are flagged.
    varsmooth_oversample : int, optional
        The oversampling the kernel will be applied with, by
        :func:`apply_kernel`.  If given, the kernel accounts for the variance
        the convolution adds beyond it.  If None, the kernel alone is assumed
        to be applied.

    Returns
    -------
    ResolutionMatch
        The kernel and the resulting offset.

    Raises
    ------
    DC3ResolutionError
        Raised if the inputs disagree in shape, are not positive, or if
        ``epsilon_sigma`` is below what the convolution can apply.  A
        ``velscale`` array must have one element per pixel.

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
    _velscale = np.asarray(velscale, dtype=float)
    if _velscale.ndim != 0 and _velscale.shape != _from.shape:
        raise DC3ResolutionError(
            f'velscale must be a scalar or have one element per pixel; got shape '
            f'{_velscale.shape} for {_from.size} pixels.'
        )
    if np.any(_velscale <= 0):
        raise DC3ResolutionError('The pixel widths in velocity must be positive.')
    if epsilon_sigma < VARSMOOTH_MIN_SIG:
        raise DC3ResolutionError(
            f'epsilon_sigma of {epsilon_sigma} pixels is below the {VARSMOOTH_MIN_SIG} pixels '
            'that the convolution applies accurately.  Below that, the realised kernel is '
            'roughly 0.71 pixels whatever is asked for, so the code would believe it had '
            'applied a far narrower kernel than it did, making dvar_inst wrong by the '
            'difference and biasing the astrophysical dispersion low.'
        )

    # The variance the matching must add, in (km/s)^2.  Negative where the
    # spectrum being prepared is already of lower resolution than the target.
    res_match = np.square(_to) - np.square(_from)
    pixel2 = np.square(_velscale)

    if varsmooth_oversample is None:
        kernel_pixels, dvar_inst, unmatched = _offset_and_kernel(
            res_match, pixel2, epsilon_sigma, sigma_floor, np.square
        )
        applied = np.square(kernel_pixels)
    else:
        # The variance the convolution applies depends on the widest kernel,
        # through D, so the kernel is solved for until that settles.  It
        # changes the result only through the small 1/(mD)^2 term, so a few
        # iterations suffice.
        kmax = np.sqrt(np.amax(np.clip(res_match / pixel2, epsilon_sigma ** 2, None)))
        for _ in range(50):
            kernel_pixels, dvar_inst, unmatched = _offset_and_kernel(
                res_match, pixel2, epsilon_sigma, sigma_floor,
                _ConvolutionVariance(varsmooth_oversample, kmax)
            )
            converged = abs(np.amax(kernel_pixels) - kmax) <= 1e-10 * kmax
            kmax = np.amax(kernel_pixels)
            if converged:
                break
        applied = _ConvolutionVariance(varsmooth_oversample, kmax)(kernel_pixels)

    kernel_sigma = kernel_pixels * _velscale
    achieved = np.sqrt(np.square(_from) + applied * pixel2)

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

    return ResolutionMatch(
        kernel_sigma, dvar_inst, unmatched, velscale, epsilon_sigma, achieved=achieved
    )


class _ConvolutionVariance:
    r"""
    The variance the convolution applies for a kernel, in pixels squared.

    :math:`g(k) = k^2 + E_1(k)`, with :math:`E_1` from :func:`varsmooth_excess`
    at a fixed oversampling and widest kernel.  Callable, and invertible by
    :meth:`inverse`, since :math:`g` increases monotonically.

    Parameters
    ----------
    oversample : int
        The oversampling passed to ``varsmooth``.
    kmax : float
        The widest kernel in the spectrum, in pixels.
    """

    def __init__(self, oversample, kmax):
        self.oversample = oversample
        self.kmax = kmax

    def __call__(self, k):
        """Return :math:`g(k)`."""
        _k = np.asarray(k, dtype=float)
        return np.square(_k) + varsmooth_excess(_k, self.oversample, self.kmax / _k)

    def inverse(self, variance, epsilon_sigma):
        r"""
        Return the kernel that applies a given variance.

        Parameters
        ----------
        variance : :class:`numpy.ndarray`
            The variance to apply, in pixels squared; at least
            :math:`g(\epsilon_\sigma)`.
        epsilon_sigma : float
            The smallest kernel, in pixels.

        Returns
        -------
        :class:`numpy.ndarray`
            The kernel, in pixels.
        """
        # Tabulated finely enough that interpolating it is exact to well below
        # anything measurable: 4096 points spaced by about 0.1 per cent
        upper = np.sqrt(np.amax(variance)) + 1.0
        k = np.geomspace(epsilon_sigma, upper, 4096)
        return np.interp(variance, self(k), k)


def _offset_and_kernel(res_match, pixel2, epsilon_sigma, sigma_floor, applied):
    r"""
    Return the kernel, the signed offset, and the unmatched pixels.

    The construction of :func:`match_resolution`, for a given relation between
    the kernel and the variance it applies.

    Parameters
    ----------
    res_match : :class:`numpy.ndarray`
        The variance the matching must add at each pixel, in
        :math:`({\rm km/s})^2`.
    pixel2 : float, :class:`numpy.ndarray`
        The square of each pixel's width, in :math:`({\rm km/s})^2`.
    epsilon_sigma : float
        The smallest kernel, in pixels.
    sigma_floor : float
        The largest pedestal permitted, in km/s.
    applied : callable
        The variance a kernel applies, in pixels squared, as a function of the
        kernel in pixels.  If it has an ``inverse`` method, that finds the
        kernel; otherwise the variance is taken to be :math:`k^2`.

    Returns
    -------
    tuple
        The kernel at each pixel, in pixels; the offset, ``dvar_inst``, in
        :math:`({\rm km/s})^2`; and the boolean unmatched pixels.
    """
    # The smallest variance the matching can add at each pixel, in (km/s)^2
    floor = applied(epsilon_sigma) * pixel2

    # The signed constant offset.  Subtracting it leaves every pixel needing at
    # least the floor, and exactly the floor at one pixel.
    dvar_inst = np.amin(res_match - floor)

    # A negative offset leaves the prepared spectrum at lower resolution than
    # the target, which is permitted only up to sigma_floor.
    unmatched = np.zeros(res_match.shape, dtype=bool)
    if dvar_inst < -np.square(sigma_floor):
        dvar_inst = -np.square(sigma_floor)
        # With the offset clamped, those pixels would need less than the floor,
        # which only a deconvolution could give; they are held at the floor.
        unmatched = res_match - dvar_inst < floor

    required = np.maximum((res_match - dvar_inst) / pixel2, applied(epsilon_sigma))
    if hasattr(applied, 'inverse'):
        kernel = applied.inverse(required, epsilon_sigma)
    else:
        kernel = np.sqrt(required)
    return kernel, float(dvar_inst), unmatched


def _break_kernel_uniformity(sig, oversample=1):
    r"""
    Perturb a kernel so that ``varsmooth`` cannot find it exactly uniform.

    A workaround for the upstream defect described under
    :data:`VARSMOOTH_MIN_SIG`.  Following Cappellari (2023, Algorithm 1),
    :func:`ppxf.ppxf_util.varsmooth` stretches the coordinate so that pixel
    :math:`i` spans :math:`m\sigma_{\rm max}/\sigma_i` samples, with :math:`m`
    the oversampling, and then places :math:`n = \lceil S \rceil` samples
    uniformly over the stretched span

    .. math::

        S = m \sum_{i=1}^{N-1} \frac{\sigma_{\rm max}}{\sigma_i}.

    For an exactly uniform kernel :math:`S = m(N-1)` is a whole number, so
    :math:`n` is one short of the :math:`S + 1` the span needs, the samples
    fall out of step with the input pixels, and the interpolation onto them and
    back broadens the result.

    **The largest element is increased**, by a factor :math:`1 + \epsilon`.
    That scales the span of every other pixel by the same factor, so :math:`S`
    grows by :math:`\epsilon S` while the stretched samples keep their place
    relative to the input pixels; for a uniform kernel, :math:`n` becomes the
    :math:`S + 1` needed.  Reducing an element instead would be undone wherever
    it sits at ``varsmooth``'s 0.1-pixel clip -- which is exactly where the
    minimum of a kernel built to ``epsilon_sigma = 0.1`` lies -- and would
    shorten the span if it reduced a unique maximum.

    **The size is a hundredth of a sample of growth**, :math:`\epsilon = 0.01 /
    S`, and is bounded on both sides.  The growth must survive the rounding of
    the sum that gives :math:`S`: one part in :math:`10^{12}`, for example, is
    lost by twenty thousand pixels.  And it must stay below one sample, or
    :math:`n` gains a point and the samples fall out of step again: a fixed
    :math:`10^{-6}` broadens a uniform kernel just as the defect does once
    :math:`S` passes a million.  It must also be scaled to :math:`S` rather
    than to :math:`m(N-1)`, which a varying kernel's span exceeds many times
    over; growing it by several samples would move every stretched sample and
    change the result at the level of the interpolation error itself, a few per
    cent of the peak.  A hundredth of a sample adds a sample to a varying kernel
    only when the fractional part of :math:`S` exceeds 0.99, and otherwise
    changes the result by a few parts in :math:`10^7` of its peak: numerically
    irrelevant.

    It is applied whether or not the kernel is exactly uniform, since a nearly
    uniform kernel can reach a whole-number span once its variation is lost in
    the same rounding.  The kernels :func:`match_resolution` builds are never
    below the clip, so :math:`S` is computed without it.

    .. todo::

        Remove this once a ``ppxf`` release fixes the defect;
        ``test_uniform_kernel_triggers_the_upstream_off_by_one`` fails when it
        does.

    Parameters
    ----------
    sig : :class:`numpy.ndarray`
        The kernel dispersion, in pixels.
    oversample : int, optional
        The oversampling that will be passed to ``varsmooth``.

    Returns
    -------
    :class:`numpy.ndarray`
        A perturbed copy; the input is not modified.
    """
    _sig = np.array(sig, dtype=float)
    imax = np.argmax(_sig)
    # The stretched span varsmooth will compute, in samples
    span = oversample * np.sum(_sig[imax] / _sig[1:])
    _sig[imax] *= 1 + 0.01 / span
    return _sig


def apply_kernel(flux, match, oversample=2):
    r"""
    Convolve a spectrum with a variable-dispersion Gaussian kernel.

    Wraps :func:`ppxf.ppxf_util.varsmooth`, keeping the ``ppxf`` import in one
    place.

    **The coordinate passed is the pixel index**, ``numpy.arange(npix)``, and
    the kernel is given in pixels, :attr:`ResolutionMatch.kernel_sigma_pixels`.
    ``varsmooth`` uses its coordinate only to convert the kernel to pixels, by a
    *centred finite-difference* gradient, which is exact only where the
    sampling is uniform.  Supplying the kernel already in pixels, each divided
    by its own pixel's width, makes that conversion exact on any grid: at a
    jump in the sampling a centred difference would give the pixels either side
    the average of the two sizes.  On a logarithmic grid the result is the same
    as passing :math:`\log_{10}\lambda`.

    It does **not** make the convolution exact on an irregular grid.  Inside
    ``varsmooth`` the convolution runs in pixel space whatever the coordinate,
    so a kernel whose footprint spans pixels of different sizes treats them as
    equal.  That is second order where the pixel size changes smoothly, and is
    what the guard bands at a jump in the sampling are for.

    The kernel is perturbed by a numerically irrelevant amount before it is
    passed, to avoid an upstream defect; see :func:`_break_kernel_uniformity`.

    **The oversampling must be at least 2.**  ``varsmooth`` interpolates the
    spectrum linearly onto its stretched grid and back, which broadens every
    line by about :math:`\Delta^2/6` in variance, :math:`\Delta` the pixel,
    plus :math:`(1/mD)^2/6` from the return trip, :math:`m` being the
    oversampling and :math:`D` the widest kernel in the spectrum relative to
    the local one.  For a *uniform* kernel the stretched grid lines up with the
    pixels and the excess is instead :math:`(1 - 1/m^2)/6`, which is **zero**
    at :math:`m = 1`: a uniform kernel would be applied exactly, and one that
    varies even slightly would add up to :math:`\Delta^2/3`.  Requiring
    :math:`m \geq 2` keeps the uniform case within 0.04 :math:`\Delta^2` of the
    rest, at about twice the cost of :math:`m = 1`.  See the developer
    documentation on template preparation, and
    ``doc/scripts/explore_convolution_methods.py``.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        Spectrum to convolve, of shape ``(npix,)`` or ``(nspec, npix)``, on the
        grid ``match`` was computed for.
    match : ResolutionMatch
        The kernel to apply, from :func:`match_resolution`.
    oversample : int, optional
        Oversampling of the *internal* stretched grid used by the convolution,
        which reduces its interpolation error; at least 2.  This is a different
        knob from ``velscale_ratio``, which oversamples the *output* grid; the
        two address different error terms.

    Returns
    -------
    :class:`numpy.ndarray`
        The convolved spectrum, with the same shape as ``flux``.

    Raises
    ------
    DC3CodingError
        Raised if ``match`` records that no matching was performed, in which
        case there is no kernel to apply and the caller should have skipped
        this step; or if ``flux`` does not have one pixel per kernel element.
    DC3ResolutionError
        Raised if ``oversample`` is less than 2.
    """
    if not match.performed:
        raise DC3CodingError(
            'apply_kernel was called with a ResolutionMatch that was not performed; there is '
            'no kernel to apply.  Check ResolutionMatch.performed before convolving.'
        )
    if oversample < 2:
        raise DC3ResolutionError(
            f'oversample must be at least 2; got {oversample}.  At 1, varsmooth applies a '
            'uniform kernel exactly but adds up to a third of a pixel squared of variance to '
            'one that varies, so the result would depend on whether the kernel happened to '
            'be uniform.'
        )
    _flux = np.asarray(flux, dtype=float)
    sig = _break_kernel_uniformity(match.kernel_sigma_pixels, oversample=oversample)
    if _flux.shape[-1] != sig.size:
        raise DC3CodingError(
            f'The spectrum has {_flux.shape[-1]} pixels, but the kernel was computed for '
            f'{sig.size}.'
        )
    # The pixel index, whose centred gradient is exactly one, so varsmooth
    # takes sig as the kernel in pixels unchanged
    x = np.arange(sig.size, dtype=float)
    if _flux.ndim == 1:
        return ppxf_util.varsmooth(x, _flux, sig, oversample=oversample)
    return np.array([ppxf_util.varsmooth(x, f, sig, oversample=oversample) for f in _flux])
