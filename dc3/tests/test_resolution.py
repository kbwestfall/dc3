"""
Tests for :mod:`~dc3.core.resolution`.

The central test, :func:`test_match_spectral_resolution`, is ported from
``mangadap/tests/test_resolution_matching.py``: build a spectrum of emission
lines of known width, match it to a different resolution, and compare against
the spectrum computed *analytically* at the resolution that should result.
Checking against a closed-form expectation rather than against the code's own
intermediates is what makes it worth having.
"""

import warnings

import numpy as np
from ppxf import ppxf_util
import pytest

from dc3.core import resolution, sampling
from dc3.core.velocity import SPEED_OF_LIGHT
from dc3.pkg.exceptions import DC3ResolutionError


LOG10LAM0 = np.log10(3800.0)
# Chosen to give a velocity scale near 7.5 km/s per pixel, which is what the
# published DiskMass data has: W11 quotes a block-replication floor of
# 0.85 dv = 6.4 km/s.  A coarser grid makes the dispersions being tested
# sub-pixel, which is not a regime this method operates in.
DLOGLAM = 1.09e-5
NPIX = 200
VELSCALE = sampling.velscale(DLOGLAM)


def loglam(npix=NPIX):
    """The logarithmic coordinate used by the convolution tests."""
    return LOG10LAM0 + DLOGLAM * np.arange(npix, dtype=float)


def emission_comb(wave, sigma_kms):
    """
    Build a comb of unit-area Gaussian emission lines.

    Each line takes the dispersion *at its own wavelength*, which matters
    whenever the dispersion varies: using one value for the whole comb would
    give a spectrum that does not actually have the resolution it is supposed
    to represent.

    Parameters
    ----------
    wave : :class:`numpy.ndarray`
        Wavelengths.
    sigma_kms : :class:`numpy.ndarray`
        Line dispersion in km/s, per pixel.

    Returns
    -------
    :class:`numpy.ndarray`
        The spectrum.
    """
    _sigma = np.broadcast_to(np.asarray(sigma_kms, dtype=float), wave.shape)
    flux = np.zeros(wave.shape, dtype=float)
    for center in np.linspace(wave[0], wave[-1], num=20)[1:-1]:
        # A dispersion in km/s is a wavelength-dependent dispersion in Angstroms
        sigma_at_line = center * np.interp(center, wave, _sigma) / SPEED_OF_LIGHT
        flux += (
            np.exp(-0.5 * np.square((wave - center) / sigma_at_line))
            / np.sqrt(2.0 * np.pi) / sigma_at_line
        )
    return flux


def realised_kernel(sigma_pixels, npix=800, sigma_in=4.0):
    """
    Measure the kernel width the convolution actually applies.

    A comfortably resolved Gaussian is convolved and its second moment
    differenced in quadrature against the input.  A resolved line rather than a
    delta function, so that the measurement is not dominated by interpolation
    ringing.

    Parameters
    ----------
    sigma_pixels : float
        The kernel dispersion requested, in pixels.
    npix : int, optional
        Length of the test spectrum.
    sigma_in : float, optional
        Width of the input line, in pixels.

    Returns
    -------
    float
        The realised kernel width, in pixels.
    """
    x = loglam(npix)
    index = np.arange(npix, dtype=float)
    y = np.exp(-0.5 * np.square((index - npix / 2) / sigma_in))
    out = ppxf_util.varsmooth(x, y, np.full(npix, sigma_pixels * DLOGLAM))
    mean = np.sum(out * index) / np.sum(out)
    width = np.sqrt(np.sum(out * np.square(index - mean)) / np.sum(out))
    return np.sqrt(max(width ** 2 - sigma_in ** 2, 0.0))


# ----------------------------------------------------------------------
# The ported end-to-end test
# ----------------------------------------------------------------------
def test_match_spectral_resolution():
    r"""
    Matching reproduces the analytically expected spectrum.

    Ported from ``mangadap/tests/test_resolution_matching.py``.  A spectrum of
    emission lines at one resolution is matched to another, and compared against
    the same lines computed directly at the resolution that should result.

    **One adaptation is forced by the design.**  ``mangadap`` convolves away the
    whole resolution difference, so its expectation is the spectrum at the
    target resolution.  Here only the *wavelength-dependent* part is convolved;
    the constant part is carried as ``dvar_inst`` rather than applied.  The
    prepared spectrum therefore sits at

    .. math::

        \sigma_{\rm prepared}^2 = \sigma_{\rm to}^2 - {\rm dvar\_inst}

    at every wavelength, and that -- not the target itself -- is what the
    expectation must use.  This identity is the one the entire scheme rests on,
    so testing it against a closed form is the point of the exercise.
    """
    npix = 4000
    wave = np.power(10.0, loglam(npix))

    # Input: constant FWHM in Angstroms, so the dispersion in km/s falls with
    # wavelength.  This is what makes the required kernel wavelength-dependent.
    idsp_from = resolution.dispersion_from_resolving_power(wave / 2.5)
    # Target: constant resolving power.
    idsp_to = resolution.dispersion_from_resolving_power(np.full(npix, 900.0))

    flux = emission_comb(wave, idsp_from)

    match = resolution.match_resolution(idsp_from, idsp_to, VELSCALE)
    out = resolution.apply_kernel(loglam(npix), flux, match)

    # What the prepared spectrum should be: the lines at the resolution that
    # actually results, which differs from the target by dvar_inst.
    sigma_prepared = np.sqrt(np.square(idsp_to) - match.dvar_inst)
    expected = emission_comb(wave, sigma_prepared)

    # Compare away from the edges, where the convolution has no data to draw on
    interior = slice(50, -50)
    scale = np.amax(expected[interior])
    residual = np.amax(np.absolute(out[interior] - expected[interior])) / scale
    assert residual < 0.02, (
        f'The matched spectrum differs from the analytic expectation by {residual:.1%} of the '
        'peak; the resolution matching does not produce the resolution it claims'
    )


def test_prepared_resolution_equals_target_less_dvar_inst():
    r"""
    The identity :math:`\sigma_{\rm prep}^2 = \sigma_{\rm to}^2 - {\rm dvar\_inst}` holds.

    Checked directly on the kernel rather than through a convolution, so a
    failure points at the construction rather than at ``varsmooth``.  Everything
    downstream -- the correction from the fitted dispersion to the astrophysical
    one -- depends on this being exact.
    """
    idsp_from = np.linspace(20.0, 35.0, NPIX)
    idsp_to = np.full(NPIX, 50.0)
    match = resolution.match_resolution(idsp_from, idsp_to, VELSCALE)

    prepared_variance = np.square(idsp_from) + np.square(match.kernel_sigma)
    assert np.allclose(prepared_variance, np.square(idsp_to) - match.dvar_inst), \
        'The prepared resolution does not differ from the target by exactly dvar_inst'


# ----------------------------------------------------------------------
# Resolving power and dispersion
# ----------------------------------------------------------------------
def test_resolving_power_round_trip():
    """The two conversions invert."""
    r = np.array([1000.0, 2000.0, 5000.0])
    sigma = resolution.dispersion_from_resolving_power(r)
    assert np.allclose(resolution.resolving_power_from_dispersion(sigma), r), \
        'Converting resolving power to dispersion and back did not recover the input'


def test_dispersion_from_resolving_power_value():
    """
    The conversion uses the Gaussian FWHM-to-sigma ratio.

    Checked against an independently written expression, so a wrong constant
    would be caught.
    """
    r = 2000.0
    expected = 299792.458 / (np.sqrt(8 * np.log(2)) * r)
    assert np.isclose(resolution.dispersion_from_resolving_power(r), expected), \
        'The resolving-power conversion does not use c / (sqrt(8 ln 2) R)'


@pytest.mark.parametrize('bad', [0.0, -1.0])
def test_nonpositive_resolving_power_is_rejected(bad):
    """A resolving power and a dispersion must both be positive."""
    with pytest.raises(DC3ResolutionError, match='must be positive'):
        resolution.dispersion_from_resolving_power(bad)
    with pytest.raises(DC3ResolutionError, match='must be positive'):
        resolution.resolving_power_from_dispersion(bad)


# ----------------------------------------------------------------------
# The construction
# ----------------------------------------------------------------------
def test_minimum_kernel_is_exactly_epsilon_sigma():
    """
    The smallest kernel equals epsilon_sigma exactly, by construction.

    This is what removes the need for mangadap's 1.01 fudge factor: the
    extremal pixel sits exactly at the target rather than just inside or just
    outside it.
    """
    match = resolution.match_resolution(
        np.linspace(20.0, 30.0, NPIX), np.full(NPIX, 40.0), VELSCALE, epsilon_sigma=0.5
    )
    assert np.isclose(np.amin(match.kernel_sigma_pixels), 0.5), \
        'The minimum kernel is not exactly epsilon_sigma, so the construction is wrong'


def test_kernel_is_real_everywhere_even_at_lower_resolution():
    """
    The kernel is real at every pixel, whatever the inputs.

    This is the no-deconvolution rule enforced by construction rather than
    checked afterwards: the constant offset is chosen precisely so the kernel
    variance cannot go negative.
    """
    # Lower resolution than the target over part of the range, which would need
    # deconvolution if matched exactly.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        match = resolution.match_resolution(
            np.linspace(20.0, 60.0, NPIX), np.full(NPIX, 40.0), VELSCALE, sigma_floor=100.0
        )
    assert np.all(np.isfinite(match.kernel_sigma)), 'The kernel is not finite everywhere'
    assert np.all(match.kernel_sigma > 0), 'The kernel is not positive everywhere'


def test_positive_offset_when_template_has_higher_resolution():
    """
    A template of higher resolution than the galaxy gives a positive offset.

    This regime is unreachable in mangadap, whose offset is min(0, ...).  It is
    the one that holds the fitted dispersion away from zero.
    """
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    assert match.dvar_inst > 0, \
        'A template of higher resolution should leave a positive dvar_inst'
    assert match.sigma_floor == 0.0, 'A positive offset should impose no floor'


def test_negative_offset_when_template_has_lower_resolution():
    """
    A template of lower resolution gives a negative offset, and warns.

    The offset then imposes a floor on the measurable astrophysical dispersion,
    which the user must be told about rather than discovering as a pile-up at
    zero.
    """
    with pytest.warns(UserWarning, match='lower resolution than the target'):
        match = resolution.match_resolution(
            np.full(NPIX, 60.0), np.full(NPIX, 40.0), VELSCALE, sigma_floor=100.0
        )
    assert match.dvar_inst < 0, 'A template of lower resolution should give a negative dvar_inst'
    assert np.isclose(match.sigma_floor, np.sqrt(-match.dvar_inst)), \
        'The reported floor does not follow from dvar_inst'


def test_offset_is_the_minimum_of_the_matching_variance():
    """The offset is the minimum matching variance, less epsilon squared."""
    idsp_from = np.linspace(10.0, 20.0, NPIX)
    idsp_to = np.full(NPIX, 40.0)
    epsilon = 0.3
    match = resolution.match_resolution(idsp_from, idsp_to, VELSCALE, epsilon_sigma=epsilon)
    expected = np.amin(np.square(idsp_to) - np.square(idsp_from)) - (epsilon * VELSCALE) ** 2
    assert np.isclose(match.dvar_inst, expected), \
        'dvar_inst does not equal min(res_match) - epsilon^2'


def test_sigma_floor_clamps_the_offset_and_flags_the_remainder():
    """
    sigma_floor caps how far the offset may go negative.

    Where the cap bites, those pixels cannot be matched; they are flagged rather
    than silently left at the wrong resolution.
    """
    with pytest.warns(UserWarning):
        match = resolution.match_resolution(
            np.full(NPIX, 60.0), np.full(NPIX, 40.0), VELSCALE, sigma_floor=5.0
        )
    assert np.isclose(match.dvar_inst, -25.0), 'The offset was not clamped to -sigma_floor^2'
    assert match.n_unmatched > 0, \
        'Clamping the offset left pixels unmatched, which should be flagged'


def test_default_sigma_floor_forbids_a_negative_offset():
    """
    By default no negative offset is permitted at all.

    A negative offset imposes a floor on the science, so it is opt-in.
    """
    with pytest.warns(UserWarning, match='cannot be matched'):
        match = resolution.match_resolution(np.full(NPIX, 60.0), np.full(NPIX, 40.0), VELSCALE)
    assert match.dvar_inst == 0.0, 'The default sigma_floor should forbid a negative offset'


def test_astrophysical_variance_round_trips():
    """The correction inverts sigma_obs^2 = sigma_*^2 + dvar_inst."""
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    sigma_star = 80.0
    sigma_obs = np.sqrt(sigma_star ** 2 + match.dvar_inst)
    assert np.isclose(match.astrophysical_variance(sigma_obs), sigma_star ** 2), \
        'The astrophysical variance does not invert the instrumental offset'


def test_astrophysical_variance_may_be_negative():
    """
    A dispersion below the floor gives a negative variance, not zero.

    The original C++ clipped it, which made an unmeasurable dispersion
    indistinguishable from a genuine one; reporting the negative value keeps
    them distinct.
    """
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    assert match.astrophysical_variance(1.0) < 0, \
        'A dispersion below the instrumental offset should give a negative variance'


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------
def test_mismatched_shapes_are_rejected():
    """The two dispersion vectors must be on the same grid."""
    with pytest.raises(DC3ResolutionError, match='same shape'):
        resolution.match_resolution(np.full(10, 20.0), np.full(20, 40.0), VELSCALE)


@pytest.mark.parametrize('bad', [0.0, -5.0])
def test_nonpositive_dispersions_are_rejected(bad):
    """An instrumental dispersion must be positive."""
    with pytest.raises(DC3ResolutionError, match='must be positive'):
        resolution.match_resolution(np.full(10, bad), np.full(10, 40.0), VELSCALE)


def test_epsilon_below_the_varsmooth_clip_is_rejected():
    """
    epsilon_sigma cannot be set below what the convolution will apply.

    Allowing it would mean the code believed it applied a narrower kernel than
    it did, making dvar_inst wrong by the difference.
    """
    with pytest.raises(DC3ResolutionError, match='below the'):
        resolution.match_resolution(
            np.full(10, 20.0), np.full(10, 40.0), VELSCALE,
            epsilon_sigma=resolution.VARSMOOTH_MIN_SIG / 2
        )


# ----------------------------------------------------------------------
# The upstream clip
# ----------------------------------------------------------------------
@pytest.mark.parametrize('requested', [0.1, 0.11, 0.2, 0.5, 1.0, 2.0])
def test_varsmooth_is_accurate_at_and_above_the_clip(requested):
    """
    At and above VARSMOOTH_MIN_SIG, the kernel applied is the kernel requested.

    This is the regime ``dc3`` operates in, guaranteed by rejecting a smaller
    ``epsilon_sigma``.  If it ceased to hold, every ``dvar_inst`` would be wrong.
    """
    assert np.isclose(realised_kernel(requested), requested, rtol=0.02), (
        f'A requested kernel of {requested} px was not the kernel applied; the convolution is '
        'not accurate in the regime dc3 relies on'
    )


@pytest.mark.parametrize('requested', [0.001, 0.01, 0.05, 0.09])
def test_varsmooth_below_the_clip_applies_a_much_wider_kernel(requested):
    """
    Below the clip, the kernel applied is neither the request nor the clip.

    A behavioural assertion on an undocumented, unexported upstream constant.
    The measured behaviour is worse than the clip alone would suggest: any
    request below 0.1 px produces a realised kernel of about 0.71 px -- roughly
    seven times the clip.  See
    :func:`test_uniform_kernel_triggers_the_upstream_off_by_one` for the actual
    cause, which is not the clip.

    This is why ``epsilon_sigma`` below :data:`VARSMOOTH_MIN_SIG` is refused
    rather than quietly raised: a spectrum prepared with such a request would
    be broadened by 0.71 px while the bookkeeping recorded the requested value,
    and ``dvar_inst`` would be wrong by the whole difference.
    """
    realised = realised_kernel(requested)
    assert not np.isclose(realised, requested, atol=0.05), (
        f'A request of {requested} px appeared to be honoured; if ppxf has removed the clip, '
        'the guard in match_resolution is no longer needed'
    )
    assert np.isclose(realised, 0.714, rtol=0.05), (
        f'A request of {requested} px gave a realised kernel of {realised:.3f} px, not the '
        '0.714 px measured when this was written; the upstream behaviour has changed and the '
        'consequences for dvar_inst need re-deriving'
    )


def test_uniform_kernel_triggers_the_upstream_off_by_one():
    """
    Pin the upstream defect that makes a sub-clip kernel misbehave.

    ``varsmooth`` sizes its internal grid as ``n = ceil(xs[-1] - xs[0])`` with
    ``xs = cumsum(sig_max/sig)``.  Since ``sig_max/sig >= 1``, that span is at
    least ``N - 1``, with equality *only* when ``sig`` is exactly uniform -- and
    then ``ceil`` returns one sample fewer than the input, so the interpolation
    round-trip broadens the result.

    The clip is only the trigger, by making ``sig`` exactly uniform.  The defect
    itself is width-independent: it fires here at 0.5 px, well inside the range
    where the convolution is supposed to be accurate.  A perturbation of one
    part in 1e-12 to a single interior element restores the correct answer,
    which is what rules out any numerical explanation.
    """
    npix = 800
    index = np.arange(npix, dtype=float)
    sigma_in = 4.0
    y = np.exp(-0.5 * np.square((index - npix / 2) / sigma_in))
    # An abscissa whose numpy.gradient is exactly 1.0, so sig equals sig_x
    x = index.copy()

    def realised(out):
        mean = np.sum(out * index) / np.sum(out)
        width = np.sqrt(np.sum(out * np.square(index - mean)) / np.sum(out))
        return np.sqrt(max(width ** 2 - sigma_in ** 2, 0.0))

    uniform = np.full(npix, 0.5)
    assert not np.isclose(realised(ppxf_util.varsmooth(x, y, uniform)), 0.5, atol=0.05), (
        'A uniform 0.5 px kernel on an exactly uniform grid was applied correctly; the upstream '
        'off-by-one appears to be fixed, and the guard on epsilon_sigma can be revisited'
    )

    perturbed = uniform.copy()
    perturbed[npix // 2] *= 1 - 1e-12
    assert np.isclose(realised(ppxf_util.varsmooth(x, y, perturbed)), 0.5, rtol=0.02), (
        'Perturbing one interior element by 1e-12 did not restore the correct kernel; the cause '
        'is no longer the integer sample count, so the diagnosis needs redoing'
    )


def test_dc3_grids_avoid_the_off_by_one():
    """
    A uniform kernel on a realistic grid is applied correctly.

    dc3's own construction produces an exactly uniform kernel whenever the two
    resolutions differ by a constant, which is a common case.  It escapes the
    upstream defect only because ``numpy.gradient`` of a logarithmic wavelength
    grid carries floating-point noise, so ``sig`` is not *exactly* uniform.
    That is luck rather than design, so it is pinned here: if it ever ceases to
    hold, every ``dvar_inst`` from such a run would be wrong.
    """
    npix = 800
    match = resolution.match_resolution(
        np.full(npix, 30.0), np.full(npix, 50.0), VELSCALE
    )
    assert np.all(match.kernel_sigma == match.kernel_sigma[0]), \
        'A constant resolution difference should give an exactly uniform kernel'

    index = np.arange(npix, dtype=float)
    sigma_in = 4.0
    flux = np.exp(-0.5 * np.square((index - npix / 2) / sigma_in))
    out = resolution.apply_kernel(loglam(npix), flux, match)
    mean = np.sum(out * index) / np.sum(out)
    width = np.sqrt(np.sum(out * np.square(index - mean)) / np.sum(out))
    measured = np.sqrt(max(width ** 2 - sigma_in ** 2, 0.0))

    assert np.isclose(measured, match.kernel_sigma_pixels[0], rtol=0.05), (
        f'A uniform kernel of {match.kernel_sigma_pixels[0]:.3f} px was applied as '
        f'{measured:.3f} px; the upstream off-by-one is now firing on dc3 grids and dvar_inst '
        'is wrong wherever the two resolutions differ by a constant'
    )


# ----------------------------------------------------------------------
# Applying the kernel
# ----------------------------------------------------------------------
def test_apply_kernel_conserves_flux():
    """Convolution redistributes flux; it does not create or destroy it."""
    x = loglam()
    flux = np.zeros(NPIX)
    flux[NPIX // 2] = 1.0
    match = resolution.ResolutionMatch(
        np.full(NPIX, 5.0 * VELSCALE), 0.0, np.zeros(NPIX, dtype=bool), VELSCALE, 0.1
    )
    assert np.isclose(np.sum(resolution.apply_kernel(x, flux, match)), 1.0, rtol=1e-3), \
        'Convolution did not conserve the total flux'


def test_apply_kernel_handles_a_set_of_spectra():
    """A 2-D input is convolved spectrum by spectrum."""
    x = loglam()
    flux = np.vstack([np.full(NPIX, 1.0), np.full(NPIX, 2.0)])
    match = resolution.ResolutionMatch(
        np.full(NPIX, 5.0 * VELSCALE), 0.0, np.zeros(NPIX, dtype=bool), VELSCALE, 0.1
    )
    out = resolution.apply_kernel(x, flux, match)
    assert out.shape == flux.shape, 'The convolved set does not have the input shape'
    assert np.isclose(out[0, NPIX // 2], 1.0, rtol=1e-3), 'A constant was not preserved'
    assert np.isclose(out[1, NPIX // 2], 2.0, rtol=1e-3), \
        'The spectra were mixed rather than convolved independently'


def test_repr_reports_the_sense_of_the_offset():
    """The representation says which spectrum is at higher resolution."""
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    assert 'higher resolution' in repr(match), \
        'The representation does not report the sense of the offset'
