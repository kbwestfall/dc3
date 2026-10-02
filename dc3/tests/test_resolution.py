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

from dc3.core import lsf, resolution, sampling
from dc3.core.velocity import SPEED_OF_LIGHT
from dc3.pkg.exceptions import DC3CodingError, DC3ResolutionError


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
    out = resolution.apply_kernel(flux, match)

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


# ----------------------------------------------------------------------
# A pixel width that varies with wavelength
# ----------------------------------------------------------------------
def test_a_constant_velscale_array_is_the_scalar_case():
    """
    One width per pixel, all equal, gives exactly the logarithmic-grid result.

    A logarithmic grid is the special case of the per-pixel construction, and
    must remain bit-for-bit what it was.
    """
    idsp_from, idsp_to = np.linspace(10.0, 20.0, NPIX), np.linspace(40.0, 45.0, NPIX)
    scalar = resolution.match_resolution(idsp_from, idsp_to, VELSCALE, epsilon_sigma=0.3)
    array = resolution.match_resolution(
        idsp_from, idsp_to, np.full(NPIX, VELSCALE), epsilon_sigma=0.3
    )
    assert array.dvar_inst == scalar.dvar_inst, \
        'A constant per-pixel width changed dvar_inst from the scalar result'
    assert np.array_equal(array.kernel_sigma, scalar.kernel_sigma), \
        'A constant per-pixel width changed the kernel from the scalar result'
    assert isinstance(scalar.velscale, float), 'A scalar velscale should be kept as a float'


def test_minimum_kernel_is_epsilon_sigma_in_pixels_where_pixels_vary():
    r"""
    With pixels of varying width, the kernel is at least epsilon_sigma *pixels*.

    A constant resolution difference needs the same kernel in km/s everywhere,
    which is fewest pixels where the pixels are widest.  So the offset must be
    set there: :math:`\delta^2 = {\rm res\_match} - (\epsilon v_{\rm max})^2`,
    not :math:`(\epsilon v)^2` at wherever res_match happens to be smallest.
    """
    pixel_velocity = np.linspace(5.0, 10.0, NPIX)
    epsilon = 0.3
    match = resolution.match_resolution(
        np.full(NPIX, 20.0), np.full(NPIX, 40.0), pixel_velocity, epsilon_sigma=epsilon
    )
    pixels = match.kernel_sigma_pixels
    assert np.all(pixels >= epsilon * (1 - 1e-12)), \
        'Some pixel was given a kernel narrower than epsilon_sigma pixels'
    assert np.isclose(pixels[-1], epsilon), \
        'The kernel should be exactly epsilon_sigma pixels where the pixels are widest'
    assert np.isclose(match.dvar_inst, 40.0 ** 2 - 20.0 ** 2 - (epsilon * 10.0) ** 2), \
        'dvar_inst should be set by the widest pixel'


def test_velscale_array_must_have_one_element_per_pixel():
    """A per-pixel width must match the dispersion vectors."""
    with pytest.raises(DC3ResolutionError, match='one element per pixel'):
        resolution.match_resolution(np.full(10, 20.0), np.full(10, 40.0), np.full(9, VELSCALE))
    with pytest.raises(DC3ResolutionError, match='pixel widths'):
        resolution.match_resolution(np.full(10, 20.0), np.full(10, 40.0), np.zeros(10))


def test_check_pixelization_uses_each_pixels_own_width():
    """
    The pixel-integration bound follows the width of each pixel.

    A dispersion that is plausible for the narrow pixels but not for the wide
    ones is flagged, and only there.
    """
    pixel_velocity = np.linspace(2.0, 20.0, NPIX)
    idsp = np.full(NPIX, 10.0 / np.sqrt(12.0))
    with pytest.warns(UserWarning, match='the width of each pixel') as record:
        passed = resolution.check_pixelization(idsp, pixel_velocity, label='test spectra')
    assert not passed, 'The check should report failure where the wide pixels bound it'
    nbelow = int(np.sum(idsp < pixel_velocity / np.sqrt(12.0)))
    assert f'in {nbelow} of {NPIX} pixels' in str(record[0].message), \
        'Only the pixels wider than the dispersion allows should be counted'


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
# The identity preparation
# ----------------------------------------------------------------------
def test_identity_records_that_no_matching_was_done():
    """
    The identity match has no kernel, no offset, and nothing unmatched.

    ``performed`` is what separates it from a genuine match that happened to
    leave no offset: both have ``dvar_inst = 0``, but only one is corrected.
    """
    identity = resolution.ResolutionMatch.identity(NPIX, VELSCALE)
    assert not identity.performed, 'The identity match should report that it was not performed'
    assert identity.dvar_inst == 0.0, 'The identity match should leave no offset'
    assert identity.kernel_sigma is None, 'The identity match should carry no kernel'
    assert identity.kernel_sigma_pixels is None, \
        'The identity match should have no kernel in pixels either'
    assert identity.n_unmatched == 0, 'Nothing was attempted, so nothing should be unmatched'
    assert identity.sigma_floor == 0.0, 'The identity match should impose no floor'

    matched = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    assert matched.performed, 'A computed match should report that it was performed'


def test_identity_leaves_the_dispersion_uncorrected():
    """With no offset, the astrophysical variance is just the fitted one squared."""
    identity = resolution.ResolutionMatch.identity(NPIX, VELSCALE)
    assert np.isclose(identity.astrophysical_variance(30.0), 900.0), \
        'The identity match should apply no correction to the fitted dispersion'


def test_identity_kernel_cannot_be_applied():
    """
    Applying the identity match is a coding error, not a no-op.

    The caller is meant to skip Step 1 entirely.  Silently returning the input
    would hide a call site that believes it convolved.
    """
    identity = resolution.ResolutionMatch.identity(NPIX, VELSCALE)
    with pytest.raises(DC3CodingError, match='not performed'):
        resolution.apply_kernel(np.ones(NPIX), identity)


def test_identity_repr_says_uncorrected():
    """The summary makes the absence of a correction visible."""
    assert 'uncorrected' in repr(resolution.ResolutionMatch.identity(NPIX, VELSCALE)), \
        'The identity match should describe itself as uncorrected'


# ----------------------------------------------------------------------
# Jumps in resolution
# ----------------------------------------------------------------------
def test_idsp_breaks_find_a_step():
    """A step in the dispersion is one break, at the first pixel after it."""
    idsp = np.concatenate([np.full(100, 30.0), np.full(100, 45.0)])
    assert np.array_equal(resolution.idsp_breaks(idsp), [100]), \
        'A step in the dispersion should be one break, at the first pixel after it'


def test_idsp_breaks_ignore_a_smooth_vector():
    """A dispersion that varies smoothly, even by a large total, has no breaks."""
    idsp = np.linspace(30.0, 60.0, 4000)
    assert resolution.idsp_breaks(idsp).size == 0, 'A smooth vector was reported as jumping'


def test_idsp_breaks_measure_the_change_against_the_smaller_value():
    """
    The change is relative to the smaller dispersion, so up and down agree.

    A step of 1.5 per cent of the smaller value exceeds the default one per
    cent whichever way it goes.
    """
    up = np.concatenate([np.full(10, 40.0), np.full(10, 40.6)])
    assert np.array_equal(resolution.idsp_breaks(up), [10]), 'A rising step was missed'
    assert np.array_equal(resolution.idsp_breaks(up[::-1]), [10]), 'A falling step was missed'
    assert resolution.idsp_breaks(up, tol=0.02).size == 0, \
        'A step of 1.5 per cent should be smooth at a tolerance of 2 per cent'


def test_idsp_breaks_report_a_spread_jump_as_a_run():
    """A jump interpolated over a few pixels flags each boundary it crosses."""
    idsp = np.concatenate([np.full(50, 30.0), [35.0, 40.0], np.full(50, 45.0)])
    assert np.array_equal(resolution.idsp_breaks(idsp), [50, 51, 52]), \
        'A jump spread over three boundaries should flag all three'


@pytest.mark.parametrize('bad', [np.zeros(10), np.ones((2, 10))], ids=['zero', '2d'])
def test_idsp_breaks_reject_an_invalid_vector(bad):
    """The dispersion must be a positive vector."""
    with pytest.raises(DC3ResolutionError):
        resolution.idsp_breaks(bad)


# ----------------------------------------------------------------------
# Sampling diagnostics
# ----------------------------------------------------------------------
@pytest.mark.parametrize('sigma', [3.0, 6.0, 20.0, 50.0])
def test_minimum_velscale_ratio_is_the_smallest_that_samples_the_fwhm(sigma):
    """
    The returned ratio puts two pixels across the FWHM, and one fewer does not.

    Checking both sides is what makes this the *smallest* such ratio rather
    than merely a sufficient one.
    """
    idsp = np.full(NPIX, sigma)
    ratio = resolution.minimum_velscale_ratio(idsp, VELSCALE)
    pixels_per_fwhm = sigma * resolution.SIGMA_TO_FWHM / (VELSCALE / ratio)
    assert pixels_per_fwhm >= 2.0, \
        f'At ratio {ratio} the FWHM spans {pixels_per_fwhm:.2f} pixels, fewer than two'
    if ratio > 1:
        coarser = sigma * resolution.SIGMA_TO_FWHM / (VELSCALE / (ratio - 1))
        assert coarser < 2.0, f'Ratio {ratio - 1} already suffices, so {ratio} is not minimal'


def test_minimum_velscale_ratio_is_set_by_the_narrowest_pixel():
    """The criterion must hold everywhere, so the narrowest point decides."""
    idsp = np.full(NPIX, 50.0)
    idsp[NPIX // 2] = 3.0
    assert resolution.minimum_velscale_ratio(idsp, VELSCALE) \
        == resolution.minimum_velscale_ratio(np.full(NPIX, 3.0), VELSCALE), \
        'The ratio should be set by the narrowest line-spread function in the range'


def test_minimum_velscale_ratio_rejects_a_nonpositive_dispersion():
    """A zero or negative dispersion has no sampling requirement to meet."""
    with pytest.raises(DC3ResolutionError, match='must be positive'):
        resolution.minimum_velscale_ratio(np.zeros(NPIX), VELSCALE)


def test_check_pixelization_accepts_a_plausible_dispersion():
    """A dispersion well above the pixel-integration bound passes silently."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolution.check_pixelization(np.full(NPIX, 30.0), VELSCALE), \
            'A dispersion of several pixels should pass the check'


def test_check_pixelization_warns_below_the_pixel_integration_bound():
    """
    A dispersion under Delta/sqrt(12) is flagged, and the likely cause named.

    Pixel integration alone gives at least that much, so a smaller value is
    most likely an error in the vector -- typically angstroms given as km/s.
    """
    idsp = np.full(NPIX, 30.0)
    idsp[:10] = 0.9 * VELSCALE / np.sqrt(12.0)
    with pytest.warns(UserWarning, match='angstroms'):
        passed = resolution.check_pixelization(idsp, VELSCALE, label='test spectra')
    assert not passed, 'The check should report failure when it warns'


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
    part in 1e12 to a single interior element restores the correct answer at
    this length, which is what rules out any numerical explanation.  (At twenty
    thousand pixels a perturbation that small is lost in rounding; see
    ``apply_kernel``'s workaround, which uses a larger one.)

    **This test is the signal to remove that workaround.**  The defect has been
    reported upstream; when a ``ppxf`` release fixes it, the first assertion
    fails.
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
        'off-by-one appears to be fixed.  Remove the workaround in '
        'resolution._break_kernel_uniformity, and revisit the guard on epsilon_sigma'
    )

    perturbed = uniform.copy()
    perturbed[npix // 2] *= 1 - 1e-12
    assert np.isclose(realised(ppxf_util.varsmooth(x, y, perturbed)), 0.5, rtol=0.02), (
        'Perturbing one interior element by 1e-12 did not restore the correct kernel; the cause '
        'is no longer the integer sample count, so the diagnosis needs redoing'
    )


@pytest.mark.parametrize(
    'npix,oversample',
    [(2000, 2), (2000, 4), (20000, 2), (20000, 4), (300000, 4), (1000000, 2)]
)
def test_apply_kernel_avoids_the_off_by_one(npix, oversample):
    r"""
    A uniform kernel is applied correctly, whatever the length and oversampling.

    dc3's own construction produces an exactly uniform kernel whenever the two
    resolutions differ by a constant, which is a common case; at the default
    epsilon_sigma it is exactly 0.1 px, on varsmooth's clip.  apply_kernel
    passes pixel coordinates, whose gradient is exact, so nothing but its
    workaround stands between that case and the upstream defect.

    The lengths reach past a million stretched samples, where a fixed
    perturbation of 1e-6 overshoots and broadens the line just as the defect
    does; the workaround's perturbation is scaled to the stretched span
    instead.

    The reference is the same perturbation made ten times larger, which
    lengthens varsmooth's internal grid by a tenth of a sample rather than a
    hundredth: any size inside the window between the rounding floor and one
    sample must give the same answer, so agreement means the defect is cured
    fully, not partly.  The unperturbed kernel is checked to fail, so the test
    is discriminating at every length and oversampling.  The oversamplings are
    those apply_kernel accepts, 2 and above; the workaround itself is tested at
    1 as well, below.
    """
    match = resolution.match_resolution(np.full(npix, 30.0), np.full(npix, 30.5), VELSCALE)
    kernel = match.kernel_sigma_pixels
    assert np.all(kernel == kernel[0]), \
        'A constant resolution difference should give an exactly uniform kernel'
    assert np.isclose(kernel[0], resolution.VARSMOOTH_MIN_SIG), \
        'This test needs the kernel at the default epsilon_sigma of 0.1 px'

    center = float(npix // 3)
    flux = lsf.gaussian_comb(npix, np.array([center]), 1.0)
    x = np.arange(npix, dtype=float)

    def width(out):
        return lsf.fit_line(out, center, 1.1)[2]

    larger = kernel.copy()
    larger[0] *= 1 + 0.1 / (oversample * (npix - 1))
    reference = width(ppxf_util.varsmooth(x, flux, larger, oversample=oversample))
    measured = width(resolution.apply_kernel(flux, match, oversample=oversample))
    defective = width(ppxf_util.varsmooth(x, flux, kernel, oversample=oversample))

    assert not np.isclose(defective, reference, rtol=5e-3), \
        'The unperturbed uniform kernel did not trigger the defect, so this test is vacuous'
    assert np.isclose(measured, reference, rtol=1e-4), (
        f'apply_kernel broadened a line to {measured:.5f} px rather than {reference:.5f} px; the '
        'workaround for the upstream off-by-one no longer protects a uniform kernel'
    )


@pytest.mark.parametrize('oversample', [1, 0, -2])
def test_apply_kernel_requires_an_oversampling_of_at_least_two(oversample):
    r"""
    An oversampling below 2 is refused.

    At 1 a uniform kernel is applied exactly while one that varies even
    slightly is broadened by up to :math:`\Delta^2/3`, so results would depend
    on whether the kernel happened to be uniform.
    """
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    with pytest.raises(DC3ResolutionError, match='at least 2'):
        resolution.apply_kernel(np.ones(NPIX), match, oversample=oversample)


def test_pixel_coordinates_reproduce_the_logarithmic_call():
    """
    Passing pixel coordinates is the same convolution as passing log wavelength.

    On a logarithmic grid the two differ only in how varsmooth converts the
    kernel to pixels, which is exact either way, so with the same kernel they
    agree to round-off; and the workaround's perturbation moves the result from
    the unperturbed call by only a few parts in 1e7 of the peak.
    """
    npix = 3000
    centers = np.arange(100, npix - 100, 61.37)
    flux = lsf.gaussian_comb(npix, centers, 1.0)
    kernel_pixels = 0.1 + 1.9 * np.abs(np.sin(np.arange(npix) / 400.0))
    match = resolution.ResolutionMatch(
        kernel_pixels * VELSCALE, 0.0, np.zeros(npix, dtype=bool), VELSCALE, 0.1
    )
    out = resolution.apply_kernel(flux, match, oversample=2)

    perturbed = resolution._break_kernel_uniformity(match.kernel_sigma_pixels, oversample=2)
    same = ppxf_util.varsmooth(loglam(npix), flux, perturbed * DLOGLAM, oversample=2)
    assert np.allclose(out, same, rtol=0.0, atol=1e-10 * np.amax(out)), \
        'Pixel coordinates and log wavelength gave different convolutions of the same kernel'

    unperturbed = ppxf_util.varsmooth(
        loglam(npix), flux, match.kernel_sigma_pixels * DLOGLAM, oversample=2
    )
    assert np.amax(np.absolute(out - unperturbed)) < 1e-6 * np.amax(out), \
        'The workaround perturbation changed the convolved spectrum by more than 1e-6 of its peak'


@pytest.mark.parametrize('oversample', [1, 4])
def test_workaround_increases_only_the_largest_element(oversample):
    """
    The perturbation raises the maximum, by a hundredth of a stretched sample.

    Raising the maximum always lengthens varsmooth's internal grid, and cannot
    be undone by its clip, which only raises values.  The stretched span of
    this kernel is oversample * (1 + 2 + 1 + 4) samples, so the maximum is
    raised by 0.01 / (8 oversample).  See _break_kernel_uniformity.
    """
    sig = np.array([0.1, 0.4, 0.2, 0.4, 0.1])
    perturbed = resolution._break_kernel_uniformity(sig, oversample=oversample)
    assert np.isclose(perturbed[1], 0.4 * (1 + 0.01 / (8 * oversample)), rtol=1e-14, atol=0), \
        'The first maximum was not raised by a hundredth of the stretched span'
    assert np.array_equal(np.delete(perturbed, 1), np.delete(sig, 1)), \
        'Elements other than the maximum were changed'
    assert sig[1] == 0.4, 'The input kernel was modified in place'


# ----------------------------------------------------------------------
# Applying the kernel
# ----------------------------------------------------------------------
def test_apply_kernel_conserves_flux():
    """Convolution redistributes flux; it does not create or destroy it."""
    flux = np.zeros(NPIX)
    flux[NPIX // 2] = 1.0
    match = resolution.ResolutionMatch(
        np.full(NPIX, 5.0 * VELSCALE), 0.0, np.zeros(NPIX, dtype=bool), VELSCALE, 0.1
    )
    assert np.isclose(np.sum(resolution.apply_kernel(flux, match)), 1.0, rtol=1e-3), \
        'Convolution did not conserve the total flux'


def test_apply_kernel_uses_each_pixels_own_width():
    """
    On pixels of varying width, each is convolved by its own kernel in pixels.

    The same kernel in km/s spans more pixels where the pixels are narrower, so
    a comb convolved on such a grid must broaden by kernel_sigma_pixels at each
    line, not by a single conversion.
    """
    npix = 3000
    pixel_velocity = np.linspace(4.0, 8.0, npix)
    centers = np.arange(150, npix - 150, 97.3)
    flux = lsf.gaussian_comb(npix, centers, 2.0)
    match = resolution.ResolutionMatch(
        np.full(npix, 12.0), 0.0, np.zeros(npix, dtype=bool), pixel_velocity, 0.1
    )
    out = resolution.apply_kernel(flux, match)
    expected = np.sqrt(4.0 + np.square(np.interp(centers, np.arange(npix), 12.0 / pixel_velocity)))
    _, sigma = lsf.measure_comb(out, centers, expected)
    assert np.allclose(sigma, expected, rtol=0.03), \
        'The realised kernel does not follow the per-pixel width in pixels'


def test_apply_kernel_refuses_a_spectrum_of_the_wrong_length():
    """The spectrum must be on the grid the kernel was computed for."""
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    with pytest.raises(DC3CodingError, match='kernel was computed for'):
        resolution.apply_kernel(np.ones(NPIX + 1), match)


def test_apply_kernel_handles_a_set_of_spectra():
    """A 2-D input is convolved spectrum by spectrum."""
    flux = np.vstack([np.full(NPIX, 1.0), np.full(NPIX, 2.0)])
    match = resolution.ResolutionMatch(
        np.full(NPIX, 5.0 * VELSCALE), 0.0, np.zeros(NPIX, dtype=bool), VELSCALE, 0.1
    )
    out = resolution.apply_kernel(flux, match)
    assert out.shape == flux.shape, 'The convolved set does not have the input shape'
    assert np.isclose(out[0, NPIX // 2], 1.0, rtol=1e-3), 'A constant was not preserved'
    assert np.isclose(out[1, NPIX // 2], 2.0, rtol=1e-3), \
        'The spectra were mixed rather than convolved independently'


def test_repr_reports_the_sense_of_the_offset():
    """The representation says which spectrum is at higher resolution."""
    match = resolution.match_resolution(np.full(NPIX, 10.0), np.full(NPIX, 40.0), VELSCALE)
    assert 'higher resolution' in repr(match), \
        'The representation does not report the sense of the offset'


# ----------------------------------------------------------------------
# The excess of the convolution, and correcting for it
# ----------------------------------------------------------------------
def test_varsmooth_excess_runs_from_zero_to_a_sixth():
    """
    The predicted excess vanishes for a vanishing kernel and tends to 1/6 for a wide one.

    At small k interpolating back at the original nodes returns the original
    samples, and the excess falls to zero in proportion to k; at large k the
    kernel resolves the triangle of linear interpolation, whose variance is
    1/6.  Between, it rises monotonically.  The 1/(6(mD)^2) term is removed by
    a very large D.
    """
    k = np.geomspace(0.01, 10.0, 200)
    excess = resolution.varsmooth_excess(k, 2, 1e8)
    assert excess[0] < 0.01, 'The excess should vanish, in proportion to k, for a small kernel'
    assert abs(excess[-1] - 1 / 6) < 1e-6, 'The excess should tend to 1/6 for a wide kernel'
    assert np.all(np.diff(excess) >= 0), 'The excess should rise monotonically with the kernel'


def corrected_match(idsp_from, idsp_to, **kwargs):
    """Match with the convolution's excess corrected for, at an oversampling of 2."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return resolution.match_resolution(
            idsp_from, idsp_to, VELSCALE, varsmooth_oversample=2, **kwargs
        )


def test_corrected_kernel_applies_the_target():
    r"""
    The corrected kernel applies, with the convolution's excess, exactly what is needed.

    :math:`k^2 + E_1(k)` pixels squared, with :math:`D` from the widest kernel,
    makes up the difference between the two resolutions less ``dvar_inst``,
    and the achieved resolution is the target less ``dvar_inst``.  The kernel
    is narrower than the uncorrected one, and reaches the floor at one pixel.
    """
    idsp_from = np.full(NPIX, 20.0)
    idsp_to = np.linspace(25.0, 60.0, NPIX)
    match = corrected_match(idsp_from, idsp_to)
    k = match.kernel_sigma_pixels
    applied = np.square(k) + resolution.varsmooth_excess(k, 2, np.amax(k) / k)
    total = np.square(idsp_from) + applied * VELSCALE ** 2
    expected = np.square(idsp_to) - match.dvar_inst
    assert np.allclose(total, expected, rtol=1e-6, atol=0.0), \
        'The kernel and the excess it brings should together make up the difference'
    assert np.allclose(match.achieved, np.sqrt(expected), rtol=1e-6, atol=0.0), \
        'The achieved resolution should be the target less dvar_inst'
    assert np.isclose(np.amin(k), 0.1), 'The kernel should reach the floor at one pixel'
    uncorrected = resolution.match_resolution(idsp_from, idsp_to, VELSCALE)
    assert np.all(k[1:] < uncorrected.kernel_sigma_pixels[1:]), \
        'The corrected kernel should be narrower than the uncorrected one'


def test_corrected_match_reports_what_unmatched_pixels_achieve():
    r"""
    Where the target cannot be reached, the achieved resolution is what the floor leaves.

    With no pedestal allowed, pixels whose source is broader than the target
    are held at the floor kernel, and achieve their own dispersion plus the
    floor's :math:`g(\epsilon_\sigma)` -- broader than the target, as reported.
    """
    idsp_from = np.full(NPIX, 30.0)
    idsp_to = np.where(np.arange(NPIX) < NPIX // 4, 25.0, 45.0)
    match = corrected_match(idsp_from, idsp_to)
    assert match.n_unmatched == NPIX // 4, 'The pixels broader than the target should be unmatched'
    k = match.kernel_sigma_pixels
    floor = 0.01 + resolution.varsmooth_excess(0.1, 2, np.amax(k) / 0.1)
    expected = np.sqrt(30.0 ** 2 + floor * VELSCALE ** 2)
    assert np.allclose(match.achieved[match.unmatched], expected), \
        'An unmatched pixel should report its own dispersion plus what the floor kernel adds'
    assert np.all(match.achieved[match.unmatched] > idsp_to[match.unmatched]), \
        'An unmatched pixel achieves a broader resolution than the target'


def test_uncorrected_match_achieves_the_kernel_alone():
    """Without the correction the achieved resolution is the quadrature sum with the kernel."""
    idsp_from = np.full(NPIX, 20.0)
    match = resolution.match_resolution(idsp_from, np.linspace(25.0, 60.0, NPIX), VELSCALE)
    assert np.allclose(
        match.achieved, np.sqrt(np.square(idsp_from) + np.square(match.kernel_sigma))
    ), 'Without the correction only the kernel is assumed to change the resolution'
