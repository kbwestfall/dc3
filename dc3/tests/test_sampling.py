"""
Tests for :mod:`~dc3.core.sampling`.
"""

import numpy as np
import pytest

from dc3.core import sampling, velocity
from dc3.pkg.exceptions import DC3Error


# ----------------------------------------------------------------------
# The dc3 logarithmic grid
# ----------------------------------------------------------------------
def test_velscale_round_trip():
    """The velocity scale and the logarithmic pixel size invert."""
    dloglam = 1e-4
    assert np.isclose(sampling.dloglam_from_velscale(sampling.velscale(dloglam)), dloglam), \
        'velscale and dloglam_from_velscale are not inverses'


def test_velscale_matches_a_direct_calculation():
    """
    The velocity scale equals the velocity difference between adjacent pixels.

    Checked against the logarithmic velocity of the redshift between two
    neighbouring pixels, which is the definition it must satisfy.
    """
    dloglam = 1e-4
    wave = sampling.log_wavelength_grid(np.log10(5000.0), dloglam, 3)
    z_between = wave[1] / wave[0] - 1
    assert np.isclose(sampling.velscale(dloglam), velocity.log_velocity(z_between)), \
        'The velocity scale does not match the velocity between adjacent pixels'


# ----------------------------------------------------------------------
# The velocity offset between grids
# ----------------------------------------------------------------------
def test_grid_velocity_offset_vanishes_for_identical_grids():
    """Two grids starting at the same wavelength have no offset."""
    assert sampling.grid_velocity_offset(3.6, 3.6, 1e-4) == 0.0, \
        'Identical grids should have no velocity offset'


@pytest.mark.parametrize('npix', [3.0, 0.25, -1.5])
def test_grid_velocity_offset_is_the_shift_in_pixels(npix):
    """A galaxy grid starting ``npix`` pixels redward is offset by ``npix`` velocity scales."""
    dloglam = 1e-4
    offset = sampling.grid_velocity_offset(3.6, 3.6 + npix * dloglam, dloglam)
    assert np.isclose(offset, npix * sampling.velscale(dloglam)), \
        'The offset should be the grid shift expressed in velocity'


@pytest.mark.parametrize('velscale_ratio', [1, 2, 3, 4])
def test_grid_velocity_offset_matches_mangadap(velscale_ratio):
    """
    The offset agrees with ``mangadap``'s calculation from wavelength vectors.

    ``PPXFFit.ppxf_tpl_obj_voff`` works from the vectors, taking the mean of
    the first ``velscale_ratio`` template log-wavelengths as the template's
    reference; its expression is reproduced here so the two formulations are
    compared rather than one checked against itself.
    """
    dloglam_obj = 1e-4
    obj_wave = sampling.log_wavelength_grid(3.61, dloglam_obj, 50)
    tpl_wave = sampling.log_wavelength_grid(3.6003, dloglam_obj / velscale_ratio, 400)
    velscale = sampling.velscale(dloglam_obj)
    dlogl = np.log(obj_wave[0]) - np.mean(np.log(tpl_wave[0:velscale_ratio]))
    expected = dlogl * velscale / np.diff(np.log(obj_wave[0:2]))[0]
    offset = sampling.grid_velocity_offset(
        np.log10(tpl_wave[0]), np.log10(obj_wave[0]), dloglam_obj, velscale_ratio=velscale_ratio
    )
    assert np.isclose(offset, expected, rtol=1e-8), \
        f'The offset disagrees with the mangadap calculation at velscale_ratio={velscale_ratio}'


@pytest.mark.parametrize('velscale_ratio', [1, 3])
def test_grid_velocity_offset_recovers_a_doppler_shift_from_a_pixel_lag(velscale_ratio):
    """
    The pixel lag plus the offset gives back the Doppler shift.

    A line at a known rest wavelength is placed in a template and, shifted by a
    known velocity, in a galaxy on a grid with a different starting wavelength.
    Its pixel position in each follows from the grid, and the lag between them
    -- after binning the template down by ``velscale_ratio`` -- plus the offset
    must equal the imposed velocity.  This is the property the offset exists
    to provide.
    """
    dloglam_obj = 1e-4
    dloglam_tpl = dloglam_obj / velscale_ratio
    log10lam0_tpl, log10lam0_obj = 3.6, 3.6137
    v_true = 1234.5
    log10_rest = np.log10(5000.0)
    log10_obs = log10_rest + v_true / (velocity.SPEED_OF_LIGHT * np.log(10.0))

    tpl_pixel = (log10_rest - log10lam0_tpl) / dloglam_tpl
    # Binning by velscale_ratio maps template pixel j to binned pixel (j - (r-1)/2)/r
    tpl_binned_pixel = (tpl_pixel - (velscale_ratio - 1) / 2) / velscale_ratio
    obj_pixel = (log10_obs - log10lam0_obj) / dloglam_obj
    lag = obj_pixel - tpl_binned_pixel

    offset = sampling.grid_velocity_offset(
        log10lam0_tpl, log10lam0_obj, dloglam_obj, velscale_ratio=velscale_ratio
    )
    assert np.isclose(sampling.velscale(dloglam_obj) * lag + offset, v_true), \
        'The pixel lag plus the grid offset should recover the imposed velocity'


# ----------------------------------------------------------------------
# Converting between pixel centres and borders
# ----------------------------------------------------------------------
def test_centers_and_borders_invert():
    """Converting centres to borders and back recovers the centres."""
    centers = np.linspace(1.0, 10.0, 10)
    borders = sampling.centers_to_borders(centers)
    assert borders.size == centers.size + 1, 'There should be one more border than centre'
    assert np.allclose(sampling.borders_to_centers(borders), centers), \
        'Converting centres to borders and back did not recover the centres'


def test_centers_and_borders_invert_geometrically():
    """The same holds for geometric binning."""
    centers = np.power(10.0, np.linspace(np.log10(3800.0), np.log10(5000.0), 20))
    borders = sampling.centers_to_borders(centers, log=True)
    assert np.allclose(sampling.borders_to_centers(borders, log=True), centers), \
        'Geometric centres were not recovered from their borders'


# ----------------------------------------------------------------------
# Grids used by the tests of sampling_type and SpectralGrid
# ----------------------------------------------------------------------
def miles_like_wave():
    """A linear grid at MILES-like sampling: 0.9 A pixels from 3500 A."""
    return 3500.0 + 0.9 * np.arange(4300)


def dms_like_wave():
    """A logarithmic grid at 7.5 km/s per pixel, as in the published DiskMass data."""
    return sampling.log_wavelength_grid(np.log10(4000.0), 1.09e-5, 4000)


def spliced_borders():
    """
    The borders of two contiguous linear sections: 50 pixels of 0.9 A, then 50 of 0.4 A.

    Border 50 is the splice.  Returned with the centres, which are the linear
    centres of the pixels.
    """
    first = 3500.0 - 0.45 + 0.9 * np.arange(51)
    second = first[-1] + 0.4 * np.arange(1, 51)
    borders = np.concatenate([first, second])
    return (borders[:-1] + borders[1:]) / 2, borders


# ----------------------------------------------------------------------
# Detecting the sampling
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'wave,kind',
    [
        (miles_like_wave(), 'linear'),
        (dms_like_wave(), 'log'),
        (spliced_borders()[0], 'irregular'),
    ],
    ids=['linear', 'log', 'spliced']
)
def test_sampling_type_identifies_each_kind(wave, kind):
    """Linear, logarithmic and spliced vectors are each recognized."""
    assert sampling.sampling_type(wave) == kind, f'A {kind} vector was not recognized as such'


@pytest.mark.parametrize('make_wave', [miles_like_wave, dms_like_wave], ids=['linear', 'log'])
def test_sampling_type_survives_float32_storage(make_wave):
    """
    A regular grid stored in single precision is still recognized as regular.

    Checked on the float32 values converted back to double precision, which is
    how such a grid arrives: the dtype no longer shows the rounding.
    """
    wave = make_wave()
    rounded = wave.astype(np.float32).astype(float)
    assert sampling.sampling_type(rounded) == sampling.sampling_type(wave), \
        'Storing a regular grid in single precision changed its detected sampling'


def test_float32_floor_is_what_admits_a_finely_sampled_grid():
    """
    At fine sampling, float32 rounding alone exceeds the default tolerance.

    The DiskMass-like grid rounded to float32 departs from a uniform grid by
    more than 1e-3 pixels, so without the floor it would be called irregular.
    This makes the preceding test discriminating rather than vacuous.
    """
    rounded = dms_like_wave().astype(np.float32).astype(float)
    departure = sampling._uniform_fit(np.log10(rounded))[2]
    assert departure > 1e-3, 'This test needs a grid whose rounding exceeds the default tolerance'
    assert sampling.sampling_type(rounded) == 'log', \
        'The float32 floor did not admit a regular grid stored in single precision'


def test_sampling_type_honours_a_user_tolerance():
    """
    A departure beyond the default tolerance is accepted if the user allows it.

    One pixel centre is displaced by a hundredth of a pixel: irregular at the
    default of 1e-3 pixels, linear at 0.05.
    """
    wave = miles_like_wave()
    wave[2000] += 0.01 * 0.9
    assert sampling.sampling_type(wave) == 'irregular', \
        'A displacement of 0.01 pixels should exceed the default tolerance'
    assert sampling.sampling_type(wave, tol=0.05) == 'linear', \
        'A displacement of 0.01 pixels should be within a tolerance of 0.05'


def test_sampling_type_calls_an_ambiguous_vector_logarithmic():
    """
    A vector both descriptions fit is called logarithmic.

    Ten pixels spanning a tiny fraction of their wavelength are linear and
    logarithmic to far better than the tolerance, so the choice cannot matter.
    """
    wave = sampling.log_wavelength_grid(np.log10(5000.0), 1e-6, 10)
    assert sampling._uniform_fit(wave)[2] < 1e-3, \
        'This test needs a vector that is also linear to within the tolerance'
    assert sampling.sampling_type(wave) == 'log', \
        'A vector both descriptions fit should be called logarithmic'


@pytest.mark.parametrize(
    'wave,match',
    [
        (np.array([5000.0]), 'at least two'),
        (np.array([-1.0, 1.0, 3.0]), 'positive'),
        (np.array([5000.0, 5001.0, 5001.0]), 'strictly ascending'),
        (np.ones((2, 3)), 'one-dimensional'),
    ]
)
def test_sampling_type_rejects_an_invalid_vector(wave, match):
    """A vector that cannot describe a grid is reported."""
    with pytest.raises(DC3Error, match=match):
        sampling.sampling_type(wave)


# ----------------------------------------------------------------------
# SpectralGrid
# ----------------------------------------------------------------------
def test_log_grid_geometry():
    """
    A logarithmic grid's centres are the geometric centres of its pixels.

    Every pixel then has the same velocity width, the grid's velocity scale.
    """
    grid = sampling.SpectralGrid.from_log_spacing(np.log10(4000.0), 1.09e-5, 100)
    assert grid.kind == 'log' and grid.is_log, 'A logarithmic grid did not report its kind'
    assert grid.npix == 100 and grid.borders.size == 101, 'The grid has the wrong number of pixels'
    assert np.allclose(sampling.borders_to_centers(grid.borders, log=True), grid.wave), \
        "A logarithmic grid's centres are not the geometric centres of its pixels"
    assert np.allclose(grid.loglam, np.log10(grid.wave)), 'loglam is not log10 of the centres'
    assert np.allclose(grid.pixel_velocity, grid.velscale), \
        'Every pixel of a logarithmic grid should have the velocity scale as its width'
    assert np.isclose(grid.velscale, sampling.velscale(1.09e-5)), \
        'The velocity scale does not follow from the pixel size'


def test_linear_grid_geometry():
    """
    A linear grid's centres are the linear centres of its pixels.

    Its pixels narrow in velocity with wavelength, as a constant width in
    angstroms is a falling fraction of the wavelength.
    """
    grid = sampling.SpectralGrid.from_linear_spacing(3500.0, 0.9, 100)
    assert grid.kind == 'linear' and not grid.is_log, 'A linear grid did not report its kind'
    assert np.allclose(np.diff(grid.borders), 0.9), 'A linear grid does not have equal pixels'
    assert np.allclose(sampling.borders_to_centers(grid.borders, log=False), grid.wave), \
        "A linear grid's centres are not the linear centres of its pixels"
    assert np.all(np.diff(grid.pixel_velocity) < 0), \
        "A linear grid's pixels should narrow in velocity with wavelength"
    assert np.allclose(
        grid.pixel_velocity, velocity.SPEED_OF_LIGHT * np.log(grid.borders[1:] / grid.borders[:-1])
    ), 'The velocity width of each pixel does not follow from its borders'


@pytest.mark.parametrize('name', ['log10lam0', 'dloglam', 'velscale'])
def test_log_only_quantities_are_refused_for_other_grids(name):
    """
    A quantity that exists only for a logarithmic grid is refused, not guessed.

    A linear grid has no single velocity scale; returning one would be wrong
    everywhere but at one wavelength.
    """
    grid = sampling.SpectralGrid.from_linear_spacing(3500.0, 0.9, 100)
    with pytest.raises(DC3Error, match='only for a logarithmically sampled grid'):
        getattr(grid, name)
    with pytest.raises(DC3Error, match='only for a logarithmically sampled grid'):
        grid.shifted(1)


def test_irregular_grid_keeps_supplied_borders():
    """An irregular grid given its borders uses them as given."""
    wave, borders = spliced_borders()
    grid = sampling.SpectralGrid('irregular', wave.size, wave=wave, borders=borders)
    assert np.array_equal(grid.borders, borders), 'Supplied borders were not used as given'
    assert np.array_equal(grid.wave, wave), 'Supplied centres were not used as given'
    assert np.allclose(
        grid.pixel_velocity, velocity.SPEED_OF_LIGHT * np.log(borders[1:] / borders[:-1])
    ), 'The velocity width of each pixel does not follow from its borders'


def test_irregular_grid_derives_borders_from_linear_centres():
    """
    Borders derived for an irregular grid are exact except beside a splice.

    The centres are taken to be linear centres.  Within each section of a
    spliced grid that is exact; at the splice, the border just below the last
    pixel before it is misplaced by a quarter of the change in pixel size.
    This pins the behaviour the class documents.
    """
    wave, true_borders = spliced_borders()
    derived = sampling.SpectralGrid('irregular', wave.size, wave=wave).borders
    error = derived - true_borders
    misplaced = np.flatnonzero(np.absolute(error) > 1e-9)
    assert misplaced.tolist() == [49], \
        'Only the border just below the last pixel before the splice should be misplaced'
    assert np.isclose(error[49], (0.9 - 0.4) / 4), \
        'The misplaced border should be off by a quarter of the change in pixel size'


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'kind': 'spline', 'npix': 10, 'start': 1.0, 'step': 1.0}, 'Unknown grid kind'),
        ({'kind': 'log', 'npix': 1, 'start': 3.6, 'step': 1e-4}, 'at least two pixels'),
        ({'kind': 'log', 'npix': 10, 'start': 3.6}, 'both its start and its step'),
        ({'kind': 'log', 'npix': 10, 'start': 3.6, 'step': 0.0}, 'must be positive'),
        ({'kind': 'linear', 'npix': 10, 'start': 0.2, 'step': 1.0}, 'positive wavelengths'),
        ({'kind': 'irregular', 'npix': 10}, 'needs its pixel centres'),
        ({'kind': 'irregular', 'npix': 10, 'wave': np.arange(1.0, 6.0)}, 'Expected 10 pixel'),
    ]
)
def test_invalid_grids_are_refused(kwargs, match):
    """Parameters that do not describe a valid grid are reported."""
    with pytest.raises(DC3Error, match=match):
        sampling.SpectralGrid(**kwargs)


@pytest.mark.parametrize(
    'borders,match',
    [
        (np.arange(0.5, 5.0), 'Expected 6 pixel borders'),
        (np.array([0.5, 1.5, 2.5, 3.5, 4.5, 4.9]), 'strictly within its borders'),
    ]
)
def test_inconsistent_irregular_borders_are_refused(borders, match):
    """Borders must number one more than the pixels and bracket every centre."""
    with pytest.raises(DC3Error, match=match):
        sampling.SpectralGrid('irregular', 5, wave=np.arange(1.0, 6.0), borders=borders)


def test_grid_arrays_are_read_only():
    """
    A grid's arrays cannot be changed, in place or by assignment.

    A grid is shared by every spectrum set built on it, so a change through one
    would silently change them all.
    """
    grid = sampling.SpectralGrid.from_log_spacing(3.6, 1e-4, 10)
    for name in ['wave', 'borders', 'loglam', 'pixel_velocity']:
        with pytest.raises(ValueError, match='read-only'):
            getattr(grid, name)[0] = 1.0
        with pytest.raises(AttributeError):
            setattr(grid, name, np.zeros(10))


def test_grid_does_not_freeze_the_callers_arrays():
    """The arrays given to an irregular grid are copied, so the caller's stay writeable."""
    wave, borders = spliced_borders()
    sampling.SpectralGrid('irregular', wave.size, wave=wave, borders=borders)
    assert wave.flags.writeable, "Building a grid made the caller's centres read-only"
    assert borders.flags.writeable, "Building a grid made the caller's borders read-only"


def test_shifted_relabels_by_whole_pixels():
    """Shifting moves every pixel by the same number of pixels, and keeps the sampling."""
    grid = sampling.SpectralGrid.from_log_spacing(np.log10(4000.0), 1e-4, 20)
    for n in [3, -2, 0]:
        shifted = grid.shifted(n)
        assert isinstance(shifted, sampling.SpectralGrid), 'Shifting did not return a grid'
        assert np.allclose(shifted.loglam, grid.loglam + n * grid.dloglam), \
            f'Shifting by {n} did not move every pixel by {n} pixels'
        assert shifted.dloglam == grid.dloglam and shifted.npix == grid.npix, \
            'Shifting changed the sampling of the grid'


# ----------------------------------------------------------------------
# SpectralGrid.from_vector
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'make_wave,kind',
    [(miles_like_wave, 'linear'), (dms_like_wave, 'log')],
    ids=['linear', 'log']
)
def test_from_vector_replaces_a_regular_vector_by_its_fitted_grid(make_wave, kind):
    """
    A regular vector becomes the uniform grid it describes.

    Built from the float32-rounded values, the fitted grid recovers the true
    one more closely than the stored values themselves do, which is the point
    of fitting rather than adopting them.
    """
    wave = make_wave()
    rounded = wave.astype(np.float32).astype(float)
    grid = sampling.SpectralGrid.from_vector(rounded)
    assert grid.kind == kind, f'A {kind} vector did not give a {kind} grid'
    step = np.diff(wave)
    assert np.amax(np.absolute(grid.wave - wave) / step.mean()) \
        < np.amax(np.absolute(rounded - wave) / step.mean()), \
        'The fitted grid should be closer to the true grid than the rounded values are'


def test_from_vector_adopts_an_irregular_vector_as_given():
    """An irregular vector is kept exactly, with its borders if they are supplied."""
    wave, borders = spliced_borders()
    grid = sampling.SpectralGrid.from_vector(wave, borders=borders)
    assert grid.kind == 'irregular', 'A spliced vector did not give an irregular grid'
    assert np.array_equal(grid.wave, wave), 'The centres of an irregular vector were altered'
    assert np.array_equal(grid.borders, borders), 'The supplied borders were not used'


def test_from_vector_accepts_consistent_borders_for_a_regular_vector():
    """Borders agreeing with the fitted grid are accepted."""
    wave = 3500.0 + 0.9 * np.arange(20)
    grid = sampling.SpectralGrid.from_vector(wave, borders=3500.0 - 0.45 + 0.9 * np.arange(21))
    assert grid.kind == 'linear', 'Consistent borders changed the detected sampling'


def test_from_vector_refuses_inconsistent_borders_for_a_regular_vector():
    """
    Borders a regular grid cannot reproduce are reported, not discarded.

    Here they are displaced by a sixth of a pixel from where a linear grid of
    these centres puts them.
    """
    wave = 3500.0 + 0.9 * np.arange(20)
    with pytest.raises(DC3Error, match='depart from that grid'):
        sampling.SpectralGrid.from_vector(wave, borders=3500.0 - 0.30 + 0.9 * np.arange(21))


def test_repr_reports_the_kind():
    """The summary identifies the sampling."""
    assert 'km/s/pix' in repr(sampling.SpectralGrid.from_log_spacing(3.6, 1e-4, 10)), \
        'A logarithmic grid should report its velocity scale'
    assert 'dlam' in repr(sampling.SpectralGrid.from_linear_spacing(3500.0, 0.9, 10)), \
        'A linear grid should report its pixel size'
    assert 'irregular' in repr(sampling.SpectralGrid.from_vector(spliced_borders()[0])), \
        'An irregular grid should say so'


# ----------------------------------------------------------------------
# Finding splices
# ----------------------------------------------------------------------
def high_resolution_spliced_wave(nper=3000):
    """
    Centres of two linear sections near 5000 A at lambda/dlambda of 1e5 and 1.7e5.

    Pixel nper is the first of the second section.  Fine enough that wavelengths
    stored in single precision carry noise of order 1e-2 in the ratio of
    neighbouring pixel sizes.
    """
    first = 5000.0 + 0.05 * np.arange(nper)
    second = first[-1] + 0.04 + 0.03 * np.arange(nper)
    return np.concatenate([first, second])


@pytest.mark.parametrize(
    'grid',
    [
        sampling.SpectralGrid.from_vector(miles_like_wave()),
        sampling.SpectralGrid.from_vector(dms_like_wave()),
    ],
    ids=['linear', 'log']
)
def test_regular_grids_have_no_breaks(grid):
    """A regularly sampled grid has no splices, by definition."""
    breaks = grid.breaks()
    assert breaks.size == 0, 'A regular grid reported a jump in its sampling'
    assert breaks.dtype.kind == 'i', 'The breaks should be integer pixel indices'


def test_a_smoothly_irregular_grid_has_no_breaks():
    """
    A grid whose pixel size changes smoothly is not spliced.

    A logarithmic grid described pixel by pixel changes its width in
    wavelength by about 1e-4 per pixel, well inside the tolerance.
    """
    wave = dms_like_wave()
    grid = sampling.SpectralGrid('irregular', wave.size, wave=wave)
    assert grid.breaks().size == 0, 'A smoothly varying sampling was reported as spliced'


def test_breaks_find_a_splice_with_supplied_borders():
    """With the true borders, a splice is exactly one break, at the splice."""
    wave, borders = spliced_borders()
    grid = sampling.SpectralGrid.from_vector(wave, borders=borders)
    assert np.array_equal(grid.breaks(), [50]), \
        'A splice with known borders should give one break, at the first pixel after it'


def test_breaks_spread_over_a_run_when_borders_are_derived():
    """
    Borders derived from the centres spread a splice over adjacent boundaries.

    The border derived just below the last pixel before a splice is misplaced
    (see the SpectralGrid documentation), which changes the widths of the two
    pixels below the splice.  The jump therefore appears across a run of three
    boundaries, ending at the splice itself, and nowhere else.
    """
    grid = sampling.SpectralGrid.from_vector(spliced_borders()[0])
    breaks = grid.breaks()
    assert 50 in breaks, 'The splice itself was not flagged'
    assert np.all(np.diff(breaks) == 1), 'The flagged boundaries should form a single run'
    assert np.all((breaks >= 48) & (breaks <= 50)), \
        'Boundaries outside the two pixels below the splice were flagged'


def test_breaks_honour_the_tolerance():
    """A jump smaller than the tolerance is treated as smooth."""
    wave, borders = spliced_borders()
    grid = sampling.SpectralGrid.from_vector(wave, borders=borders)
    # The splice changes the pixel size from 0.9 to 0.4 A, a ratio of 0.44
    assert grid.breaks(tol=0.5).size == 1, 'A change of 56 per cent exceeds a tolerance of 0.5'
    assert grid.breaks(tol=0.6).size == 0, \
        'A change of 56 per cent should be treated as smooth at a tolerance of 0.6'


def test_breaks_survive_float32_storage():
    """
    A finely sampled spliced grid stored in single precision has only its splice.

    At these samplings the rounding alone puts noise into the ratio of
    neighbouring pixel sizes that exceeds the default tolerance, which the
    float32 floor absorbs; the splice, a change of 40 per cent, still stands.
    """
    exact = high_resolution_spliced_wave()
    rounded = exact.astype(np.float32).astype(float)
    grid = sampling.SpectralGrid.from_vector(rounded)
    breaks = grid.breaks()
    assert breaks.size > 0, 'The splice was lost'
    assert np.all((breaks >= 2998) & (breaks <= 3000)), \
        'Single-precision rounding was reported as a splice'

    # The floor is what admits it: without it, the rounding exceeds the tolerance
    width = np.diff(grid.borders)
    noise = np.absolute(width[1:] / width[:-1] - 1)
    noise = np.delete(noise, np.arange(2997, 3002))
    assert np.amax(noise) > 0.01, \
        'This test needs rounding noise that exceeds the default tolerance by itself'
