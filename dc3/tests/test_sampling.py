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


def test_grid_round_trip():
    """A constructed grid is recovered by grid_from_wave."""
    log10lam0, dloglam, npix = np.log10(3800.0), 1e-4, 100
    wave = sampling.log_wavelength_grid(log10lam0, dloglam, npix)
    recovered = sampling.grid_from_wave(wave)
    assert np.isclose(recovered[0], log10lam0), 'The starting wavelength was not recovered'
    assert np.isclose(recovered[1], dloglam), 'The pixel size was not recovered'


def test_linear_grid_is_rejected():
    """
    A linearly sampled grid is refused rather than silently accepted.

    Every velocity in dc3 assumes logarithmic sampling, so accepting a linear
    grid here would produce results that are wrong in a way nothing downstream
    could detect.  Resample is the supported way to convert one.
    """
    with pytest.raises(DC3Error, match='not uniformly sampled'):
        sampling.grid_from_wave(np.linspace(3800.0, 5000.0, 100))


@pytest.mark.parametrize(
    'wave,match',
    [
        (np.array([5000.0]), 'at least two'),
        (np.array([-1.0, 1.0, 3.0]), 'positive'),
        (np.array([5000.0, 4000.0, 3000.0]), 'ascending'),
    ]
)
def test_invalid_grids_are_rejected(wave, match):
    """A malformed wavelength vector is reported, not worked around."""
    with pytest.raises(DC3Error, match=match):
        sampling.grid_from_wave(wave)


# ----------------------------------------------------------------------
# Grid helpers
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


def test_grid_borders_and_centers_are_consistent():
    """The centres derived from the borders match those computed directly."""
    rng = [3800.0, 5000.0]
    npix = 50
    borders = sampling.grid_borders(rng, npix, log=True)[0]
    centers = sampling.grid_centers(rng, npix, log=True)[0]
    assert borders.size == npix + 1, 'grid_borders did not return npix+1 borders'
    assert np.allclose(sampling.borders_to_centers(borders, log=True), centers), \
        'grid_borders and grid_centers disagree about where the pixels are'


def test_grid_npix_covers_the_range():
    """The pixel count and adjusted range are mutually consistent."""
    npix, rng = sampling.grid_npix(rng=[3800.0, 5000.0], dx=1e-4, log=True)
    assert npix > 0, 'No pixels were allocated for a valid range'
    dloglam = (np.log10(rng[1]) - np.log10(rng[0])) / (npix - 1)
    assert np.isclose(dloglam, 1e-4), \
        'The adjusted range does not divide into the requested pixel size'


def test_grid_npix_requires_a_two_element_range():
    """A malformed range is reported."""
    with pytest.raises(DC3Error, match='2-element'):
        sampling.grid_npix(rng=[1.0, 2.0, 3.0], dx=0.1)


# ----------------------------------------------------------------------
# Resample
# ----------------------------------------------------------------------
def test_resample_conserves_the_integral():
    """
    Resampling conserves the integral of the input.

    This is the property that makes it usable on flux: the total is preserved
    even though the pixel boundaries move.
    """
    x = np.linspace(1.0, 100.0, 100)
    y = np.exp(-0.5 * ((x - 50.0) / 10.0) ** 2)

    r = sampling.Resample(y, x=x, newRange=[1.0, 100.0], newpix=50, newLog=False)
    integral_in = np.sum(y * np.diff(sampling.centers_to_borders(x)))
    integral_out = np.sum(r.outy * np.diff(r.outborders))
    assert np.isclose(integral_in, integral_out, rtol=1e-6), \
        'Resampling did not conserve the integral of the input'


def test_resample_preserves_a_constant():
    """A constant function resamples to the same constant."""
    x = np.linspace(1.0, 100.0, 100)
    y = np.full(100, 3.0)
    r = sampling.Resample(y, x=x, newRange=[10.0, 90.0], newpix=37, newLog=False)
    assert np.allclose(r.outy, 3.0), \
        'A constant did not survive resampling; the normalization is wrong'


def test_resample_linear_to_log():
    """
    A linearly sampled spectrum resamples onto a logarithmic grid.

    This is the pre-processing step a user needs before dc3 will accept a
    spectrum, since grid_from_wave refuses linear sampling.
    """
    wave = np.linspace(3800.0, 5000.0, 500)
    flux = 1.0 + 0.5 * np.sin((wave - 3800.0) / 100.0)

    r = sampling.Resample(flux, x=wave, newRange=[3810.0, 4990.0], newpix=400, newLog=True)
    # The output grid must now satisfy the check that rejected the input
    log10lam0, dloglam = sampling.grid_from_wave(r.outx)
    assert dloglam > 0, 'The resampled grid is not logarithmically sampled'
    assert np.all(np.isfinite(r.outy)), 'The resampled flux contains non-finite values'
    assert np.allclose(np.mean(r.outy), np.mean(flux), rtol=0.05), \
        'The resampled mean flux differs substantially from the input'


def test_resample_two_dimensional():
    """A set of spectra resamples along the last axis."""
    x = np.linspace(1.0, 100.0, 100)
    y = np.vstack([np.full(100, 1.0), np.full(100, 2.0), np.full(100, 3.0)])
    r = sampling.Resample(y, x=x, newRange=[10.0, 90.0], newpix=40, newLog=False)
    assert r.outy.shape == (3, 40), 'The resampled set does not have the expected shape'
    assert np.allclose(r.outy[:, 5], [1.0, 2.0, 3.0]), \
        'The spectra were mixed together rather than resampled independently'


def test_resample_propagates_errors():
    """Errors are resampled in quadrature."""
    x = np.linspace(1.0, 100.0, 100)
    y = np.full(100, 1.0)
    e = np.full(100, 0.1)
    r = sampling.Resample(y, e=e, x=x, newRange=[10.0, 90.0], newpix=40, newLog=False)
    assert r.oute is not None, 'No errors were returned despite errors being supplied'
    assert np.all(r.oute > 0), 'The resampled errors are not positive'


def test_resample_reports_the_valid_fraction():
    """
    The fraction of each output pixel covered by valid input is reported.

    Pixels beyond the input range have no valid data, which is what lets a
    caller distinguish "zero flux" from "no information".
    """
    x = np.linspace(10.0, 90.0, 80)
    y = np.full(80, 1.0)
    r = sampling.Resample(y, x=x, newRange=[1.0, 100.0], newpix=99, newLog=False)
    assert np.any(r.outf == 0), 'No output pixel was reported as uncovered by valid input'
    assert np.any(np.isclose(r.outf, 1.0)), 'No output pixel was reported as fully covered'


def test_resample_masks_are_honoured():
    """A masked input pixel does not contribute valid data."""
    x = np.linspace(1.0, 100.0, 100)
    y = np.full(100, 1.0)
    mask = np.zeros(100, dtype=bool)
    mask[40:60] = True
    r = sampling.Resample(y, mask=mask, x=x, newRange=[1.0, 100.0], newpix=100, newLog=False)
    assert np.any(r.outf < 1.0), 'The masked region was not reflected in the valid fraction'


def test_resample_does_not_modify_the_callers_mask():
    """
    The input mask is copied, not adopted.

    Resample merges the masks of the data and the errors into its own, which
    would otherwise modify an array the caller still holds.
    """
    x = np.linspace(1.0, 100.0, 100)
    y = np.ma.MaskedArray(np.full(100, 1.0), mask=np.zeros(100, dtype=bool))
    y.mask[10:20] = True
    mask = np.zeros(100, dtype=bool)
    sampling.Resample(y, mask=mask, x=x, newRange=[1.0, 100.0], newpix=50, newLog=False)
    assert not np.any(mask), "Resample modified the caller's mask array in place"


def test_resample_covariance():
    """
    Resampling correlates neighbouring output pixels, and can report it.

    The covariance is not used on any dc3 path yet -- spectral covariance is
    deliberately ignored -- but this is the hook if that decision changes, and
    it exercises the astropy Covariance API that replaced mangadap's.
    """
    x = np.linspace(1.0, 100.0, 100)
    y = np.full(100, 1.0)
    e = np.full(100, 0.1)
    r = sampling.Resample(
        y, e=e, x=x, newRange=[10.0, 90.0], newpix=40, newLog=False, covar=True
    )
    assert r.covar is not None, 'No covariance was returned despite being requested'
    dense = r.covar.to_dense()
    assert dense.shape == (40, 40), 'The covariance does not have one entry per output pixel pair'
    off_diagonal = dense - np.diag(np.diag(dense))
    assert np.any(np.absolute(off_diagonal) > 0), \
        'The covariance is diagonal; resampling must correlate neighbouring output pixels'


def test_resample_covariance_requires_step_resampling():
    """The covariance is only defined for step resampling, and says so."""
    x = np.linspace(1.0, 100.0, 100)
    with pytest.raises(DC3Error, match='step resampling'):
        sampling.Resample(
            np.full(100, 1.0), x=x, newRange=[10.0, 90.0], newpix=40, step=False, covar=True
        )


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'x': np.arange(10.0), 'xRange': [0.0, 9.0]}, 'One and only one'),
        ({}, 'One and only one'),
    ]
)
def test_resample_rejects_ambiguous_input_grids(kwargs, match):
    """The input grid must be specified exactly once."""
    with pytest.raises(DC3Error, match=match):
        sampling.Resample(np.ones(10), newRange=[0.0, 9.0], newpix=5, **kwargs)


def test_resample_rejects_mismatched_shapes():
    """Errors and masks must match the data."""
    x = np.arange(10.0)
    with pytest.raises(DC3Error, match='Error array shape'):
        sampling.Resample(np.ones(10), e=np.ones(5), x=x, newRange=[0.0, 9.0], newpix=5)
    with pytest.raises(DC3Error, match='Mask array shape'):
        sampling.Resample(
            np.ones(10), mask=np.zeros(5, dtype=bool), x=x, newRange=[0.0, 9.0], newpix=5
        )


def test_resample_rejects_three_dimensional_input():
    """Only 1-D and 2-D data can be resampled."""
    with pytest.raises(DC3Error, match='1D or 2D'):
        sampling.Resample(np.ones((2, 3, 4)), x=np.arange(4.0), newRange=[0.0, 3.0], newpix=2)
