"""
Tests for :mod:`~dc3.core.sampling`.
"""

import numpy as np
import pytest

from dc3.core import sampling
from dc3.pkg.exceptions import DC3Error


# ----------------------------------------------------------------------
# Velocity and redshift
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'forward,inverse',
    [
        (sampling.log_velocity, sampling.redshift_from_log_velocity),
        (sampling.relativistic_velocity, sampling.redshift_from_relativistic_velocity),
        (sampling.classical_velocity, sampling.redshift_from_classical_velocity),
    ]
)
def test_velocity_conversions_round_trip(forward, inverse):
    """Each velocity convention inverts exactly."""
    z = np.array([0.0, 1e-5, 1e-3, 0.01, 0.1])
    assert np.allclose(inverse(forward(z)), z), \
        f'{forward.__name__} and {inverse.__name__} are not inverses'


def test_velocity_conventions_agree_to_first_order():
    """
    The three conventions agree at small redshift and diverge at large.

    This is what makes them easy to confuse, and why they are named separately
    rather than selected by a flag.
    """
    z_small = 1e-4
    velocities = [
        sampling.log_velocity(z_small),
        sampling.relativistic_velocity(z_small),
        sampling.classical_velocity(z_small),
    ]
    assert np.ptp(velocities) < 0.01, \
        'The conventions should agree to well under a km/s at z = 1e-4'

    z_large = 0.5
    velocities = [
        sampling.log_velocity(z_large),
        sampling.relativistic_velocity(z_large),
        sampling.classical_velocity(z_large),
    ]
    assert np.ptp(velocities) > 1e4, \
        'The conventions should differ by more than 10000 km/s at z = 0.5'


def test_log_velocity_is_additive():
    """
    Composing two redshifts adds their logarithmic velocities.

    This is the property that makes de-redshifting separable from fitting: the
    velocity removed and the velocity fitted simply add.  Neither of the other
    two conventions has it.
    """
    z1, z2 = 0.01, 0.003
    combined = (1 + z1) * (1 + z2) - 1
    assert np.isclose(
        sampling.log_velocity(combined),
        sampling.log_velocity(z1) + sampling.log_velocity(z2)
    ), 'Logarithmic velocities did not add under composition of redshifts'

    # The classical convention does not have this property
    assert not np.isclose(
        sampling.classical_velocity(combined),
        sampling.classical_velocity(z1) + sampling.classical_velocity(z2)
    ), 'The classical convention should not be additive; the test is not discriminating'


# ----------------------------------------------------------------------
# The logarithmic grid
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
    assert np.isclose(sampling.velscale(dloglam), sampling.log_velocity(z_between)), \
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
    could detect.
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
