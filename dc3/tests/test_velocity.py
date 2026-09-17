"""
Tests for :mod:`~dc3.core.velocity`.
"""

import numpy as np
import pytest

from dc3.core import velocity


@pytest.mark.parametrize(
    'forward,inverse',
    [
        (velocity.log_velocity, velocity.redshift_from_log_velocity),
        (velocity.relativistic_velocity, velocity.redshift_from_relativistic_velocity),
        (velocity.classical_velocity, velocity.redshift_from_classical_velocity),
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
        velocity.log_velocity(z_small),
        velocity.relativistic_velocity(z_small),
        velocity.classical_velocity(z_small),
    ]
    assert np.ptp(velocities) < 0.01, \
        'The conventions should agree to well under a km/s at z = 1e-4'

    z_large = 0.5
    velocities = [
        velocity.log_velocity(z_large),
        velocity.relativistic_velocity(z_large),
        velocity.classical_velocity(z_large),
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
        velocity.log_velocity(combined),
        velocity.log_velocity(z1) + velocity.log_velocity(z2)
    ), 'Logarithmic velocities did not add under composition of redshifts'

    # The classical convention does not have this property
    assert not np.isclose(
        velocity.classical_velocity(combined),
        velocity.classical_velocity(z1) + velocity.classical_velocity(z2)
    ), 'The classical convention should not be additive; the test is not discriminating'


def test_speed_of_light_is_in_the_expected_units():
    """Every velocity in dc3 is in km/s, so the constant must be too."""
    assert np.isclose(velocity.SPEED_OF_LIGHT, 299792.458), \
        'The speed of light is not expressed in km/s'
