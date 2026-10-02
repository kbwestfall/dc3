"""
Tests for :mod:`~dc3.core.deredshift`.
"""

import numpy as np
import pytest

from dc3.core import deredshift, sampling
from dc3.core.velocity import log_velocity
from dc3.pkg.exceptions import DC3Error
from dc3.spectra import GalaxySpectra


LOG10LAM0 = np.log10(3800.0)
DLOGLAM = 1.09e-5
NPIX = 200
VELSCALE = sampling.velscale(DLOGLAM)


def make_spectra(nspec=3):
    """Build a galaxy spectrum set with distinguishable content in every array."""
    rng = np.random.default_rng(7)
    return GalaxySpectra(
        1.0 + 0.1 * rng.standard_normal((nspec, NPIX)),
        sampling.SpectralGrid.from_log_spacing(LOG10LAM0, DLOGLAM, NPIX),
        ivar=np.full((nspec, NPIX), 100.0),
        idsp=np.linspace(28.0, 32.0, NPIX),
    )


# ----------------------------------------------------------------------
# The shift itself
# ----------------------------------------------------------------------
def test_shift_is_the_nearest_whole_pixel():
    """The shift is the nearest integer number of pixels, not a truncation."""
    # Exactly 100.4 pixels of redshift should round down to 100
    z = np.power(10.0, 100.4 * DLOGLAM) - 1.0
    assert deredshift.pixel_shift(z, DLOGLAM).n_shift == 100, \
        'A shift of 100.4 pixels should round to 100'
    # ... and 100.6 should round up to 101
    z = np.power(10.0, 100.6 * DLOGLAM) - 1.0
    assert deredshift.pixel_shift(z, DLOGLAM).n_shift == 101, \
        'A shift of 100.6 pixels should round to 101'


def test_applied_redshift_is_exact_for_the_whole_pixel_shift():
    """
    The redshift reported as applied is the one the integer shift removes.

    Not the requested one: the difference is the sub-pixel remainder, and
    conflating the two is what would make the reported velocity wrong.
    """
    z_guess = 0.00382
    shift = deredshift.pixel_shift(z_guess, DLOGLAM)
    expected = np.power(10.0, shift.n_shift * DLOGLAM) - 1.0
    assert np.isclose(shift.z_applied, expected), \
        'The applied redshift does not correspond to the integer pixel shift'
    assert shift.z_applied != z_guess, \
        'This test is not discriminating unless the guess is not a whole number of pixels'


def test_residual_is_under_half_a_pixel():
    """
    The remainder left for the fit is bounded by half a pixel.

    It is a remainder, not an error: the fitted velocity measures displacement
    from the relabelled grid, so it is recovered rather than lost.
    """
    for z_guess in np.linspace(0.0, 0.02, 41):
        shift = deredshift.pixel_shift(z_guess, DLOGLAM)
        assert abs(shift.residual_velocity) <= VELSCALE / 2 + 1e-9, (
            f'z={z_guess} left a residual of {shift.residual_velocity:.3f} km/s, more than half '
            f'the {VELSCALE:.3f} km/s pixel'
        )


def test_zero_redshift_is_a_no_op():
    """Removing no redshift shifts nothing."""
    shift = deredshift.pixel_shift(0.0, DLOGLAM)
    assert shift.n_shift == 0, 'A zero redshift should give a zero shift'
    assert shift.z_applied == 0.0, 'A zero shift should apply no redshift'
    assert shift.velocity_applied == 0.0, 'A zero shift should remove no velocity'


def test_blueshift_is_supported():
    """A negative redshift shifts the other way."""
    shift = deredshift.pixel_shift(-0.001, DLOGLAM)
    assert shift.n_shift < 0, 'A blueshift should give a negative pixel shift'
    assert shift.velocity_applied < 0, 'A blueshift should remove a negative velocity'


@pytest.mark.parametrize(
    'z_guess,dloglam,match',
    [
        (-1.0, DLOGLAM, 'greater than -1'),
        (-2.0, DLOGLAM, 'greater than -1'),
        (0.01, 0.0, 'must be positive'),
        (0.01, -1e-5, 'must be positive'),
    ]
)
def test_invalid_input_is_rejected(z_guess, dloglam, match):
    """Nonsensical input is reported rather than producing a silent shift."""
    with pytest.raises(DC3Error, match=match):
        deredshift.pixel_shift(z_guess, dloglam)


# ----------------------------------------------------------------------
# Velocity bookkeeping
# ----------------------------------------------------------------------
def test_fitted_velocity_refers_back_exactly():
    """
    A velocity fitted in the rest frame refers back to the observed frame.

    The addition is exact because the internal convention, c ln(1+z), is the one
    that is additive on a logarithmic grid.  Verified against a redshift
    composed multiplicatively, which is how redshifts actually combine.
    """
    z_guess = 0.0038
    shift = deredshift.pixel_shift(z_guess, DLOGLAM)

    # Suppose the true total redshift is the applied one composed with a further
    # peculiar motion; the fit should measure exactly that peculiar part.
    z_peculiar = 0.0004
    z_total = (1 + shift.z_applied) * (1 + z_peculiar) - 1
    velocity_fit = log_velocity(z_peculiar)

    assert np.isclose(shift.observed_velocity(velocity_fit), log_velocity(z_total)), \
        'The fitted velocity did not add to the removed velocity to give the total'
    assert np.isclose(shift.observed_redshift(velocity_fit), z_total), \
        'The recovered total redshift does not match the composition of the two'


def test_observed_velocity_handles_arrays():
    """The bookkeeping works for a whole set of fitted velocities at once."""
    shift = deredshift.pixel_shift(0.004, DLOGLAM)
    fits = np.array([-50.0, 0.0, 50.0])
    assert np.allclose(shift.observed_velocity(fits), fits + shift.velocity_applied), \
        'The velocity correction was not applied elementwise'


def test_repr_reports_the_shift():
    """The representation gives the shift, the redshift and the remainder."""
    text = repr(deredshift.pixel_shift(0.004, DLOGLAM))
    assert 'pixels' in text, 'The representation does not report the pixel shift'
    assert 'residual' in text, 'The representation does not report the remainder'


# ----------------------------------------------------------------------
# Applying it
# ----------------------------------------------------------------------
def test_wavelengths_move_by_exactly_the_applied_redshift():
    """
    Every wavelength divides by exactly (1 + z_applied).

    This is the definition the relabelling has to satisfy, checked against the
    whole grid rather than one endpoint.
    """
    spec = make_spectra()
    shift = deredshift.pixel_shift(0.0038, DLOGLAM)
    rest = deredshift.to_rest_frame(spec, shift)
    assert np.allclose(rest.wave, spec.wave / (1 + shift.z_applied)), \
        'The relabelled wavelengths are not the observed ones divided by (1 + z_applied)'


def test_no_data_moves():
    """
    Nothing but the grid origin changes.

    This is the substantive claim of the whole module: a whole-pixel shift on a
    logarithmic grid is a relabelling, so the flux, errors, mask and
    instrumental dispersion are carried across untouched.  Any difference here
    would mean flux had been redistributed.
    """
    spec = make_spectra()
    rest = deredshift.to_rest_frame(spec, deredshift.pixel_shift(0.0038, DLOGLAM))

    assert np.array_equal(rest.flux, spec.flux), 'The flux was altered by de-redshifting'
    assert np.array_equal(rest.ivar, spec.ivar), 'The inverse variance was altered'
    assert np.array_equal(rest.idsp, spec.idsp), \
        'The instrumental dispersion was altered; in velocity units it is redshift-independent '\
        'per pixel and must be carried across unchanged'
    assert np.array_equal(rest.mask.mask, spec.mask.mask), 'The mask was altered'
    assert rest.npix == spec.npix, \
        'Pixels were lost; a relabelling truncates nothing, unlike rolling the arrays'
    assert rest.dloglam == spec.dloglam, 'The sampling changed, so this was not a pure shift'
    assert isinstance(rest, GalaxySpectra), 'De-redshifting lost the GalaxySpectra type'


def test_the_original_is_untouched():
    """De-redshifting returns a new set and leaves the input alone."""
    spec = make_spectra()
    original = spec.log10lam0
    rest = deredshift.to_rest_frame(spec, deredshift.pixel_shift(0.0038, DLOGLAM))
    assert spec.log10lam0 == original, 'De-redshifting modified the input spectra in place'
    assert rest.log10lam0 != original, 'The returned spectra were not actually shifted'
    rest.flux[0, 0] = -999.0
    assert spec.flux[0, 0] != -999.0, 'The returned spectra share arrays with the input'


def test_round_trip_through_the_observed_frame():
    """Removing a redshift and putting it back recovers the original grid."""
    spec = make_spectra()
    shift = deredshift.pixel_shift(0.0038, DLOGLAM)
    rest = deredshift.to_rest_frame(spec, shift)
    back = deredshift.to_rest_frame(
        rest, deredshift.DeRedshift(shift.z_applied, -shift.n_shift, DLOGLAM)
    )
    assert np.allclose(back.wave, spec.wave), \
        'Shifting to the rest frame and back did not recover the original wavelengths'


def test_shift_for_the_wrong_grid_is_rejected():
    """
    A shift computed for a different pixel size is refused.

    Applying it would move the spectra to wavelengths that are wrong by the
    ratio of the two samplings, with nothing downstream able to notice.
    """
    spec = make_spectra()
    wrong = deredshift.pixel_shift(0.0038, DLOGLAM * 2)
    with pytest.raises(DC3Error, match='pixel size'):
        deredshift.to_rest_frame(spec, wrong)


def test_zero_shift_leaves_the_grid_alone():
    """A zero shift is a genuine no-op on the wavelengths."""
    spec = make_spectra()
    rest = deredshift.to_rest_frame(spec, deredshift.pixel_shift(0.0, DLOGLAM))
    assert np.array_equal(rest.wave, spec.wave), 'A zero shift changed the wavelengths'


def test_rest_frame_wavelengths_are_shorter():
    """De-redshifting moves a spectrum bluewards, which is the point."""
    spec = make_spectra()
    rest = deredshift.to_rest_frame(spec, deredshift.pixel_shift(0.0038, DLOGLAM))
    assert np.all(rest.wave < spec.wave), \
        'Removing a positive redshift should shorten every wavelength'
