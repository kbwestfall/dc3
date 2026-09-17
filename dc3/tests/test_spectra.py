"""
Tests for :class:`~dc3.spectra.Spectra`.
"""

import numpy as np
import pytest

from dc3.core import sampling
from dc3.pkg.exceptions import DC3Error
from dc3.spectra import Spectra, SpectrumBitMask, SpectrumMask


LOG10LAM0 = np.log10(3800.0)
DLOGLAM = 1e-4
NPIX = 50


def make_spectra(nspec=3, npix=NPIX, with_ivar=True, with_idsp=True):
    """Build a simple, entirely valid spectrum set for the tests."""
    rng = np.random.default_rng(42)
    flux = 1.0 + 0.1 * rng.standard_normal((nspec, npix))
    return Spectra(
        flux, LOG10LAM0, DLOGLAM,
        ivar=np.full((nspec, npix), 100.0) if with_ivar else None,
        idsp=np.full(npix, 30.0) if with_idsp else None,
    )


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def test_shapes_and_grid():
    """The set reports its shape and reconstructs its wavelength grid."""
    spec = make_spectra()
    assert spec.shape == (3, NPIX), 'The flux array does not have the expected shape'
    assert len(spec) == 3, 'len() does not report the number of spectra'
    assert np.isclose(spec.wave[0], 3800.0), 'The first wavelength is not the declared one'
    assert np.isclose(spec.velscale, sampling.velscale(DLOGLAM)), \
        'The velocity scale does not follow from the pixel size'


def test_single_spectrum_is_promoted_to_two_dimensions():
    """
    A 1-D input becomes a set of one, so nothing downstream special-cases it.
    """
    spec = Spectra(np.ones(NPIX), LOG10LAM0, DLOGLAM)
    assert spec.shape == (1, NPIX), 'A single spectrum was not promoted to 2-D'
    assert spec.nspec == 1, 'A single spectrum should report nspec of one'


def test_idsp_may_be_shared_across_spectra():
    """One instrumental-dispersion vector can serve every spectrum."""
    spec = make_spectra(nspec=4)
    assert spec.idsp.shape == (4, NPIX), \
        'A shared idsp vector was not broadcast to every spectrum'
    assert np.all(spec.idsp == 30.0), 'Broadcasting the idsp vector changed its values'


def test_from_wave():
    """A set can be built from an explicit wavelength vector."""
    wave = sampling.log_wavelength_grid(LOG10LAM0, DLOGLAM, NPIX)
    spec = Spectra.from_wave(wave, np.ones(NPIX))
    assert np.allclose(spec.wave, wave), 'The reconstructed grid does not match the input'


def test_from_wave_rejects_a_length_mismatch():
    """A wavelength vector that does not match the flux is an error."""
    wave = sampling.log_wavelength_grid(LOG10LAM0, DLOGLAM, NPIX)
    with pytest.raises(DC3Error, match='pixels'):
        Spectra.from_wave(wave, np.ones(NPIX + 1))


@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'ivar': np.ones((2, NPIX))}, 'ivar has shape'),
        ({'cont': np.ones((2, NPIX))}, 'cont has shape'),
        ({'idsp': np.ones((2, NPIX))}, 'idsp has shape'),
        ({'mask': np.zeros((2, NPIX), dtype=bool)}, 'mask has shape'),
    ]
)
def test_mismatched_shapes_are_rejected(kwargs, match):
    """An array whose shape disagrees with the flux is reported."""
    with pytest.raises(DC3Error, match=match):
        Spectra(np.ones((3, NPIX)), LOG10LAM0, DLOGLAM, **kwargs)


def test_nonpositive_pixel_size_is_rejected():
    """A logarithmic pixel size must be positive."""
    with pytest.raises(DC3Error, match='must be positive'):
        Spectra(np.ones(NPIX), LOG10LAM0, 0.0)


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def test_invalid_values_are_flagged_regardless_of_the_input_mask():
    """
    Unusable values are detected rather than trusted to the caller.

    A non-finite flux, a non-positive inverse variance and a non-positive
    instrumental dispersion are unusable whatever the input mask says.
    """
    flux = np.ones((1, 5))
    flux[0, 0] = np.nan
    ivar = np.full((1, 5), 100.0)
    ivar[0, 1] = 0.0
    idsp = np.full((1, 5), 30.0)
    idsp[0, 2] = -1.0

    spec = Spectra(flux, LOG10LAM0, DLOGLAM, ivar=ivar, idsp=idsp)
    assert spec.mask.INVALID[0, 0], 'A non-finite flux was not flagged'
    assert spec.mask.NOIVAR[0, 1], 'A zero inverse variance was not flagged'
    assert spec.mask.NOIDSP[0, 2], 'A negative instrumental dispersion was not flagged'
    assert np.array_equal(spec.gpm[0], [False, False, False, True, True]), \
        'The good-pixel mask does not exclude exactly the unusable pixels'


def test_boolean_mask_is_recorded_as_a_user_flag():
    """A boolean input mask means the user masked those pixels."""
    mask = np.zeros((2, NPIX), dtype=bool)
    mask[0, :5] = True
    spec = Spectra(np.ones((2, NPIX)), LOG10LAM0, DLOGLAM, mask=mask)
    assert np.all(spec.mask.USER[0, :5]), 'A boolean mask was not recorded as the USER bit'
    assert not np.any(spec.mask.USER[1]), 'Masking one spectrum affected another'


def test_integer_mask_is_adopted_as_bit_values():
    """An integer input mask is taken as raw bit values."""
    mask = np.zeros((1, NPIX), dtype=SpectrumBitMask.minimum_dtype())
    mask[0, 3] = 1 << SpectrumBitMask.bit('REGION')
    spec = Spectra(np.ones(NPIX), LOG10LAM0, DLOGLAM, mask=mask)
    assert spec.mask.REGION[0, 3], 'An integer mask was not adopted as bit values'


def test_nvalid():
    """The count of usable pixels excludes everything flagged."""
    mask = np.zeros((2, NPIX), dtype=bool)
    mask[0, :10] = True
    spec = Spectra(np.ones((2, NPIX)), LOG10LAM0, DLOGLAM, mask=mask)
    assert np.array_equal(spec.nvalid(), [NPIX - 10, NPIX]), \
        'The count of usable pixels does not match the mask'


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def test_mean_excludes_masked_pixels():
    """
    The mean is taken over unmasked pixels only.

    This is load-bearing: the mean is subtracted before correlation, so
    including masked pixels would make the zero point of the cross-correlation
    function depend on how much was masked.
    """
    flux = np.ones((1, 10))
    flux[0, :5] = 100.0          # a bright region, which will be masked
    mask = np.zeros((1, 10), dtype=bool)
    mask[0, :5] = True

    spec = Spectra(flux, LOG10LAM0, DLOGLAM, mask=mask)
    assert np.isclose(spec.mean()[0], 1.0), \
        'The mean included masked pixels; it should average only the unmasked ones'

    unmasked = Spectra(flux, LOG10LAM0, DLOGLAM)
    assert np.isclose(unmasked.mean()[0], 50.5), \
        'Without a mask, the mean should include every pixel'


def test_mean_of_a_fully_masked_spectrum_is_zero():
    """A spectrum with nothing usable gives zero rather than a division error."""
    spec = Spectra(
        np.ones((1, 10)), LOG10LAM0, DLOGLAM, mask=np.ones((1, 10), dtype=bool)
    )
    assert spec.mean()[0] == 0.0, 'A fully masked spectrum should give a mean of zero'
    assert spec.nvalid()[0] == 0, 'A fully masked spectrum should have no usable pixels'


def test_snr():
    """The signal-to-noise ratio follows the flux and inverse variance."""
    spec = Spectra(
        np.full((1, 20), 2.0), LOG10LAM0, DLOGLAM, ivar=np.full((1, 20), 25.0)
    )
    assert np.isclose(spec.snr()[0], 10.0), \
        'The signal-to-noise ratio should be the flux times the root inverse variance'


def test_fiducial_resolution():
    """
    The set reduces to one resolution, by the requested rule.

    Template preparation runs once per run and so needs a single resolution to
    match against; unless every spectrum has the same one, it matches none of
    them exactly.
    """
    idsp = np.vstack([
        np.full(NPIX, 30.0), np.full(NPIX, 40.0), np.full(NPIX, 50.0)
    ])
    spec = Spectra(np.ones((3, NPIX)), LOG10LAM0, DLOGLAM, idsp=idsp)

    assert np.allclose(spec.fiducial_resolution(), 40.0), \
        'The median fiducial should be the middle resolution'
    assert np.allclose(spec.fiducial_resolution(method='max'), 50.0), \
        'The max fiducial should be the lowest resolution present'
    assert np.allclose(spec.fiducial_resolution(method='min'), 30.0), \
        'The min fiducial should be the highest resolution present'


def test_fiducial_resolution_rejects_bad_input():
    """A set with no dispersion, or an unknown rule, is reported."""
    with pytest.raises(DC3Error, match='no instrumental dispersion'):
        make_spectra(with_idsp=False).fiducial_resolution()
    with pytest.raises(DC3Error, match='Unrecognized method'):
        make_spectra().fiducial_resolution(method='mean')


def test_snr_without_errors_is_zero():
    """A set with no inverse variance has no signal-to-noise ratio to report."""
    spec = make_spectra(with_ivar=False)
    assert np.all(spec.snr() == 0), \
        'A set without inverse variance should report zero, not fail'


# ----------------------------------------------------------------------
# Manipulation
# ----------------------------------------------------------------------
def test_copy_is_independent():
    """A copy shares no array with its original."""
    spec = make_spectra()
    other = spec.copy()
    other.flux[0, 0] = -999.0
    other.mask.turn_on('USER', select=np.s_[0, 0])
    assert spec.flux[0, 0] != -999.0, 'Modifying a copy changed the original flux'
    assert not spec.mask.USER[0, 0], 'Modifying a copy changed the original mask'


def test_selection_keeps_the_grid():
    """Selecting a subset keeps the wavelength grid and the per-pixel arrays."""
    spec = make_spectra(nspec=4)
    subset = spec[1:3]
    assert subset.nspec == 2, 'The subset does not have the expected number of spectra'
    assert subset.log10lam0 == spec.log10lam0, 'Selecting a subset changed the grid'
    assert np.allclose(subset.flux, spec.flux[1:3]), 'The subset has the wrong flux'

    single = spec[2]
    assert single.nspec == 1, 'Selecting one spectrum should give a set of one'
    assert np.allclose(single.flux[0], spec.flux[2]), 'The single selection has the wrong flux'


def test_selection_does_not_share_the_mask():
    """A selected subset owns its mask, so modifying it leaves the original alone."""
    spec = make_spectra(nspec=4)
    subset = spec[0:2]
    subset.mask.turn_on('USER')
    assert not np.any(spec.mask.USER), 'Masking a subset modified the original mask'


def test_repr_summarizes_the_set():
    """The representation reports the size and the sampling."""
    text = repr(make_spectra())
    assert '3 spectra' in text, 'The representation does not report the number of spectra'
    assert 'km/s/pix' in text, 'The representation does not report the velocity scale'


def test_mask_is_a_spectrum_mask():
    """The mask uses the declared spectral bit definitions."""
    assert isinstance(make_spectra().mask, SpectrumMask), \
        'The mask is not a SpectrumMask, so its bits would not be interpretable'
