"""
Tests for :mod:`~dc3.templates`.
"""

import warnings

import numpy as np
import pytest

from dc3 import templates
from dc3.core import sampling
from dc3.par.dc3par import TemplatePar
from dc3.pkg.exceptions import DC3Error
from dc3.spectra import Spectra


GALAXY_DLOGLAM = 1.09e-5
GALAXY_NPIX = 400
GALAXY_LOG10LAM0 = np.log10(4000.0)


def make_galaxy(nspec=5, idsp_low=40.0, idsp_high=45.0):
    """A galaxy set spanning 4000-4040 A with a varying resolution."""
    return Spectra(
        np.ones((nspec, GALAXY_NPIX)), GALAXY_LOG10LAM0, GALAXY_DLOGLAM,
        ivar=np.full((nspec, GALAXY_NPIX), 100.0),
        idsp=np.linspace(idsp_low, idsp_high, GALAXY_NPIX),
    )


def make_library(idsp=20.0, ntpl=2, wave_range=(3900.0, 4150.0), ratio=2):
    """A library at finer sampling, spanning beyond the galaxy."""
    dloglam = GALAXY_DLOGLAM / ratio
    log10lam0 = np.log10(wave_range[0])
    npix = int(np.ceil((np.log10(wave_range[1]) - log10lam0) / dloglam))
    wave = sampling.log_wavelength_grid(log10lam0, dloglam, npix)
    one = 1.0 + 0.3 * np.sin((wave - wave[0]) / 7.0)
    flux = np.vstack([one * (1 + 0.1 * i) for i in range(ntpl)])
    return templates.TemplateLibrary(
        flux, log10lam0, dloglam, key='TEST', idsp=np.full(npix, idsp)
    )


def prepare_quietly(library, galaxy, par, **kwargs):
    """Run the pipeline, suppressing the out-of-range warning the setup causes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return templates.prepare(library, galaxy, par, **kwargs)


# ----------------------------------------------------------------------
# TemplateLibrary
# ----------------------------------------------------------------------
def test_library_requires_a_dispersion():
    """
    Templates with no instrumental dispersion cannot be prepared.

    There would be nothing to match their resolution *from*, so this is caught
    at construction rather than producing a meaningless kernel later.
    """
    with pytest.raises(DC3Error, match='no instrumental dispersion'):
        templates.TemplateLibrary(np.ones((2, 100)), 3.6, 1e-5)


def test_library_rejects_errors():
    """
    Templates carrying errors are refused.

    Templates are treated as noise-free throughout, as in ppxf; only the galaxy
    contributes to the covariance.  Errors here most likely mean the galaxy and
    template arguments were exchanged, which is worth saying out loud.
    """
    with pytest.raises(DC3Error, match='treated as noise-free'):
        templates.TemplateLibrary(
            np.ones((2, 100)), 3.6, 1e-5, idsp=np.full(100, 20.0),
            ivar=np.full((2, 100), 1.0)
        )


def test_library_is_a_spectra():
    """A library is a Spectra, so its arrays are reachable without indirection."""
    library = make_library()
    assert isinstance(library, Spectra), 'A TemplateLibrary should be a Spectra'
    assert library.flux.ndim == 2, 'The flux should be reachable directly'
    assert library.ntpl == library.nspec, 'ntpl should report the number of templates'


def test_library_from_spectra():
    """A library can be built from an existing spectrum set."""
    npix = 100
    spec = Spectra(np.ones((3, npix)), 3.6, 1e-5, idsp=np.full(npix, 20.0))
    library = templates.TemplateLibrary.from_spectra(spec, key='FROMSPEC')
    assert library.key == 'FROMSPEC', 'The library name was not carried over'
    assert np.array_equal(library.flux, spec.flux), 'The flux was not carried over'


def test_library_copy_and_slice_keep_the_name():
    """
    The library name survives copying and selecting a subset.

    The name feeds the preparation cache key, so losing it on a slice would make
    two different libraries collide.
    """
    library = make_library(ntpl=4)
    assert library.copy().key == library.key, 'Copying the library lost its name'
    subset = library[1:3]
    assert subset.key == library.key, 'Selecting a subset lost the library name'
    assert subset.ntpl == 2, 'The subset does not have the expected number of templates'


# ----------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------
def test_prepared_resolution_is_the_fiducial_offset_by_dvar_inst():
    r"""
    The identity :math:`\sigma_{\rm prep}^2 = \sigma_{\rm fid}^2 - dvar\_inst` holds.

    This is what the whole scheme rests on: everything downstream converts a
    fitted dispersion to an astrophysical one using exactly this offset, so if
    the prepared resolution were anything else, every reported dispersion would
    be wrong.
    """
    galaxy = make_galaxy()
    prepared = prepare_quietly(make_library(), galaxy, TemplatePar())

    fiducial = np.interp(prepared.wave, galaxy.wave, galaxy.fiducial_resolution())
    expected = np.sqrt(np.square(fiducial) - prepared.dvar_inst)
    # Compare only where the galaxy actually constrains the fiducial
    inside = (prepared.wave >= galaxy.wave[0]) & (prepared.wave <= galaxy.wave[-1])
    assert np.allclose(prepared.idsp[0][inside], expected[inside], rtol=1e-6), \
        'The prepared resolution is not the fiducial offset by dvar_inst'


def test_higher_resolution_templates_give_a_positive_offset():
    """
    Templates of higher resolution than the galaxy leave a positive offset.

    This is the regime that holds the fitted dispersion away from zero.
    """
    prepared = prepare_quietly(make_library(idsp=20.0), make_galaxy(), TemplatePar())
    assert prepared.dvar_inst > 0, \
        'Templates of higher resolution should leave a positive dvar_inst'


def test_velscale_ratio_sets_the_output_sampling():
    """The prepared grid is finer than the galaxy's by exactly velscale_ratio."""
    galaxy = make_galaxy()
    for ratio in [1, 2, 4]:
        prepared = prepare_quietly(
            make_library(ratio=max(ratio, 2)), galaxy, TemplatePar(velscale_ratio=ratio)
        )
        assert np.isclose(galaxy.velscale / prepared.velscale, ratio), \
            f'velscale_ratio={ratio} did not give a grid that many times finer'
        assert prepared.velscale_ratio == ratio, 'The ratio was not recorded on the result'


def test_prepared_is_a_spectra_carrying_its_provenance():
    """The result is a Spectra that also knows how it was made."""
    prepared = prepare_quietly(make_library(), make_galaxy(), TemplatePar())
    assert isinstance(prepared, Spectra), 'PreparedTemplates should be a Spectra'
    assert prepared.match is not None, 'The resolution matching was not recorded'
    assert len(prepared.key) > 0, 'No cache key was recorded'


def test_selection_keeps_the_provenance():
    """
    Selecting a subset keeps the offset and the key.

    The offset is a property of the preparation, not of any one template, so it
    must survive the multi-template path discarding templates of zero weight.
    """
    prepared = prepare_quietly(make_library(ntpl=4), make_galaxy(), TemplatePar())
    subset = prepared[0:2]
    assert subset.dvar_inst == prepared.dvar_inst, 'The offset was lost on selection'
    assert subset.key == prepared.key, 'The cache key was lost on selection'
    assert subset.velscale_ratio == prepared.velscale_ratio, 'The ratio was lost on selection'


def test_astrophysical_variance_round_trips():
    """The correction on the prepared set inverts the instrumental offset."""
    prepared = prepare_quietly(make_library(), make_galaxy(), TemplatePar())
    sigma_star = 120.0
    sigma_obs = np.sqrt(sigma_star ** 2 + prepared.dvar_inst)
    assert np.isclose(prepared.astrophysical_variance(sigma_obs), sigma_star ** 2), \
        'The astrophysical variance does not invert the instrumental offset'


def test_non_overlapping_wavelengths_are_rejected():
    """
    Templates that do not overlap the galaxy are an error, not an empty result.

    The most likely cause is a galaxy that was never de-redshifted, so the
    message says so.
    """
    library = make_library(wave_range=(3000.0, 3100.0))
    with pytest.raises(DC3Error, match='do not overlap'):
        templates.prepare(library, make_galaxy(), TemplatePar())


def test_template_range_beyond_the_galaxy_warns():
    """
    Extending beyond the galaxy is allowed, but the user is told.

    The fiducial resolution is held at its nearest measured value there, so the
    matching rests on an assumption rather than a measurement.
    """
    with pytest.warns(UserWarning, match='outside the galaxy'):
        templates.prepare(make_library(), make_galaxy(), TemplatePar())


def test_galaxy_is_not_modified():
    """Preparation reads the galaxy; it never alters it."""
    galaxy = make_galaxy()
    before = galaxy.flux.copy(), galaxy.idsp.copy(), galaxy.log10lam0
    prepare_quietly(make_library(), galaxy, TemplatePar())
    assert np.array_equal(galaxy.flux, before[0]), 'Preparation altered the galaxy flux'
    assert np.array_equal(galaxy.idsp, before[1]), 'Preparation altered the galaxy resolution'
    assert galaxy.log10lam0 == before[2], 'Preparation altered the galaxy wavelength grid'


def test_library_is_not_modified():
    """Preparation returns a new set; the raw library is left alone."""
    library = make_library()
    before = library.flux.copy()
    prepare_quietly(library, make_galaxy(), TemplatePar())
    assert np.array_equal(library.flux, before), 'Preparation altered the raw templates'


# ----------------------------------------------------------------------
# The cache key
# ----------------------------------------------------------------------
def test_key_is_stable_for_identical_input():
    """The same preparation gives the same key, so a cache can be reused."""
    galaxy, library, par = make_galaxy(), make_library(), TemplatePar()
    first = prepare_quietly(library, galaxy, par)
    second = prepare_quietly(library, galaxy, par)
    assert first.key == second.key, 'Two identical preparations gave different keys'


@pytest.mark.parametrize(
    'field,value',
    [
        ('library_key', 'OTHER'),
        ('velscale', 8.0),
        ('velscale_ratio', 4),
        ('epsilon_sigma', 0.2),
        ('sigma_floor', 5.0),
        ('oversample', 2),
    ]
)
def test_key_changes_with_every_input(field, value):
    """
    Changing anything that changes the product changes the key.

    A key that missed one of these would let a stale cache be reused, which is
    worse than having no cache at all.
    """
    base = dict(
        library_key='TEST', fiducial_idsp=np.full(10, 40.0), velscale=7.5, velscale_ratio=1,
        epsilon_sigma=0.1, sigma_floor=0.0, oversample=1,
    )
    changed = dict(base)
    changed[field] = value
    assert templates.preparation_key(**base) != templates.preparation_key(**changed), \
        f'Changing {field} did not change the cache key'


def test_key_changes_with_the_fiducial_resolution():
    """The key covers the fiducial resolution, which is a vector."""
    base = dict(
        library_key='TEST', fiducial_idsp=np.full(10, 40.0), velscale=7.5, velscale_ratio=1,
        epsilon_sigma=0.1, sigma_floor=0.0, oversample=1,
    )
    changed = dict(base, fiducial_idsp=np.full(10, 41.0))
    assert templates.preparation_key(**base) != templates.preparation_key(**changed), \
        'A different fiducial resolution should give a different key'


# ----------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------
def test_unmatched_regions_are_masked_when_requested():
    """
    Regions that could not be matched are flagged only when asked for.

    The default is to retain them and report the mismatch rather than hide it.
    """
    # Templates of LOWER resolution than the galaxy, with no pedestal allowed
    library = make_library(idsp=60.0)
    galaxy = make_galaxy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kept = templates.prepare(library, galaxy, TemplatePar(mask_unmatched_idsp=False))
        masked = templates.prepare(library, galaxy, TemplatePar(mask_unmatched_idsp=True))

    assert kept.match.n_unmatched > 0, \
        'This test needs a configuration that leaves pixels unmatched'
    assert not np.any(kept.mask.UNMATCHED), \
        'Unmatched pixels were masked despite mask_unmatched_idsp being False'
    assert np.any(masked.mask.UNMATCHED), \
        'Unmatched pixels were not masked despite mask_unmatched_idsp being True'
