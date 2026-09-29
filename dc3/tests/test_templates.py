"""
Tests for :mod:`~dc3.templates`.
"""

import warnings

import numpy as np
from pydantic import ValidationError
import pytest

from dc3 import templates
from dc3.core import resolution, sampling
from dc3.pkg.exceptions import DC3Error
from dc3.spectra import Spectra
from dc3.templates import TemplateLibraryPar, TemplatePar


GALAXY_DLOGLAM = 1.09e-5
GALAXY_NPIX = 400
GALAXY_LOG10LAM0 = np.log10(4000.0)


def make_galaxy(nspec=5, idsp_low=40.0, idsp_high=45.0, with_idsp=True):
    """A galaxy set spanning 4000-4040 A with a varying resolution."""
    return Spectra(
        np.ones((nspec, GALAXY_NPIX)), GALAXY_LOG10LAM0, GALAXY_DLOGLAM,
        ivar=np.full((nspec, GALAXY_NPIX), 100.0),
        idsp=np.linspace(idsp_low, idsp_high, GALAXY_NPIX) if with_idsp else None,
    )


def make_library(idsp=20.0, ntpl=2, wave_range=(3900.0, 4150.0), ratio=2):
    """
    A library at finer sampling, spanning beyond the galaxy.

    ``idsp`` may be None for a library that carries no instrumental dispersion.
    """
    dloglam = GALAXY_DLOGLAM / ratio
    log10lam0 = np.log10(wave_range[0])
    npix = int(np.ceil((np.log10(wave_range[1]) - log10lam0) / dloglam))
    wave = sampling.log_wavelength_grid(log10lam0, dloglam, npix)
    one = 1.0 + 0.3 * np.sin((wave - wave[0]) / 7.0)
    flux = np.vstack([one * (1 + 0.1 * i) for i in range(ntpl)])
    return templates.TemplateLibrary(
        flux, log10lam0, dloglam, key='TEST',
        idsp=None if idsp is None else np.full(npix, idsp)
    )


def prepare_quietly(library, galaxy, par=None):
    """
    Run the pipeline, suppressing the out-of-range warning the setup causes.

    The parameter set is expanded over the function's keywords, which is the
    calling pattern the package uses; see :func:`~dc3.templates.prepare`.
    """
    if par is None:
        par = TemplatePar()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return templates.prepare(library, galaxy, **par.to_kwargs())


# ----------------------------------------------------------------------
# The parameter sets
# ----------------------------------------------------------------------
def test_epsilon_sigma_floor_matches_varsmooth():
    """
    epsilon_sigma cannot be set below the clip inside the upstream convolution.

    Allowing it would mean the code believed it applied a narrower kernel than
    it did, making the instrumental offset wrong by the difference.
    """
    assert TemplatePar().epsilon_sigma == 0.1, \
        'epsilon_sigma default does not match the upstream varsmooth clip of 0.1 px'
    with pytest.raises(ValidationError):
        TemplatePar(epsilon_sigma=0.05)


def test_velscale_ratio_accepts_an_integer_or_auto():
    """The ratio is a positive integer, or 'auto'; nothing else."""
    assert TemplatePar(velscale_ratio='auto').velscale_ratio == 'auto', \
        'velscale_ratio should accept "auto"'
    assert TemplatePar(velscale_ratio=3).velscale_ratio == 3, \
        'velscale_ratio should accept a positive integer'
    for bad in [0, -1, 'other']:
        with pytest.raises(ValidationError):
            TemplatePar(velscale_ratio=bad)


@pytest.mark.parametrize('value', ['auto', 3])
def test_velscale_ratio_survives_a_configuration_file(value, tmp_path):
    """
    Both forms of velscale_ratio can be set from TOML.

    This is why automatic selection is spelled 'auto' rather than None: TOML
    has no null, so a None-valued setting could never be written in a file.
    """
    path = tmp_path / 'template.toml'
    path.write_text(TemplatePar(velscale_ratio=value).to_toml())
    assert TemplatePar.from_toml(path).velscale_ratio == value, \
        f'velscale_ratio = {value!r} did not survive a round trip through TOML'


@pytest.mark.parametrize(
    'wave_limit', [[5000.0], [4000.0, 5000.0, 6000.0], [7000.0, 3000.0]],
    ids=['too-short', 'too-long', 'out-of-order']
)
def test_library_wavelength_limit_must_be_an_ordered_pair(wave_limit):
    """A wavelength limit that is not an ordered pair is rejected."""
    with pytest.raises(ValidationError):
        TemplateLibraryPar(wave_limit=wave_limit)


# ----------------------------------------------------------------------
# TemplateLibrary
# ----------------------------------------------------------------------
def test_library_without_a_dispersion_is_accepted():
    """
    Templates with no instrumental dispersion can still form a library.

    They cannot be resolution-matched, but that is handled by :func:`prepare`,
    which warns and proceeds with ``dvar_inst = 0``; refusing them here would
    make that path unreachable.
    """
    library = templates.TemplateLibrary(np.ones((2, 100)), 3.6, 1e-5)
    assert library.idsp is None, 'A library built without idsp should carry none'


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
    prepared = prepare_quietly(make_library(), galaxy)

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
    prepared = prepare_quietly(make_library(idsp=20.0), make_galaxy())
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
    prepared = prepare_quietly(make_library(), make_galaxy())
    assert isinstance(prepared, Spectra), 'PreparedTemplates should be a Spectra'
    assert prepared.match is not None, 'The resolution matching was not recorded'
    assert len(prepared.key) > 0, 'No cache key was recorded'


def test_selection_keeps_the_provenance():
    """
    Selecting a subset keeps the offset and the key.

    The offset is a property of the preparation, not of any one template, so it
    must survive the multi-template path discarding templates of zero weight.
    """
    prepared = prepare_quietly(make_library(ntpl=4), make_galaxy())
    subset = prepared[0:2]
    assert subset.dvar_inst == prepared.dvar_inst, 'The offset was lost on selection'
    assert subset.key == prepared.key, 'The cache key was lost on selection'
    assert subset.velscale_ratio == prepared.velscale_ratio, 'The ratio was lost on selection'


def test_astrophysical_variance_round_trips():
    """The correction on the prepared set inverts the instrumental offset."""
    prepared = prepare_quietly(make_library(), make_galaxy())
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
        templates.prepare(library, make_galaxy(), **TemplatePar().to_kwargs())


def test_template_range_beyond_the_galaxy_warns():
    """
    Extending beyond the galaxy is allowed, but the user is told.

    The fiducial resolution is held at its nearest measured value there, so the
    matching rests on an assumption rather than a measurement.
    """
    with pytest.warns(UserWarning, match='outside the galaxy'):
        templates.prepare(make_library(), make_galaxy(), **TemplatePar().to_kwargs())


def test_galaxy_is_not_modified():
    """Preparation reads the galaxy; it never alters it."""
    galaxy = make_galaxy()
    before = galaxy.flux.copy(), galaxy.idsp.copy(), galaxy.log10lam0
    prepare_quietly(make_library(), galaxy)
    assert np.array_equal(galaxy.flux, before[0]), 'Preparation altered the galaxy flux'
    assert np.array_equal(galaxy.idsp, before[1]), 'Preparation altered the galaxy resolution'
    assert galaxy.log10lam0 == before[2], 'Preparation altered the galaxy wavelength grid'


def test_library_is_not_modified():
    """Preparation returns a new set; the raw library is left alone."""
    library = make_library()
    before = library.flux.copy()
    prepare_quietly(library, make_galaxy())
    assert np.array_equal(library.flux, before), 'Preparation altered the raw templates'


# ----------------------------------------------------------------------
# When matching does not happen
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'library_idsp,galaxy_idsp,named',
    [
        (None, True, 'templates'),
        (20.0, False, 'galaxy spectra'),
        (None, False, 'templates and the galaxy spectra'),
    ],
    ids=['no-template-idsp', 'no-galaxy-idsp', 'neither']
)
def test_missing_dispersion_warns_and_leaves_no_offset(library_idsp, galaxy_idsp, named):
    """
    Without both resolution vectors, preparation proceeds with ``dvar_inst = 0``.

    The zero is a statement of ignorance rather than a measurement, so the
    warning must say the reported dispersions are uncorrected, and name which
    input is missing so the user knows what to supply.
    """
    library = make_library(idsp=library_idsp)
    galaxy = make_galaxy(with_idsp=galaxy_idsp)
    with pytest.warns(UserWarning, match='UNCORRECTED') as record:
        prepared = templates.prepare(library, galaxy, **TemplatePar().to_kwargs())
    assert any(named in str(w.message) for w in record), \
        f'The warning does not name the {named} as the input lacking a dispersion'
    assert prepared.dvar_inst == 0.0, 'An unmatched preparation should leave dvar_inst at zero'
    assert not prepared.match.performed, \
        'An unmatched preparation should record that no matching was performed'


def test_unmatched_templates_are_resampled_but_not_convolved():
    """
    Step 1 is skipped entirely, but Step 2 still runs.

    The templates must still land on the galaxy's sampling to be fit at all, so
    the output is exactly what resampling the raw templates gives.
    """
    library = make_library(idsp=None)
    galaxy = make_galaxy()
    prepared = prepare_quietly(library, galaxy)
    expected = sampling.Resample(
        library.flux, x=library.wave, newRange=[library.wave[0], library.wave[-1]],
        newdx=galaxy.dloglam, newLog=True
    )
    assert np.allclose(prepared.flux, np.atleast_2d(expected.outy)), \
        'Unmatched templates should be resampled without any convolution'


def test_unmatched_preparation_carries_the_template_dispersion_if_any():
    """
    The prepared dispersion is the templates' own, when they have one.

    With no matching the templates keep their native resolution, so that is
    what the prepared set should report -- and nothing, if they had none.
    """
    with_template_idsp = prepare_quietly(make_library(idsp=20.0), make_galaxy(with_idsp=False))
    assert np.allclose(with_template_idsp.idsp, 20.0), \
        'Unmatched templates should keep their own instrumental dispersion'

    without = prepare_quietly(make_library(idsp=None), make_galaxy())
    assert without.idsp is None, \
        'Templates with no dispersion should give a prepared set with none'


# ----------------------------------------------------------------------
# Output sampling and input diagnostics
# ----------------------------------------------------------------------
def narrow_lsf_setup():
    """
    Templates whose prepared line-spread function is too narrow for ratio 1.

    With the templates at 3 km/s and the galaxy just above, the prepared
    dispersion is about 3 km/s against a galaxy pixel of about 7.5 km/s, so a
    FWHM of two pixels needs a ratio of 3.
    """
    return make_library(idsp=3.0, ratio=4), make_galaxy(idsp_low=4.0, idsp_high=5.0)


def warning_messages(library, galaxy, par):
    """Run the pipeline and return every warning message it issued."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        prepared = templates.prepare(library, galaxy, **par.to_kwargs())
    return prepared, [str(w.message) for w in record]


def test_auto_velscale_ratio_chooses_the_smallest_nyquist_ratio():
    """
    'auto' gives the smallest ratio that puts two pixels across the FWHM.

    The expectation is computed from the prepared dispersion directly, so this
    checks the pipeline applies the criterion to the right line-spread
    function -- the one Step 1 produced, not the templates' native one.
    """
    library, galaxy = narrow_lsf_setup()
    prepared = prepare_quietly(library, galaxy, TemplatePar(velscale_ratio='auto'))
    sigma_min = 2.0 / resolution.SIGMA_TO_FWHM
    expected = int(np.ceil(sigma_min * galaxy.velscale / np.amin(prepared.idsp)))
    assert expected > 1, 'This test needs a configuration that requires oversampling'
    assert prepared.velscale_ratio == expected, \
        f'"auto" chose a ratio of {prepared.velscale_ratio}, not the minimum of {expected}'
    assert np.isclose(prepared.velscale, galaxy.velscale / expected), \
        'The prepared sampling does not follow from the chosen ratio'


def test_undersampled_velscale_ratio_warns_with_the_needed_value():
    """
    An explicit ratio too small to sample the prepared LSF warns, naming a fix.

    A ratio that is large enough must not warn, so the test is discriminating.
    """
    library, galaxy = narrow_lsf_setup()
    needed = prepare_quietly(library, galaxy, TemplatePar(velscale_ratio='auto')).velscale_ratio

    _, messages = warning_messages(library, galaxy, TemplatePar(velscale_ratio=1))
    nyquist = [m for m in messages if 'not Nyquist-sampled' in m]
    assert len(nyquist) == 1, 'An undersampling ratio should issue exactly one Nyquist warning'
    assert f'velscale_ratio = {needed}' in nyquist[0], \
        'The Nyquist warning should name the ratio that would suffice'

    _, messages = warning_messages(library, galaxy, TemplatePar(velscale_ratio=needed))
    assert not any('not Nyquist-sampled' in m for m in messages), \
        'A sufficient ratio should not issue a Nyquist warning'


def test_auto_velscale_ratio_without_a_template_dispersion_falls_back_to_one():
    """With no dispersion to choose from, 'auto' uses 1 and says so."""
    _, messages = warning_messages(
        make_library(idsp=None), make_galaxy(), TemplatePar(velscale_ratio='auto')
    )
    assert any('Using a ratio of 1' in m for m in messages), \
        '"auto" with no template dispersion should warn that it fell back to 1'


def test_default_setup_raises_no_sampling_warnings():
    """
    The ordinary configuration triggers neither new diagnostic.

    Otherwise the warnings would fire on every well-formed run and be ignored.
    """
    _, messages = warning_messages(make_library(), make_galaxy(), TemplatePar())
    assert not any('not Nyquist-sampled' in m for m in messages), \
        'A well-sampled preparation should not issue a Nyquist warning'
    assert not any('pixel integration' in m for m in messages), \
        'Plausible dispersions should not issue a pixelization warning'


@pytest.mark.parametrize('which', ['templates', 'galaxy spectra'])
def test_implausibly_small_dispersion_warns(which):
    """
    A dispersion below what pixel integration gives is flagged, on either input.

    Each input is checked against its own sampling, so the test gives each a
    value that is below its own bound but not the other's.
    """
    if which == 'templates':
        library, galaxy = make_library(idsp=0.1), make_galaxy()
    else:
        library, galaxy = make_library(), make_galaxy(idsp_low=0.5, idsp_high=0.6)
    _, messages = warning_messages(library, galaxy, TemplatePar())
    flagged = [m for m in messages if 'pixel integration' in m]
    assert len(flagged) == 1, f'An implausible {which} dispersion should be flagged once'
    assert which in flagged[0], f'The warning should name the {which}'


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


def test_key_distinguishes_an_unmatched_preparation():
    """
    An unmatched preparation never shares a key with a matched one.

    Otherwise a cache could serve templates at their native resolution where
    matched ones were expected, and every corrected dispersion would be wrong.
    """
    base = dict(
        library_key='TEST', fiducial_idsp=np.full(10, 40.0), velscale=7.5, velscale_ratio=1,
        epsilon_sigma=0.1, sigma_floor=0.0, oversample=1,
    )
    unmatched = dict(base, fiducial_idsp=None)
    assert templates.preparation_key(**base) != templates.preparation_key(**unmatched), \
        'An unmatched preparation should not share a key with a matched one'


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
        kept = templates.prepare(
            library, galaxy, **TemplatePar(mask_unmatched_idsp=False).to_kwargs()
        )
        masked = templates.prepare(
            library, galaxy, **TemplatePar(mask_unmatched_idsp=True).to_kwargs()
        )

    assert kept.match.n_unmatched > 0, \
        'This test needs a configuration that leaves pixels unmatched'
    assert not np.any(kept.mask.UNMATCHED), \
        'Unmatched pixels were masked despite mask_unmatched_idsp being False'
    assert np.any(masked.mask.UNMATCHED), \
        'Unmatched pixels were not masked despite mask_unmatched_idsp being True'
