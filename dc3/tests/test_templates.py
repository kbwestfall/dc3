"""
Tests for :mod:`~dc3.templates`.
"""

import warnings

import numpy as np
from pydantic import ValidationError
import pytest

from dc3 import templates
from dc3.core import resample, resolution, sampling
from dc3.pkg.exceptions import DC3Error
from dc3.spectra import GalaxySpectra, Spectra
from dc3.templates import TemplateLibraryPar, TemplatePar


GALAXY_DLOGLAM = 1.09e-5
GALAXY_NPIX = 400
GALAXY_LOG10LAM0 = np.log10(4000.0)


def galaxy_grid(dloglam=GALAXY_DLOGLAM):
    """The galaxy's logarithmic grid."""
    return sampling.SpectralGrid.from_log_spacing(GALAXY_LOG10LAM0, dloglam, GALAXY_NPIX)


def make_galaxy(nspec=5, idsp_low=40.0, idsp_high=45.0, with_idsp=True):
    """A galaxy set spanning 4000-4040 A with a varying resolution."""
    return GalaxySpectra(
        np.ones((nspec, GALAXY_NPIX)), galaxy_grid(),
        ivar=np.full((nspec, GALAXY_NPIX), 100.0),
        idsp=np.linspace(idsp_low, idsp_high, GALAXY_NPIX) if with_idsp else None,
    )


def library_flux(wave, ntpl):
    """A smooth, distinguishable flux for each template."""
    one = 1.0 + 0.3 * np.sin((wave - wave[0]) / 7.0)
    return np.vstack([one * (1 + 0.1 * i) for i in range(ntpl)])


def make_library(idsp=20.0, ntpl=2, wave_range=(3900.0, 4150.0), ratio=2):
    """
    A logarithmically sampled library at finer sampling, spanning beyond the galaxy.

    ``idsp`` may be None for a library that carries no instrumental dispersion.
    """
    dloglam = GALAXY_DLOGLAM / ratio
    log10lam0 = np.log10(wave_range[0])
    npix = int(np.ceil((np.log10(wave_range[1]) - log10lam0) / dloglam))
    grid = sampling.SpectralGrid.from_log_spacing(log10lam0, dloglam, npix)
    return templates.TemplateLibrary(
        library_flux(grid.wave, ntpl), grid, key='TEST',
        idsp=None if idsp is None else np.full(npix, idsp)
    )


def make_linear_library(idsp=20.0, ntpl=2, wave_range=(3900.0, 4150.0), dlam=0.02):
    """
    A linearly sampled library, as most libraries are delivered.

    At 0.02 A the pixels are between about 1.4 and 1.5 km/s wide, finer than
    the galaxy's 7.5 km/s throughout.
    """
    npix = int(np.ceil((wave_range[1] - wave_range[0]) / dlam))
    grid = sampling.SpectralGrid.from_linear_spacing(wave_range[0], dlam, npix)
    return templates.TemplateLibrary(
        library_flux(grid.wave, ntpl), grid, key='LINEAR',
        idsp=None if idsp is None else np.full(npix, idsp)
    )


def spliced_grid(nsections, wave_range=(3900.0, 4150.0), dlam=(0.02, 0.03)):
    """
    A grid of contiguous linear sections whose pixel size alternates.

    Built from its borders, so that each splice is a single break.

    Returns
    -------
    tuple
        The :class:`~dc3.core.sampling.SpectralGrid`, and the index of the first
        pixel of every section after the first.
    """
    edges = np.linspace(wave_range[0], wave_range[1], nsections + 1)
    borders = [np.array([edges[0]])]
    starts = []
    for i in range(nsections):
        step = dlam[i % 2]
        npix = int(np.round((edges[i + 1] - edges[i]) / step))
        starts.append(sum(b.size for b in borders) - 1)
        borders.append(borders[-1][-1] + step * np.arange(1, npix + 1))
    borders = np.concatenate(borders)
    grid = sampling.SpectralGrid(
        'irregular', borders.size - 1, wave=(borders[1:] + borders[:-1]) / 2, borders=borders
    )
    return grid, np.array(starts[1:])


def make_spliced_library(nsections, idsp=20.0, ntpl=2):
    """A library spliced from sections of alternating sampling."""
    grid, _ = spliced_grid(nsections)
    return templates.TemplateLibrary(
        library_flux(grid.wave, ntpl), grid, key='SPLICED',
        idsp=None if idsp is None else np.full(grid.npix, idsp)
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


def test_varsmooth_oversample_is_at_least_two():
    """
    The internal oversampling of the convolution defaults to 2 and cannot be lower.

    At 1, a uniform kernel is applied exactly but one that varies even slightly
    is broadened by up to a third of a pixel squared, so results would depend on
    whether the kernel happened to be uniform.
    """
    assert TemplatePar().varsmooth_oversample == 2, 'varsmooth_oversample should default to 2'
    with pytest.raises(ValidationError):
        TemplatePar(varsmooth_oversample=1)


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
    grid = sampling.SpectralGrid.from_log_spacing(3.6, 1e-5, 100)
    library = templates.TemplateLibrary(np.ones((2, 100)), grid)
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
            np.ones((2, 100)), sampling.SpectralGrid.from_log_spacing(3.6, 1e-5, 100),
            idsp=np.full(100, 20.0), ivar=np.full((2, 100), 1.0)
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
    spec = Spectra(
        np.ones((3, npix)), sampling.SpectralGrid.from_linear_spacing(4000.0, 0.5, npix),
        idsp=np.full(npix, 20.0)
    )
    library = templates.TemplateLibrary.from_spectra(spec, key='FROMSPEC')
    assert library.key == 'FROMSPEC', 'The library name was not carried over'
    assert np.array_equal(library.flux, spec.flux), 'The flux was not carried over'
    assert library.grid is spec.grid, 'The grid was not carried over'


def test_library_finds_its_splices_and_resolution_jumps():
    """
    The library reports where its sampling and its resolution jump.

    The resolution is taken from the first template, as the matching does.
    """
    grid, starts = spliced_grid(3)
    idsp = np.where(np.arange(grid.npix) < 1000, 20.0, 25.0)
    library = templates.TemplateLibrary(library_flux(grid.wave, 2), grid, idsp=idsp)
    assert np.array_equal(library.sampling_breaks, starts), \
        'The library did not report its splices at the first pixel of each new section'
    assert np.array_equal(library.idsp_breaks, [1000]), \
        'The library did not report the jump in its resolution'

    regular = make_library()
    assert regular.sampling_breaks.size == 0, 'A regular library reported a splice'
    assert regular.idsp_breaks.size == 0, 'A constant resolution reported a jump'
    assert make_library(idsp=None).idsp_breaks.size == 0, \
        'A library without a dispersion should report no jumps in it'


def test_library_tolerances_are_used_and_kept():
    """The jump tolerances govern detection and survive copying and selection."""
    grid, _ = spliced_grid(3)
    flux = library_flux(grid.wave, 3)
    # The pixel size changes by a factor of 1.5 at each splice
    assert templates.TemplateLibrary(flux, grid, sampling_jump_tol=0.6).sampling_breaks.size \
        == 0, 'A splice smaller than the tolerance was reported'
    library = templates.TemplateLibrary(flux, grid, sampling_jump_tol=0.2, idsp_jump_tol=0.05)
    for derived in [library.copy(), library[0:2]]:
        assert derived.sampling_jump_tol == 0.2, 'The sampling tolerance was lost'
        assert derived.idsp_jump_tol == 0.05, 'The resolution tolerance was lost'
    with pytest.raises(DC3Error, match='must be positive'):
        templates.TemplateLibrary(flux, grid, idsp_jump_tol=0.0)


@pytest.mark.parametrize('nsections,warns', [(5, False), (6, True)])
def test_many_splices_warn(nsections, warns):
    """
    More than 5 sampling segments warns that the tolerance is probably too tight.

    Spliced libraries join a handful of sections; many more suggests noise.
    """
    library = make_spliced_library(nsections)
    _, messages = warning_messages(library, make_galaxy(), TemplatePar())
    flagged = [m for m in messages if 'sampling_jump_tol' in m]
    assert len(flagged) == int(warns), \
        f'{nsections} segments should {"" if warns else "not "}warn about sampling_jump_tol'
    if warns:
        assert f'{nsections} segments' in flagged[0], 'The warning should count the segments'


def test_a_run_of_breaks_counts_as_one_splice():
    """
    One splice that flags adjacent boundaries is counted once.

    Borders derived from the centres spread a splice over three boundaries, and
    counting each would trip the warning with only two real splices.
    """
    assert templates._count_jumps(np.array([48, 49, 50, 120, 300, 301])) == 3, \
        'Runs of adjacent breaks should each count as one jump'
    assert templates._count_jumps(np.zeros(0, dtype=int)) == 0, 'No breaks should be no jumps'


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


@pytest.mark.parametrize('ratio', [1, 2, 3, 7])
def test_prepared_sampling_matches_the_galaxy_exactly(ratio):
    """
    The template pixel is the galaxy pixel divided by the ratio, to round-off.

    Only the sampling has to match for the offset between the grids to be a
    constant velocity; the tolerance is set far tighter than ``numpy.isclose``'s
    default, since an error in the pixel size accumulates across the spectrum.
    """
    galaxy = make_galaxy()
    prepared = prepare_quietly(
        make_library(ratio=max(ratio, 2)), galaxy, TemplatePar(velscale_ratio=ratio)
    )
    assert prepared.dloglam * ratio == pytest.approx(galaxy.dloglam, rel=1e-15), \
        f'The prepared pixel is not the galaxy pixel divided by {ratio}'


@pytest.mark.parametrize('ratio', [1, 3])
def test_velocity_offset_follows_the_grids(ratio):
    """The offset is the one the two grids imply, and the grids do differ."""
    galaxy = make_galaxy()
    prepared = prepare_quietly(
        make_library(ratio=max(ratio, 2)), galaxy, TemplatePar(velscale_ratio=ratio)
    )
    expected = sampling.grid_velocity_offset(
        prepared.log10lam0, galaxy.log10lam0, galaxy.dloglam, velscale_ratio=ratio
    )
    assert prepared.velocity_offset(galaxy) == expected, \
        'The offset does not follow from the prepared and galaxy grids'
    assert expected % galaxy.velscale != 0.0, \
        'This test needs grids offset by a fraction of a pixel'


def test_velocity_offset_refuses_a_galaxy_of_different_sampling():
    """A galaxy the templates were not prepared against has no constant offset."""
    prepared = prepare_quietly(make_library(), make_galaxy())
    other = GalaxySpectra(np.ones(GALAXY_NPIX), galaxy_grid(GALAXY_DLOGLAM * 1.01))
    with pytest.raises(DC3Error, match='not related by a constant velocity offset'):
        prepared.velocity_offset(other)


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
    before = galaxy.flux.copy(), galaxy.idsp.copy(), galaxy.grid
    prepare_quietly(make_library(), galaxy)
    assert np.array_equal(galaxy.flux, before[0]), 'Preparation altered the galaxy flux'
    assert np.array_equal(galaxy.idsp, before[1]), 'Preparation altered the galaxy resolution'
    assert galaxy.grid is before[2], 'Preparation altered the galaxy wavelength grid'


def test_library_is_not_modified():
    """Preparation returns a new set; the raw library is left alone."""
    library = make_library()
    before = library.flux.copy()
    prepare_quietly(library, make_galaxy())
    assert np.array_equal(library.flux, before), 'Preparation altered the raw templates'


def test_linear_library_is_prepared_onto_the_galaxy_sampling():
    """
    A linearly sampled library is prepared directly, resampled once.

    The output is logarithmic at the galaxy's sampling, and the identity the
    scheme rests on -- the prepared resolution is the fiducial offset by
    dvar_inst -- holds exactly as for a logarithmic library.
    """
    galaxy = make_galaxy()
    library = make_linear_library()
    assert library.grid.kind == 'linear', 'This test needs a linearly sampled library'
    prepared = prepare_quietly(library, galaxy)

    assert prepared.grid.is_log, 'The prepared templates are not logarithmically sampled'
    assert prepared.dloglam == pytest.approx(galaxy.dloglam, rel=1e-15), \
        'The prepared sampling does not match the galaxy'
    fiducial = np.interp(prepared.wave, galaxy.wave, galaxy.fiducial_resolution())
    expected = np.sqrt(np.square(fiducial) - prepared.dvar_inst)
    inside = (prepared.wave >= galaxy.wave[0]) & (prepared.wave <= galaxy.wave[-1])
    assert np.allclose(prepared.idsp[0][inside], expected[inside], rtol=1e-6), \
        'The prepared resolution of a linear library is not the fiducial offset by dvar_inst'


def test_linear_library_minimum_kernel_is_set_by_the_narrowest_pixel():
    """
    On a linear grid the minimum kernel is epsilon_sigma in pixels, not km/s.

    Pixels narrow in velocity with wavelength on a linear grid, so the same
    kernel in km/s is widest in pixels at the red end.  The offset must leave
    every pixel's kernel at least epsilon_sigma pixels wide, reaching it
    exactly at one pixel.
    """
    library = make_linear_library()
    prepared = prepare_quietly(library, make_galaxy())
    kernel_pixels = prepared.match.kernel_sigma_pixels
    assert np.all(kernel_pixels >= prepared.match.epsilon_sigma * (1 - 1e-10)), \
        'Some pixel was given a kernel narrower than epsilon_sigma pixels'
    assert np.isclose(np.amin(kernel_pixels), prepared.match.epsilon_sigma), \
        'The narrowest kernel should be exactly epsilon_sigma pixels'


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
    expected = resample.Resample(
        library.flux, xBorders=library.grid.borders, inLog=True,
        newRange=[library.wave[0], library.wave[-1]], newdx=galaxy.dloglam, newLog=True
    )
    assert np.allclose(prepared.flux, np.atleast_2d(expected.outy)), \
        'Unmatched templates should be resampled without any convolution'


def test_unmatched_preparation_carries_the_template_dispersion_if_any():
    """
    The prepared dispersion is the templates' own, when they have one.

    With no matching the templates keep their native resolution, so that is
    what the prepared set should report -- and nothing, if they had none.
    With the excess corrected for, the resampling still broadens them, and
    the dispersion reported includes it: by at most a quarter of a native
    pixel squared in variance.
    """
    library, galaxy = make_library(idsp=20.0), make_galaxy(with_idsp=False)
    uncorrected = prepare_quietly(library, galaxy, TemplatePar(correct_lsf_excess=False))
    assert np.allclose(uncorrected.idsp, 20.0), \
        'Unmatched templates should keep their own instrumental dispersion'
    corrected = prepare_quietly(library, galaxy)
    widest = np.sqrt(20.0 ** 2 + np.amax(library.pixel_velocity) ** 2 / 4)
    assert np.all(corrected.idsp > 20.0) and np.all(corrected.idsp <= widest), \
        'Unmatched templates should report their own dispersion broadened by the resampling'

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
        ('convolution_mask_growth', 4.0),
        ('sampling_jump_tol', 0.05),
        ('idsp_jump_tol', 0.05),
        ('correct_lsf_excess', False),
        ('resample_excess_method', 'expectation'),
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


# ----------------------------------------------------------------------
# Jumps in the library: the Step 2 helpers
# ----------------------------------------------------------------------
# Ten native pixels of unit width, with a break at pixel 5 (border 5.0 lies
# between pixels 4 and 5).  Positions only matter relative to one another.
NATIVE_BORDERS = np.arange(11, dtype=float)
NATIVE_WAVE = NATIVE_BORDERS[:-1] + 0.5
BREAK = np.array([5])


def test_straddling_pixel_takes_the_majority_segment_dispersion():
    """
    A pixel straddling a break takes its dispersion from the segment covering more of it.

    Interpolating across the break would give it a value neither side has.
    """
    values = np.where(np.arange(10) < 5, 1.0, 2.0)
    # A pixel mostly below the break, one mostly above, and one clear of it
    out_borders = np.array([4.5, 5.3, 5.6, 6.6])
    out_wave = np.array([4.9, 5.1, 6.1])
    _, majority = templates._segment_overlap(out_borders, NATIVE_BORDERS, BREAK)
    out = templates._interp_within_segments(out_wave, NATIVE_WAVE, values, BREAK, majority)
    assert np.array_equal(out, [1.0, 2.0, 2.0]), \
        'A straddling pixel did not take the value of the segment covering most of it'

    smooth = templates._interp_within_segments(
        out_wave, NATIVE_WAVE, values, np.zeros(0, dtype=int), np.zeros(3, dtype=int)
    )
    assert 1.0 < smooth[0] < 2.0, \
        'Without a break the value should be interpolated, which is what the break prevents'


def test_straddling_pixel_is_unmatched_if_either_side_is():
    """A pixel straddling a break carries UNMATCHED from either side."""
    unmatched = np.arange(10) >= 5
    out_borders = np.array([3.5, 4.5, 5.3, 6.3])
    out_wave = np.array([4.0, 4.9, 5.8])
    overlap, majority = templates._segment_overlap(out_borders, NATIVE_BORDERS, BREAK)
    assert majority[1] == 0, 'This test needs the straddling pixel to lie mostly below the break'
    out = templates._flag_within_segments(out_wave, NATIVE_WAVE, unmatched, BREAK, overlap)
    assert np.array_equal(out, [False, True, True]), \
        'The straddling pixel should be unmatched because the segment above it is'


def test_break_is_grown_by_the_wider_kernel():
    """
    A break is grown by convolution_mask_growth times the larger kernel on each side.

    With kernels of 0.5 and 1.0 native pixels either side of the break and a
    growth of 2, the band is two native pixels each side of border 5.
    """
    kernel = np.where(np.arange(10) < 5, 0.5, 1.0)
    out_borders = np.arange(0.0, 10.5, 0.5)
    flagged = templates._grow_regions(out_borders, NATIVE_BORDERS, BREAK, BREAK, kernel, 2.0)
    centres = (out_borders[:-1] + out_borders[1:]) / 2
    assert np.array_equal(centres[flagged], [3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75]), \
        'The band should span border 5 plus and minus two native pixels'


def test_growth_is_one_pixel_without_matching():
    """
    With no matching there is no kernel, and a break is grown by one native pixel each side.

    The minimum still matters: it catches an output pixel straddling the break.
    """
    out_borders = np.arange(0.0, 10.5, 0.5)
    flagged = templates._grow_regions(out_borders, NATIVE_BORDERS, BREAK, BREAK, None, 3.0)
    centres = (out_borders[:-1] + out_borders[1:]) / 2
    assert np.array_equal(centres[flagged], [4.25, 4.75, 5.25, 5.75]), \
        'Without a kernel the band should be one native pixel either side of the break'
    narrow = templates._grow_regions(
        out_borders, NATIVE_BORDERS, BREAK, BREAK, np.full(10, 0.1), 3.0
    )
    assert np.array_equal(narrow, flagged), \
        'A band narrower than one native pixel should be widened to one'


def test_masked_run_is_grown_by_the_widest_kernel_it_touches():
    """
    A run of masked pixels is grown by the widest kernel across it and its neighbours.

    Pixels 4 and 5 are masked.  The widest kernel among pixels 3 to 6 is 1.0
    native pixel, at pixel 6, so with a growth of 2 the grown run spans borders
    2 to 8.
    """
    gpm = np.ones(10, dtype=bool)
    gpm[4:6] = False
    starts, ends = templates._masked_runs(gpm)
    assert np.array_equal(starts, [4]) and np.array_equal(ends, [6]), \
        'Pixels 4 and 5 should form one run, bounded by borders 4 and 6'
    kernel = np.where(np.arange(10) < 6, 0.5, 1.0)
    out_borders = np.arange(0.0, 10.5, 0.5)
    flagged = templates._grow_regions(out_borders, NATIVE_BORDERS, starts, ends, kernel, 2.0)
    centres = (out_borders[:-1] + out_borders[1:]) / 2
    assert np.array_equal(centres[flagged], np.arange(2.25, 8.0, 0.5)), \
        'The masked run should be grown to borders 2 to 8'


def test_masked_runs_are_found_at_the_ends_and_apart():
    """Runs touching either end of the spectrum, and separate runs, are each found."""
    gpm = np.array([False, True, True, False, False, True, True, True, True, False])
    starts, ends = templates._masked_runs(gpm)
    assert np.array_equal(starts, [0, 3, 9]) and np.array_equal(ends, [1, 5, 10]), \
        'Each run of masked pixels should be found, including those at the ends'


# ----------------------------------------------------------------------
# Jumps in the library: the pipeline
# ----------------------------------------------------------------------
def stepped_library(step_wave=4020.0):
    """A regular library whose resolution steps from 20 to 25 km/s inside the galaxy range."""
    library = make_library()
    idsp = np.where(library.wave < step_wave, 20.0, 25.0)
    return templates.TemplateLibrary(library.flux, library.grid, key='STEPPED', idsp=idsp)


def flagged_wave(prepared, flag):
    """The wavelengths of the prepared pixels carrying a flag."""
    return prepared.wave[getattr(prepared.mask, flag)[0]]


def test_one_segment_library_sets_neither_jump_flag():
    """
    A library without jumps is prepared as before, and nothing is flagged.

    With one segment, interpolation within segments is ordinary interpolation.
    """
    galaxy, library = make_galaxy(), make_library()
    prepared = prepare_quietly(library, galaxy, TemplatePar(correct_lsf_excess=False))
    assert not np.any(prepared.mask.SAMP_JUMP), 'A regular library set SAMP_JUMP'
    assert not np.any(prepared.mask.RES_JUMP), 'A constant resolution set RES_JUMP'
    assert not np.any(prepared.mask.TPL_MASKED), 'A library with no masked pixels set TPL_MASKED'
    fiducial = np.interp(library.wave, galaxy.wave, galaxy.fiducial_resolution())
    expected = np.interp(prepared.wave, library.wave, np.sqrt(fiducial ** 2 - prepared.dvar_inst))
    # Round-off only: the output centres are recomputed from the logarithmic grid
    assert np.allclose(prepared.idsp[0], expected, rtol=1e-12, atol=0.0), \
        'With one segment the prepared dispersion should be interpolated as before'


@pytest.mark.parametrize('method', ['expectation', 'local'])
def test_corrected_preparation_reports_the_target_where_matched(method):
    """
    With the excess corrected for, the dispersion reported is the target less dvar_inst.

    The correction chooses the kernel so that both steps together reach the
    target, and it is the target that is reported.  The agreement is to the
    accuracy of the table that inverts the convolution's variance, a part in
    a million.
    """
    galaxy, library = make_galaxy(), make_library()
    prepared = prepare_quietly(library, galaxy, TemplatePar(resample_excess_method=method))
    assert prepared.match.n_unmatched == 0, 'This test needs every pixel to be matched'
    fiducial = np.interp(library.wave, galaxy.wave, galaxy.fiducial_resolution())
    expected = np.interp(prepared.wave, library.wave, np.sqrt(fiducial ** 2 - prepared.dvar_inst))
    assert np.allclose(prepared.idsp[0], expected, rtol=1e-6, atol=0.0), \
        'The corrected preparation should report the target, offset by dvar_inst'


def test_resample_variance_follows_the_phase_of_the_output_borders():
    r"""
    The local prediction is exact where the grids align; the expectation is not.

    The output grid starts with a pixel centred on the first native pixel.
    For a logarithmic library at half the galaxy's pixel, a ratio of 2 makes
    the output pixels the native ones, every border falling on a native
    border: no excess.  A ratio of 1 makes each output pixel two native ones,
    every border falling midway through a native pixel: :math:`\phi = 1/2`, and
    an excess of a quarter of a native pixel squared.  The expectation is a
    sixth in both.
    """
    library, galaxy = make_library(ratio=2), make_galaxy()
    pixel2 = np.square(library.pixel_velocity)
    idsp = np.full(library.npix, 20.0)
    for ratio, expected in [(2, 0.0), (1, 0.25)]:
        out_borders = templates._resample(
            np.zeros(library.npix), library, galaxy, ratio
        ).outborders
        out_velscale = galaxy.velscale / ratio
        local = templates._resample_variance(library, out_borders, out_velscale, idsp, 'local')
        assert np.allclose(local, expected * pixel2, rtol=0.0, atol=1e-9 * np.amax(pixel2)), \
            f'At velscale_ratio = {ratio} the local excess should be {expected} native pixel^2'
        mean = templates._resample_variance(
            library, out_borders, out_velscale, idsp, 'expectation'
        )
        assert np.allclose(mean, pixel2 / 6), \
            'The expected excess should be a sixth of the native pixel squared'


def test_correction_narrows_the_kernel():
    """
    The corrected kernel is narrower than the uncorrected one wherever it is above the floor.

    Both steps add variance of their own, which the kernel need not supply.
    """
    galaxy, library = make_galaxy(), make_library()
    corrected = prepare_quietly(library, galaxy)
    uncorrected = prepare_quietly(library, galaxy, TemplatePar(correct_lsf_excess=False))
    above = uncorrected.match.kernel_sigma_pixels > 0.2
    assert np.all(corrected.match.kernel_sigma[above] < uncorrected.match.kernel_sigma[above]), \
        'The corrected kernel should be narrower than the uncorrected one'


def test_sampling_splice_sets_only_samp_jump():
    """A splice in the sampling is guarded by SAMP_JUMP alone, around the splice."""
    grid, starts = spliced_grid(2)
    library = make_spliced_library(2)
    prepared = prepare_quietly(library, make_galaxy())
    assert not np.any(prepared.mask.RES_JUMP), 'A splice in the sampling alone set RES_JUMP'
    flagged = flagged_wave(prepared, 'SAMP_JUMP')
    splice = grid.borders[starts[0]]
    assert flagged.size > 0, 'The splice was not guarded'
    assert flagged[0] < splice < flagged[-1], 'The guard band does not contain the splice'
    assert np.all(np.diff(np.flatnonzero(prepared.mask.SAMP_JUMP[0])) == 1), \
        'A single splice should give one contiguous band'


def test_resolution_step_sets_only_res_jump():
    """A jump in the resolution of a regular library is guarded by RES_JUMP alone."""
    prepared = prepare_quietly(stepped_library(), make_galaxy())
    assert not np.any(prepared.mask.SAMP_JUMP), 'A regular library set SAMP_JUMP'
    flagged = flagged_wave(prepared, 'RES_JUMP')
    assert flagged.size > 0, 'The resolution step was not guarded'
    assert flagged[0] < 4020.0 < flagged[-1], 'The guard band does not contain the step'


def test_a_join_in_both_sets_both_flags():
    """A splice where the resolution also jumps carries both flags."""
    grid, starts = spliced_grid(2)
    idsp = np.where(np.arange(grid.npix) < starts[0], 20.0, 25.0)
    library = templates.TemplateLibrary(library_flux(grid.wave, 2), grid, idsp=idsp)
    prepared = prepare_quietly(library, make_galaxy())
    assert np.any(prepared.mask.SAMP_JUMP & prepared.mask.RES_JUMP), \
        'A join that jumps in both sampling and resolution should set both flags'


def library_with_masked_gap(gap=(4019.0, 4021.0)):
    """
    The default library, and a copy whose first template has a masked gap of zeros.

    The gap is a run of pixels set to zero and flagged as the user's, as a gap
    between spliced sections padded with zeros would be; the second template
    is untouched.
    """
    library = make_library()
    flux = library.flux.copy()
    mask = np.zeros(flux.shape, dtype=bool)
    gap = (library.wave > gap[0]) & (library.wave < gap[1])
    flux[0, gap] = 0.0
    mask[0, gap] = True
    gapped = templates.TemplateLibrary(
        flux, library.grid, key='GAPPED', mask=mask, idsp=library.idsp
    )
    return library, gapped


def test_masked_library_pixels_set_tpl_masked_in_their_template():
    """
    A masked run in one template is grown and flagged TPL_MASKED in that template alone.

    Neither jump flag is set, since the sampling and resolution are regular.
    """
    _, gapped = library_with_masked_gap()
    prepared = prepare_quietly(gapped, make_galaxy())
    flagged = prepared.wave[prepared.mask.TPL_MASKED[0]]
    assert flagged.size > 0, 'The masked gap was not flagged'
    assert flagged[0] < 4019.0 and flagged[-1] > 4021.0, \
        'The flagged region should contain the whole masked gap'
    assert np.all(np.diff(np.flatnonzero(prepared.mask.TPL_MASKED[0])) == 1), \
        'One masked gap should give one contiguous flagged region'
    assert not np.any(prepared.mask.TPL_MASKED[1]), \
        'A template with no masked pixels should not be flagged'
    assert not np.any(prepared.mask.SAMP_JUMP | prepared.mask.RES_JUMP), \
        'A masked gap in a regular library should set neither jump flag'


def test_masked_gap_affects_the_prepared_flux_only_within_the_flagged_region():
    """
    Zeros in a masked gap are convolved in, but their effect is confined to TPL_MASKED.

    The prepared template is compared with one prepared from the same library
    without the gap.  At the default growth of three kernel dispersions, the
    gap's influence beyond the flagged region is the Gaussian tail, below a
    part in a thousand of the flux.
    """
    library, gapped = library_with_masked_gap()
    galaxy = make_galaxy()
    clean = prepare_quietly(library, galaxy)
    prepared = prepare_quietly(gapped, galaxy)
    # Only where the clean preparation has data, since NODATA pixels are zero
    data = clean.gpm[0]
    change = np.absolute(prepared.flux[0, data] / clean.flux[0, data] - 1)
    inside = prepared.mask.TPL_MASKED[0, data]
    assert np.amax(change[inside]) > 0.5, \
        'This test needs the gap to change the prepared flux substantially'
    assert np.amax(change[~inside]) < 1e-3, \
        f'Outside the flagged region the gap changed the flux by {np.amax(change[~inside]):.2e}'


def test_guard_band_follows_the_kernel():
    """
    The band scales with convolution_mask_growth and the kernel, and is narrow without matching.

    The prepared band at a resolution step spans about 2
    convolution_mask_growth kernel dispersions; doubling convolution_mask_growth
    doubles it, and with no galaxy resolution to match there is no kernel and
    only the one-pixel minimum remains.
    """
    library, galaxy = stepped_library(), make_galaxy()
    narrow = prepare_quietly(library, galaxy, TemplatePar(convolution_mask_growth=3.0))
    wide = prepare_quietly(library, galaxy, TemplatePar(convolution_mask_growth=6.0))
    n_narrow, n_wide = np.sum(narrow.mask.RES_JUMP[0]), np.sum(wide.mask.RES_JUMP[0])
    assert abs(n_wide - 2 * n_narrow) <= 2, \
        'Doubling convolution_mask_growth should double the band, to within the output ' \
        'pixels it cuts'

    # The larger kernel at the step, in native pixels, sets the band
    step = library.idsp_breaks[0]
    kernel = narrow.match.kernel_sigma_pixels
    sigma = max(kernel[step - 1], kernel[step])
    expected = 2 * 3.0 * sigma * library.pixel_velocity[step] / narrow.velscale
    assert abs(n_narrow - expected) <= 2, \
        f'The band covers {n_narrow} prepared pixels, not the {expected:.1f} its kernel implies'

    unmatched = prepare_quietly(library, make_galaxy(with_idsp=False))
    assert 0 < np.sum(unmatched.mask.RES_JUMP[0]) <= 2, \
        'Without matching, only the output pixels within a native pixel of the step are guarded'


def test_spliced_library_is_unaffected_away_from_the_splice():
    """
    Away from its splice, a spliced library is prepared as a regular one is.

    The red section of a two-section library is sampled at 0.03 A, so beyond
    the guard band it must give the same prepared spectrum as a library sampled
    at 0.03 A throughout.  This is what makes the guard band, rather than any
    wider treatment, sufficient.
    """
    galaxy = make_galaxy()
    spliced = make_spliced_library(2)
    npix = int(np.round(250.0 / 0.03))
    regular_grid = sampling.SpectralGrid.from_linear_spacing(3900.015, 0.03, npix)
    # The same function of wavelength on both grids
    regular_flux = 1.0 + 0.3 * np.sin((regular_grid.wave - spliced.wave[0]) / 7.0)
    regular = templates.TemplateLibrary(
        np.vstack([regular_flux, regular_flux * 1.1]), regular_grid, idsp=np.full(npix, 20.0)
    )
    a = prepare_quietly(spliced, galaxy)
    b = prepare_quietly(regular, galaxy)

    away = (a.wave > 4040.0) & (a.wave < 4130.0)
    assert not np.any(a.mask.SAMP_JUMP[0][away]), 'This region should be clear of the guard band'
    other = np.interp(a.wave[away], b.wave, b.flux[0])
    assert np.allclose(a.flux[0][away], other, rtol=1e-4), \
        'Away from the splice, the spliced library was prepared differently from a regular one'
