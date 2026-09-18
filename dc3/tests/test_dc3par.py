"""
Tests for the dc3 parameter sets.

These check the declarations themselves -- that every parameter is documented,
that the cross-parameter rules fire, and that the whole set survives the round
trips -- rather than the behaviour of the base class, which
``test_parset.py`` covers.
"""

import inspect
import re
import tomllib

from numpydoc.docscrape import FunctionDoc
import pytest
from pydantic import ValidationError

from dc3 import templates
from dc3.par import dc3par
from dc3.par.dc3par import ContinuumPar, DC3Par, FitPar, TemplatePar
from dc3.par.parset import ParSet

from .test_parset import check_declaration


ALL_PARSETS = [getattr(dc3par, name) for name in dc3par.__all__]

# The parameter sets that are expanded over a function's keywords, paired with
# the function they configure.  A set listed here must agree with its function
# name for name and default for default; see test_parset_matches_its_function.
EXPANDED_PARSETS = {
    TemplatePar: templates.prepare,
}

# The parameter sets that are NOT expanded over a function, with the reason.
# Every dc3 parameter set must appear either here or in EXPANDED_PARSETS, so
# that adding one is a deliberate decision about which it is.
UNEXPANDED_PARSETS = {
    dc3par.TemplateLibraryPar: 'Declares a library on disk; its reader is not yet written.',
    dc3par.ConvolvePar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.CorrelatePar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.MaskPar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.WindowPar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.ContinuumPar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.FitPar: 'Consumed inside the fit, not at a single function boundary.',
    dc3par.QAPar: 'Read by the plotting tier, which is not yet written.',
    dc3par.DC3Par: 'The top-level set; it nests the others and is never expanded.',
}


# ----------------------------------------------------------------------
# Declaration
# ----------------------------------------------------------------------
@pytest.mark.parametrize('cls', ALL_PARSETS, ids=[c.__name__ for c in ALL_PARSETS])
def test_declaration(cls):
    """Every dc3 parameter set is completely declared."""
    check_declaration(cls)


@pytest.mark.parametrize('cls', ALL_PARSETS, ids=[c.__name__ for c in ALL_PARSETS])
def test_metadata(cls):
    """Every dc3 parameter set can be written to a config file and a header."""
    assert issubclass(cls, ParSet), f'{cls.__name__} is not a ParSet'
    assert cls.default_key is not None, \
        f'{cls.__name__} has no default_key, so it cannot name its configuration section'
    assert cls.card_prefix is not None, \
        f'{cls.__name__} has no card_prefix, so it cannot be written to a FITS header'


def test_card_prefixes_are_unique():
    """Two parameter sets sharing a prefix would overwrite each other in a header."""
    prefixes = [cls.card_prefix for cls in ALL_PARSETS]
    assert len(set(prefixes)) == len(prefixes), \
        f'Duplicate card_prefix among the dc3 parameter sets: {sorted(prefixes)}'


def test_section_keys_are_unique():
    """Two parameter sets sharing a key would collide in a configuration file."""
    keys = [cls.default_key for cls in ALL_PARSETS]
    assert len(set(keys)) == len(keys), \
        f'Duplicate default_key among the dc3 parameter sets: {sorted(keys)}'


def test_every_parset_is_reachable_from_the_top():
    """
    Each parameter set is nested in DC3Par.

    A set that is declared but not reachable would never be read from a
    configuration file or written to a header, so it would silently keep its
    defaults.
    """
    nested = {DC3Par.model_fields[key].annotation for key in DC3Par.nested()}
    for cls in ALL_PARSETS:
        if cls is DC3Par:
            continue
        assert cls in nested, \
            f'{cls.__name__} is declared but not nested in DC3Par, so it can never be configured'


# ----------------------------------------------------------------------
# Agreement between a parameter set and the function it configures
# ----------------------------------------------------------------------
def test_every_parset_is_classified():
    """
    Each parameter set is either expanded over a function or exempted.

    The two tables below are what keeps the agreement check honest: without
    this, a new parameter set could be added to a function's signature and the
    check would simply not run on it.
    """
    classified = set(EXPANDED_PARSETS) | set(UNEXPANDED_PARSETS)
    for cls in ALL_PARSETS:
        assert cls in classified, \
            f'{cls.__name__} appears in neither EXPANDED_PARSETS nor UNEXPANDED_PARSETS.  ' \
            'Decide whether it is expanded over a function and record it in the right one.'
    assert len(set(EXPANDED_PARSETS) & set(UNEXPANDED_PARSETS)) == 0, \
        'A parameter set is listed as both expanded and unexpanded'


@pytest.mark.parametrize(
    'cls,func', list(EXPANDED_PARSETS.items()),
    ids=[c.__name__ for c in EXPANDED_PARSETS]
)
def test_parset_matches_its_function(cls, func):
    """
    An expanded parameter set agrees with its function, key for key.

    The call is ``func(..., **par.to_kwargs())``, so a parameter the function
    does not accept raises a TypeError at run time and a keyword the parameter
    set does not declare is silently unreachable from a configuration file.
    Both are caught here instead.  The defaults must agree too, since otherwise
    calling the function directly and calling it through the parameter set
    would do different things.
    """
    check_declaration(cls, flat=True)

    signature = inspect.signature(func)
    accepted = {
        name: p for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    assert set(cls.keys()) == set(accepted), (
        f'{cls.__name__} and {func.__name__} disagree: '
        f'only in the parameter set: {sorted(set(cls.keys()) - set(accepted))}; '
        f'only in the signature: {sorted(set(accepted) - set(cls.keys()))}'
    )
    for key, value in cls().to_kwargs().items():
        assert accepted[key].default == value, \
            f'{cls.__name__}.{key} defaults to {value!r} but {func.__name__} defaults to ' \
            f'{accepted[key].default!r}; the two calling routes would not agree'


@pytest.mark.parametrize(
    'cls,func', list(EXPANDED_PARSETS.items()),
    ids=[c.__name__ for c in EXPANDED_PARSETS]
)
def test_parset_matches_its_docstring(cls, func):
    """
    The function's documented parameters are the parameter set's, verbatim.

    :func:`~dc3.par.parset.document_parameters` generates the Parameters
    entries from the parameter set, so agreement holds by construction -- but
    only for as long as the decorator is applied and its placeholder is in the
    docstring.  Dropping either would leave the entries hand-written and free to
    drift, silently, which is exactly what the arrangement exists to prevent.
    This is the check that notices.

    It compares the rendered text rather than calling the generator, so that it
    fails rather than agreeing with itself if the section is hand-written again.
    """
    documented = {p.name: p for p in FunctionDoc(func)['Parameters']}
    signature = inspect.signature(func)
    assert set(documented) == set(signature.parameters), (
        f'{func.__name__} and its docstring disagree: '
        f'undocumented: {sorted(set(signature.parameters) - set(documented))}; '
        f'documented but not in the signature: '
        f'{sorted(set(documented) - set(signature.parameters))}'
    )

    for key, f in cls.model_fields.items():
        entry = documented[key]
        assert entry.type.endswith(', optional'), \
            f'{func.__name__} does not document {key} as optional, though it has a default'
        rendered = re.sub(r'\s+', ' ', ' '.join(entry.desc)).strip()
        expected = re.sub(r'\s+', ' ', f.description).strip()
        assert rendered == expected, (
            f'The docstring entry for {func.__name__}({key}=) is not the description declared '
            f'by {cls.__name__}.  It should be generated from the parameter set, not written '
            f'out.\n  docstring: {rendered}\n  parameter set: {expected}'
        )


# ----------------------------------------------------------------------
# Values carried over from the published algorithm
# ----------------------------------------------------------------------
def test_published_defaults():
    """
    The defaults that encode published or empirical choices are what they were.

    These are not arbitrary: each was either published, or justified empirically
    in the original development notes, so a change to one is a change to the
    algorithm and should be deliberate.
    """
    par = DC3Par()
    assert par.fit.x_scale == [1.0, 100.0, 100.0], \
        'Parameter scales differ from the characteristic scales given in the paper'
    assert par.convolve.min_sigma == 0.85, \
        'Sub-Nyquist convolution floor differs from 0.85 px, i.e. a 2 px FWHM'
    assert par.convolve.max_block == 8, 'Block-replication limit differs from the published value'
    assert par.window.nfwhm == 2.0, 'Fit-window width differs from the value used in the paper'
    assert par.correlate.apodization == 'none', \
        'Apodization default differs from the empirically justified choice of none'
    assert par.correlate.length_factor == 2.2, 'Transform length factor differs from 2.2'
    assert par.mask.grow_sigma == 2.0, 'Mask grow/shrink factor differs from the published value'


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


# ----------------------------------------------------------------------
# Cross-parameter rules
# ----------------------------------------------------------------------
def test_gauss_hermite_requires_matched_resolution():
    """
    Higher moments are rejected when a pedestal is permitted.

    The Gauss-Hermite parameterization is defined relative to the fitted
    Gaussian, so the higher moments mean nothing with an instrumental offset.
    This must fail rather than warn: a warning produces numbers that look fine
    and are not.
    """
    with pytest.raises(ValidationError, match='matched template resolution'):
        DC3Par(fit={'moments': 4}, template={'sigma_floor': 5.0})
    assert DC3Par(fit={'moments': 4}).fit.moments == 4, \
        'Higher moments were rejected even with matched resolution'
    assert DC3Par(template={'sigma_floor': 5.0}).template.sigma_floor == 5.0, \
        'A pedestal was rejected even when fitting only two moments'


def test_continuum_order_and_iterations_must_agree():
    """
    Zero in either continuum parameter disables the continuum, so both must be.

    The C++ implementation silently forced one to zero when the other was, which
    let a configuration assert a continuum order that was never used.
    """
    with pytest.raises(ValidationError, match='zero together'):
        ContinuumPar(order=4, iterations=0)
    with pytest.raises(ValidationError, match='zero together'):
        ContinuumPar(order=0, iterations=10)
    assert ContinuumPar(order=4, iterations=10).order == 4, \
        'A consistent continuum configuration was rejected'
    assert ContinuumPar().order == 0, 'The continuum is not disabled by default'


def test_mask_vdiff_unit_is_explicit():
    """
    The unit of mask_vdiff is its own parameter, not the sign of the value.

    The C++ implementation used a negative mask_vdiff to mean "in pixels", which
    made the value unreadable without knowing the convention.
    """
    assert FitPar().mask_vdiff_unit == 'km/s', 'Default unit for mask_vdiff changed'
    assert FitPar(mask_vdiff=0.1, mask_vdiff_unit='pixel').mask_vdiff == 0.1, \
        'A pixel-valued convergence criterion was not accepted'
    with pytest.raises(ValidationError):
        FitPar(mask_vdiff=-1.0)


def test_mask_iterations_null_means_iterate_to_convergence():
    """
    Iterating to convergence is expressed as null, not as a negative count.

    The C++ implementation encoded it as a negative iteration count.
    """
    assert FitPar().mask_iterations == 0, 'Mask iteration default changed'
    assert FitPar(mask_iterations=None).mask_iterations is None, \
        'Null was not accepted as "iterate until converged"'
    with pytest.raises(ValidationError):
        FitPar(mask_iterations=-1)


# ----------------------------------------------------------------------
# Round trips
# ----------------------------------------------------------------------
def test_toml_round_trip(tmp_path):
    """The whole parameter set survives a round trip through a TOML file."""
    par = DC3Par(
        template={'velscale_ratio': 4},
        fit={'method': 'de', 'restarts': 2},
        continuum={'order': 4, 'iterations': 10},
    )
    f = tmp_path / 'dc3.toml'
    par.to_toml(cfg_file=f)
    assert DC3Par.from_toml(f) == par, \
        'The dc3 parameter set did not survive a round trip through a TOML file'


def test_default_toml_is_valid_and_complete():
    """The emitted default configuration parses and contains every subsection."""
    content = DC3Par().to_toml()
    doc = tomllib.loads(content)['dc3']
    for key in DC3Par.nested():
        assert key in doc, f'Subsection {key} is missing from the emitted configuration'


def test_each_section_is_described_exactly_once():
    """
    A nested section gets one comment, not two.

    The parent's field description and the child's default_comment both describe
    the same section, so the more specific one is used and the other suppressed.
    """
    content = DC3Par().to_toml()
    assert content.count('# Template preparation, run once per execution.') == 1, \
        "The subsection's own comment is missing or duplicated"
    assert '# Template preparation.\n' not in content, \
        'The parent description was emitted alongside the subsection comment'
    # A subsection with no default_comment falls back to the parent's description
    assert '# Quality-assessment plots.' in content, \
        'A subsection without its own comment lost the parent description'


def test_header_round_trip():
    """The whole parameter set survives a round trip through a FITS header."""
    par = DC3Par(template={'velscale_ratio': 4}, fit={'method': 'mcmc'})
    assert DC3Par.from_header(par.to_header()) == par, \
        'The dc3 parameter set did not survive a FITS header round trip'


def test_rst_table_covers_every_parset():
    """The generated documentation reaches every nested parameter set."""
    content = '\n'.join(DC3Par.to_rst_table())
    for cls in ALL_PARSETS:
        marker = f'Class Instantiation: :class:`~{cls.__module__}.{cls.__name__}`'
        assert content.count(marker) == 1, \
            f'{cls.__name__} is documented {content.count(marker)} times; expected exactly once'
