"""
Tests for :class:`~dc3.par.parset.ParSet`.

This module also holds :func:`check_declaration`, the coding-time check that
every :class:`~dc3.par.parset.ParSet` subclass is declared completely.  It lives
here rather than on the class because nothing a *user* does can violate it: it
constrains how a parameter set is written, so it belongs with the tests that
enforce it.
"""

import importlib
import pkgutil
import tomllib
import warnings
from pathlib import Path
from typing import Annotated, Literal

from astropy.io import fits
from ppxf import ppxf_util
from pydantic import Field, ValidationError, model_validator
import pytest

import dc3
from dc3.par.funcpar import FuncPar
from dc3.par.parset import (
    ParSet, _is_parset, _plain_reference, document_parameters, parameter_docstring
)
from dc3.pkg.exceptions import DC3CodingError, DC3ParameterError


# ----------------------------------------------------------------------
# Declaration check
# ----------------------------------------------------------------------
def check_declaration(cls, flat=False):
    """
    Check that a :class:`~dc3.par.parset.ParSet` subclass is declared completely.

    Three things are required.  Every parameter must carry a description, since
    the generated documentation and the commented TOML output are built from it.
    A parameter set that declares a ``card_prefix`` must keep it short enough to
    leave room for the suffix character in an 8-character FITS keyword.  And a
    :class:`~dc3.par.funcpar.FuncPar` must declare an ``api_doc``, because its
    generated descriptions say nothing beyond naming the wrapped function, so
    the pointer upstream is the only documentation its users get.

    Parameters
    ----------
    cls : type
        The :class:`~dc3.par.parset.ParSet` subclass to check.
    flat : bool, optional
        Additionally require that no parameter is itself a parameter set.  This
        is asked of any set that is expanded over a function's keywords with
        :func:`~dc3.par.parset.ParSet.to_kwargs`, since a function keyword is
        never a parameter set.  Nesting such a set would break its call site at
        run time; checking it here makes that a declaration error instead.

    Raises
    ------
    DC3CodingError
        Raised if the declaration is incomplete.
    """
    missing = [
        k for k, f in cls.model_fields.items()
        if f.description is None and not _is_parset(f.annotation)
    ]
    if len(missing) > 0:
        raise DC3CodingError(
            f'Parameters of {cls.__name__} are missing descriptions: {missing}.  Every '
            'parameter must be documented where it is declared.'
        )
    if cls.card_prefix is not None and len(cls.card_prefix) > 7:
        raise DC3CodingError(
            f'card_prefix for {cls.__name__} is {len(cls.card_prefix)} characters; a FITS '
            'keyword allows 8, and one is reserved for the suffix.'
        )
    # FuncPar itself is abstract and wraps nothing, so it is exempt.
    if issubclass(cls, FuncPar) and cls.func is not None and cls.api_doc is None:
        raise DC3CodingError(
            f'{cls.__name__} wraps {cls.func.__name__} but declares no api_doc.  A FuncPar '
            'defers to the wrapped function\'s documentation instead of reproducing it, so '
            'that pointer is the only documentation its parameters have.'
        )
    if flat and len(cls.nested()) > 0:
        raise DC3CodingError(
            f'{cls.__name__} is expanded over a function\'s keywords, so it must not contain '
            f'nested parameter sets; it contains {cls.nested()}.'
        )


def all_parset_subclasses(cls=ParSet):
    """
    Recursively collect every :class:`~dc3.par.parset.ParSet` subclass.

    Parameters
    ----------
    cls : type, optional
        The class whose subclasses are collected.

    Returns
    -------
    list
        Every subclass, at any depth.
    """
    subclasses = []
    for sub in cls.__subclasses__():
        subclasses.append(sub)
        subclasses += all_parset_subclasses(sub)
    return subclasses


def package_parset_subclasses():
    """
    Collect every :class:`~dc3.par.parset.ParSet` the package itself declares.

    Every ``dc3`` module is imported first.  ``__subclasses__`` only reports
    classes that have been *defined*, so without that a parameter set would
    escape the checks below simply because nothing had imported its module --
    which, now that each is declared beside the code it configures, is easy to
    arrange by accident.

    :class:`~dc3.par.funcpar.FuncPar` is excluded: it is an abstract base that
    wraps nothing, and the concrete subclasses it exists for are ordinary
    parameter sets that are collected here like any other.

    Returns
    -------
    list
        The parameter sets declared under ``dc3``, excluding test code.

    Raises
    ------
    ImportError
        Raised if any module in the package fails to import.
    """
    for info in pkgutil.walk_packages(dc3.__path__, prefix='dc3.'):
        if '.tests' in info.name:
            continue
        importlib.import_module(info.name)
    return [
        cls for cls in all_parset_subclasses()
        if cls.__module__.startswith('dc3.') and '.tests' not in cls.__module__
        and cls is not FuncPar
    ]


class no_deprecation:
    """
    Context manager asserting that no `DeprecationWarning` is raised.

    Used to keep the item-access shims off pydantic's deprecated instance-level
    ``model_fields``, which is removed in pydantic 3.0.
    """

    def __enter__(self):
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter('error', DeprecationWarning)
        return self

    def __exit__(self, *args):
        return self._ctx.__exit__(*args)


# ----------------------------------------------------------------------
# Example parameter sets used by the tests below
# ----------------------------------------------------------------------
class ExampleTemplatePar(ParSet):
    """Stand-in for the real TemplatePar, used to exercise the base class."""

    default_key = 'template'
    card_prefix = 'XTPL'

    velscale_ratio: Annotated[int, Field(
        default=1, ge=1,
        description='Integer prepared-template pixels per galaxy pixel.'
    )]
    epsilon_sigma: Annotated[float, Field(
        default=0.1, ge=0.1,
        description='Target for the minimum dispersion of the preparation kernel, in pixels.'
    )]
    mask_unmatched_idsp: Annotated[bool, Field(
        default=False,
        description='Mask template regions that cannot reach the target resolution.'
    )]


class ExampleFitPar(ParSet):
    """Stand-in for the real FitPar."""

    default_key = 'fit'
    card_prefix = 'XFIT'

    method: Annotated[Literal['lsq', 'de', 'mcmc'], Field(
        default='lsq', description='Optimizer/sampler backend.'
    )]
    moments: Annotated[int, Field(
        default=2, ge=2, le=6, description='Number of LOSVD moments to fit.'
    )]


class ExampleDC3Par(ParSet):
    """Stand-in for the top-level parameter set, exercising nesting."""

    default_key = 'dc3'
    card_prefix = 'XDC3'

    template: Annotated[ExampleTemplatePar, Field(
        default_factory=ExampleTemplatePar, description='Template preparation parameters.'
    )]
    fit: Annotated[ExampleFitPar, Field(
        default_factory=ExampleFitPar, description='Fitting parameters.'
    )]
    output_dir: Annotated[Path, Field(
        default_factory=Path.cwd, description='Directory for output products.'
    )]

    @model_validator(mode='after')
    def _moments_require_matched_resolution(self):
        """Gauss-Hermite moments are only interpretable at matched resolution."""
        if self.fit.moments > 2 and self.template.epsilon_sigma > 0.5:
            raise ValueError('Gauss-Hermite moments require matched template resolution.')
        return self


# ----------------------------------------------------------------------
# Declaration
# ----------------------------------------------------------------------
def test_package_parsets_are_declared_completely():
    """
    Every ParSet the package ships documents all of its parameters.

    Restricted to what the package declares: this module deliberately defines
    invalid parameter sets to test the checker itself, and some of them are
    created inside test functions, so an unfiltered walk of ``__subclasses__``
    would pick them up or not depending on test execution order.
    """
    for cls in package_parset_subclasses():
        check_declaration(cls)


def test_example_parsets_are_declared_completely():
    """The fixtures below are themselves well formed, so the checker passes."""
    for cls in [ExampleTemplatePar, ExampleFitPar, ExampleDC3Par]:
        check_declaration(cls)


def test_check_declaration_catches_missing_description():
    """A parameter without a description is a coding error."""
    class _Undocumented(ParSet):
        oops: Annotated[int, Field(default=1)]

    with pytest.raises(DC3CodingError, match='missing descriptions'):
        check_declaration(_Undocumented)


def test_check_declaration_catches_funcpar_without_api_doc():
    """A FuncPar that does not point upstream has no documentation at all."""
    class Undocumented(FuncPar):
        func = ppxf_util.varsmooth
        kw_subset = ['oversample']

    with pytest.raises(DC3CodingError, match='declares no api_doc'):
        check_declaration(Undocumented)


def test_check_declaration_catches_long_card_prefix():
    """A card_prefix must leave room for the suffix in an 8-character keyword."""
    class _LongPrefix(ParSet):
        card_prefix = 'TOOLONGX'
        a: Annotated[int, Field(default=1, description='A parameter.')]

    with pytest.raises(DC3CodingError, match='card_prefix'):
        check_declaration(_LongPrefix)


def test_check_declaration_catches_nesting_when_flatness_is_required():
    """A set expanded over a function's keywords must not nest."""
    check_declaration(ExampleTemplatePar, flat=True)
    with pytest.raises(DC3CodingError, match='nested parameter sets'):
        check_declaration(ExampleDC3Par, flat=True)


# ----------------------------------------------------------------------
# Construction and validation
# ----------------------------------------------------------------------
def test_defaults():
    """Defaults resolve, including through nesting and default_factory."""
    p = ExampleDC3Par()
    assert p.template.velscale_ratio == 1, \
        'Nested parameter set did not resolve its declared default'
    assert p.fit.method == 'lsq', 'Nested Literal parameter did not resolve its default'
    assert p.output_dir == Path.cwd(), 'default_factory was not invoked at instantiation'


def test_default_factory_gives_independent_instances():
    """Two instances must not share a nested parameter set."""
    assert ExampleDC3Par().template is not ExampleDC3Par().template, \
        'Two parameter sets share one nested instance; mutating one would alter the other'


@pytest.mark.parametrize(
    'kwargs',
    [
        {'velscale_ratio': 0},          # below the ge bound
        {'epsilon_sigma': 0.05},        # below the varsmooth floor
        {'velscale_ratio': 'two'},      # wrong type
        {'bogus': 1},                   # unknown key
        {'velscale_ratio': None},       # None is not silently allowed
    ]
)
def test_invalid_values_are_rejected(kwargs):
    """Bad values raise rather than being absorbed."""
    with pytest.raises(ValidationError):
        ExampleTemplatePar(**kwargs)


def test_nested_validation_reports_location():
    """A bad value in a nested set is reported with its full path."""
    with pytest.raises(ValidationError) as e:
        ExampleDC3Par(template={'velscale_ratio': 0})
    assert e.value.errors()[0]['loc'] == ('template', 'velscale_ratio'), \
        'Validation error did not report the full path to the offending nested parameter'


def test_literal_options_are_enforced():
    """A value outside the declared Literal set is rejected."""
    with pytest.raises(ValidationError):
        ExampleFitPar(method='amoeba')


def test_cross_parameter_validation():
    """A model_validator can enforce a rule spanning nested parameter sets."""
    with pytest.raises(ValidationError, match='matched template resolution'):
        ExampleDC3Par(fit={'moments': 4}, template={'epsilon_sigma': 1.0})
    # The same moments are fine when the resolution is matched
    assert ExampleDC3Par(fit={'moments': 4}).fit.moments == 4, \
        'Cross-parameter rule rejected a combination it should allow'


def test_assignment_is_validated():
    """validate_assignment means a bad value cannot be set after construction."""
    p = ExampleTemplatePar()
    with pytest.raises(ValidationError):
        p.velscale_ratio = 0


# ----------------------------------------------------------------------
# Item-access shims
# ----------------------------------------------------------------------
def test_item_access():
    """Item access mirrors attribute access, and rejects unknown keys."""
    p = ExampleTemplatePar()
    assert p['velscale_ratio'] == p.velscale_ratio, \
        'Item access returned a different value than attribute access'
    p['velscale_ratio'] = 4
    assert p.velscale_ratio == 4, 'Item assignment did not set the attribute'
    assert len(p) == 3, 'len() did not report the number of declared parameters'
    with pytest.raises(KeyError):
        p['bogus']
    with pytest.raises(KeyError):
        p['bogus'] = 1


def test_item_access_avoids_deprecated_model_fields():
    """
    The item-access shims must not touch ``model_fields`` on the instance.

    Instance access is deprecated in pydantic 2.11 and removed in 3.0, so this
    fails the build rather than emitting a warning that would be lost in the
    output.
    """
    p = ExampleTemplatePar()
    with no_deprecation():
        p['velscale_ratio']
        p['velscale_ratio'] = 2
        len(p)


# ----------------------------------------------------------------------
# Dictionaries, TOML, headers
# ----------------------------------------------------------------------
def test_dict_round_trip():
    """to_dict/from_dict round-trips, including nesting."""
    p = ExampleDC3Par(template={'velscale_ratio': 4}, fit={'method': 'de'})
    assert ExampleDC3Par.from_dict(p.to_dict()) == p, \
        'Parameter set did not survive a dictionary round trip'


def test_to_kwargs_keeps_python_objects():
    """
    to_kwargs returns values as Python objects, where to_dict serializes them.

    The distinction is load-bearing: to_dict must turn a Path into a string so
    it can be written to TOML, but a function expecting a Path must be handed
    one.  Conflating the two would pass a string where a Path was declared.
    """
    class _Flat(ParSet):
        output_dir: Annotated[Path, Field(
            default=Path('/tmp/example'), description='Directory for output products.'
        )]
        velscale_ratio: Annotated[int, Field(default=1, description='A parameter.')]

    p = _Flat()
    assert isinstance(p.to_kwargs()['output_dir'], Path), \
        'to_kwargs serialized a Path, so it could not be passed to a function expecting one'
    assert isinstance(p.to_dict()['output_dir'], str), \
        'to_dict did not serialize a Path, so it could not be written to TOML'
    assert p.to_kwargs()['velscale_ratio'] == 1, 'to_kwargs did not return the parameter value'


def test_parameter_docstring_renders_numpy_style():
    """
    The generated entries carry the type, the options, and the description.

    Options are rendered in place of the type, which is what NumPy style asks
    for and is more informative than reporting ``str``.
    """
    lines = parameter_docstring(ExampleFitPar, indent='    ')
    text = '\n'.join(lines)
    assert "    method : {'lsq', 'de', 'mcmc'}, optional" in lines, \
        'A parameter restricted to a fixed set of values does not report them as its type'
    assert '    moments : int, optional' in lines, \
        'An unrestricted parameter does not report its type'
    for f in ExampleFitPar.model_fields.values():
        assert f.description.split('.')[0] in text, \
            'The rendered entry does not carry the declared description'


def test_document_parameters_fills_the_placeholder():
    """The placeholder is replaced, and the rest of the docstring is untouched."""
    @document_parameters(ExampleFitPar)
    def example(moments=2):
        """
        Summary line.

        Parameters
        ----------
        {parameters}

        Returns
        -------
        None
        """
    assert '{parameters}' not in example.__doc__, 'The placeholder was not replaced'
    assert 'Summary line.' in example.__doc__, 'Replacing the placeholder lost the summary'
    assert 'Returns' in example.__doc__, 'Replacing the placeholder lost a later section'
    assert 'moments : int, optional' in example.__doc__, \
        'The generated entries were not inserted'


def test_document_parameters_requires_the_placeholder():
    """
    A docstring with nowhere to put the parameters is a coding error.

    Failing at import is the point: it is what makes removing the placeholder
    impossible to do quietly.
    """
    with pytest.raises(DC3CodingError, match='exactly once'):
        @document_parameters(ExampleFitPar)
        def _no_placeholder():
            """A docstring with no placeholder."""

    with pytest.raises(DC3CodingError, match='no docstring'):
        @document_parameters(ExampleFitPar)
        def _no_docstring():
            pass


def test_document_parameters_rejects_a_nested_parameter_set():
    """A nested set does not describe a function's keywords."""
    with pytest.raises(DC3ParameterError, match='nested parameter sets'):
        parameter_docstring(ExampleDC3Par)


def test_to_kwargs_rejects_a_nested_parameter_set():
    """
    A nested set cannot be expanded over a function, so it is reported.

    A function keyword is never a parameter set, so silently passing one would
    surface as a confusing failure inside the callee.
    """
    with pytest.raises(DC3ParameterError, match='nested parameter sets'):
        ExampleDC3Par().to_kwargs()


def test_toml_round_trip(tmp_path):
    """to_toml/from_toml round-trips through a real file."""
    p = ExampleDC3Par(template={'velscale_ratio': 4}, fit={'method': 'de'})
    f = tmp_path / 'dc3.toml'
    p.to_toml(cfg_file=f)
    assert ExampleDC3Par.from_toml(f) == p, \
        'Parameter set did not survive a round trip through a TOML file'


def test_toml_is_parseable_and_commented():
    """The emitted TOML parses, and carries the descriptions as comments."""
    p = ExampleDC3Par()
    content = p.to_toml()
    doc = tomllib.loads(content)
    assert set(doc['dc3'].keys()) == {'output_dir', 'template', 'fit'}, \
        'Emitted TOML does not contain exactly the declared top-level parameters'
    assert doc['dc3']['template']['velscale_ratio'] == 1, \
        'Nested parameter did not survive emission as a dotted TOML subsection'
    assert doc['dc3']['template']['mask_unmatched_idsp'] is False, \
        'Boolean was not emitted as a TOML literal; it parsed back as another type'
    assert 'prepared-template pixels' in content, \
        'Parameter description was not emitted as a comment'


def test_toml_exclude_defaults():
    """exclude_defaults omits unchanged parameters."""
    p = ExampleTemplatePar(velscale_ratio=4)
    doc = tomllib.loads(p.to_toml(exclude_defaults=True))
    assert 'velscale_ratio' in doc['template'], \
        'A parameter changed from its default was omitted'
    assert 'epsilon_sigma' not in doc['template'], \
        'A parameter left at its default was emitted despite exclude_defaults'


def test_toml_keeps_none_valued_parameters_discoverable(tmp_path):
    """
    A parameter whose value is None is emitted commented out, not omitted.

    TOML has no null, so such a parameter cannot be written as an assignment.
    Omitting it entirely would hide a knob from the file that is supposed to
    document it, so it is emitted as a comment instead; reading the file back
    leaves it unset.
    """
    class Nullable(ParSet):
        default_key = 'nullable'
        max_nfev: Annotated[int | None, Field(default=None, description='An optional limit.')]
        ftol: Annotated[float, Field(default=1e-8, description='A tolerance.')]

    content = Nullable().to_toml()
    assert '# max_nfev = <unset>' in content, \
        'A None-valued parameter was omitted entirely, making it undiscoverable'
    assert 'An optional limit.' in content, \
        'A None-valued parameter lost its description as well as its value'

    # It is still a comment, so the value round-trips as None
    f = tmp_path / 'nullable.toml'
    f.write_text(content)
    assert Nullable.from_toml(f).max_nfev is None, \
        'The commented placeholder was parsed as a value rather than left unset'


@pytest.mark.parametrize(
    'awkward',
    [
        r'C:\Users\someone\output',       # a Windows path
        r'a\tb',                          # a literal backslash-t, not a tab
        'a "quoted" phrase',              # embedded double quotes
        r'^\s*(\d+)$',                    # a regular expression
        'trailing backslash\\',
    ]
)
def test_toml_escapes_awkward_strings(awkward, tmp_path):
    """
    A string containing a backslash or a quote survives the round trip.

    A TOML basic string treats a backslash as an escape, so emitting one
    verbatim produces a file that cannot be read back.  Windows paths make this
    certain, but a regular expression or a quoted phrase does it just as well,
    so it is not a platform quirk.
    """
    class Awkward(ParSet):
        default_key = 'awkward'
        value: Annotated[str, Field(default='', description='A string value.')]

    par = Awkward(value=awkward)
    f = tmp_path / 'awkward.toml'
    par.to_toml(cfg_file=f)

    assert tomllib.loads(f.read_text())['awkward']['value'] == awkward, \
        'The emitted TOML did not parse back to the string that was written'
    assert Awkward.from_toml(f).value == awkward, \
        'The parameter set did not survive a round trip through TOML'


def test_from_toml_missing_section(tmp_path):
    """Asking for a section that is not in the file is an error, not a default."""
    f = tmp_path / 'other.toml'
    f.write_text('[something_else]\na = 1\n')
    with pytest.raises(DC3ParameterError, match=r'no \[dc3\] section'):
        ExampleDC3Par.from_toml(f)


def test_header_round_trip():
    """to_header/from_header round-trips through a FITS header."""
    p = ExampleDC3Par(template={'velscale_ratio': 4}, fit={'method': 'de'})
    hdr = p.to_header()
    assert ExampleDC3Par.from_header(hdr) == p, \
        'Parameter set did not survive a FITS header round trip'


def test_header_survives_a_file(tmp_path):
    """The header round-trip survives an actual FITS write and read."""
    p = ExampleDC3Par(template={'velscale_ratio': 4})
    f = tmp_path / 'test.fits'
    fits.PrimaryHDU(header=p.to_header()).writeto(f)
    with fits.open(f) as hdu:
        assert ExampleDC3Par.from_header(hdu[0].header) == p, \
            'Parameter set did not survive being written to and read from a FITS file; the ' \
            'JSON card is long enough to need the FITS long-string convention'


def test_header_rejects_a_missing_card():
    """Reading a header that has no card for this ParSet is an error."""
    hdr = ExampleTemplatePar().to_header()
    # ExampleFitPar has its own card_prefix, so its cards are simply absent
    with pytest.raises(DC3ParameterError, match='does not include the card'):
        ExampleFitPar.from_header(hdr)


def test_header_rejects_the_wrong_class():
    """
    A header written by a *different* class sharing the card prefix is an error.

    This is the case that arises when a ParSet is renamed or moved but keeps its
    ``card_prefix``: the cards are present and parse, but they describe a
    different class, so the values cannot be trusted to mean the same thing.
    """
    class RenamedTemplatePar(ParSet):
        card_prefix = 'XTPL'     # deliberately the same as ExampleTemplatePar
        velscale_ratio: Annotated[int, Field(default=1, description='A parameter.')]

    hdr = ExampleTemplatePar().to_header()
    with pytest.raises(DC3ParameterError, match='Expected'):
        RenamedTemplatePar.from_header(hdr)


def test_header_requires_card_prefix():
    """A ParSet without a card_prefix cannot be written to a header."""
    class _NoPrefix(ParSet):
        a: Annotated[int, Field(default=1, description='A parameter.')]

    with pytest.raises(DC3CodingError, match='card_prefix'):
        _NoPrefix().to_header()


# ----------------------------------------------------------------------
# Layered merge
# ----------------------------------------------------------------------
def test_from_layers_precedence():
    """Command line beats file beats default."""
    p = ExampleTemplatePar.from_layers(
        file_cfg={'velscale_ratio': 2, 'epsilon_sigma': 0.5},
        cli_cfg={'velscale_ratio': 8},
    )
    assert p.velscale_ratio == 8, 'Command-line value did not take precedence over the file'
    assert p.epsilon_sigma == 0.5, 'File value was not retained where the command line was silent'
    assert p.mask_unmatched_idsp is False, \
        'Default was not retained where neither layer supplied a value'


def test_from_layers_ignores_unset_cli_values():
    """A None from argparse means 'not supplied', so it must not override."""
    p = ExampleTemplatePar.from_layers(
        file_cfg={'velscale_ratio': 2},
        cli_cfg={'velscale_ratio': None},
    )
    assert p.velscale_ratio == 2, \
        'An unset (None) command-line value overrode the configuration file'


def test_from_layers_merges_nested_sections():
    """Setting one member of a subsection must not discard its siblings."""
    p = ExampleDC3Par.from_layers(
        file_cfg={'template': {'velscale_ratio': 4, 'epsilon_sigma': 0.2}},
        cli_cfg={'template': {'velscale_ratio': 8}},
    )
    assert p.template.velscale_ratio == 8, \
        'Command-line value did not take precedence within a nested section'
    assert p.template.epsilon_sigma == 0.2, \
        'Overriding one member of a nested section discarded its siblings'


# ----------------------------------------------------------------------
# Documentation
# ----------------------------------------------------------------------
def test_rst_table():
    """The rst table includes every parameter, its default and description."""
    content = '\n'.join(ExampleTemplatePar.to_rst_table())
    for key in ExampleTemplatePar.keys():
        assert f'``{key}``' in content, f'Parameter {key} is missing from the generated table'
    assert 'Integer prepared-template pixels per galaxy pixel.' in content, \
        'Parameter description is missing from the generated table'
    assert '.. _exampletemplatepar:' in content, \
        'Generated table has no cross-reference anchor for the class'


def test_rst_table_recurses_without_repeating():
    """
    Nested sets get their own table, emitted exactly once.

    Counted on the "Class Instantiation" line rather than the section heading,
    because the heading text also appears as a hyperlink reference in the parent
    table's Default column.
    """
    content = '\n'.join(ExampleDC3Par.to_rst_table())
    for cls in [ExampleDC3Par, ExampleTemplatePar, ExampleFitPar]:
        marker = f'Class Instantiation: :class:`~{cls.__module__}.{cls.__name__}`'
        assert content.count(marker) == 1, \
            f'Table for {cls.__name__} was emitted {content.count(marker)} times; each nested ' \
            'parameter set must be documented exactly once'
    for name in ['ExampleTemplatePar', 'ExampleFitPar']:
        assert f'`{name} Parameters`_' in content, \
            f'Parent table does not link to the {name} section, so it is unreachable'


def test_rst_table_reports_literal_options():
    """Literal options are recovered for the documentation."""
    content = '\n'.join(ExampleFitPar.to_rst_table())
    for option in ['lsq', 'de', 'mcmc']:
        assert f'``{option}``' in content, \
            f'Allowed value {option!r} was not recovered from the Literal annotation'


def test_introspection():
    """keys(), nested() and field_default() report the declaration."""
    assert ExampleTemplatePar.keys() == [
        'velscale_ratio', 'epsilon_sigma', 'mask_unmatched_idsp'
    ], 'keys() did not report the declared parameters in declaration order'
    assert ExampleDC3Par.nested() == ['template', 'fit'], \
        'nested() did not identify exactly the parameters that are themselves parameter sets'
    assert ExampleTemplatePar.field_default('velscale_ratio') == 1, \
        'field_default() did not report the declared default'
    assert ExampleDC3Par.field_default('output_dir').endswith('()'), \
        'field_default() should name the default_factory, since there is no fixed default value'


# ----------------------------------------------------------------------
# api_doc
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    'reference,expected',
    [
        # Sphinx roles, which must lose their markup for plain-text output
        (':class:`~dc3.par.dc3par.TemplatePar`', 'dc3.par.dc3par.TemplatePar'),
        (':func:`scipy.optimize.least_squares`', 'scipy.optimize.least_squares'),
        (':py:func:`numpy.polynomial.legendre.Legendre.fit`',
         'numpy.polynomial.legendre.Legendre.fit'),
        # The explicit `text <target>` form resolves to the target
        (':func:`the fitter <scipy.optimize.least_squares>`', 'scipy.optimize.least_squares'),
        # Anything that is not a role passes through untouched
        ('https://pypi.org/project/ppxf/', 'https://pypi.org/project/ppxf/'),
        ('plain text', 'plain text'),
    ]
)
def test_plain_reference(reference, expected):
    """Sphinx role markup is stripped; URLs and plain text are left alone."""
    assert _plain_reference(reference) == expected, \
        f'{reference!r} was not reduced to its plain-text target'


def test_api_doc_in_toml_is_plain_text():
    """A Sphinx role is emitted into a TOML comment without its markup."""
    class Documented(ParSet):
        default_key = 'documented'
        api_doc = ':func:`scipy.optimize.least_squares`'
        xtol: Annotated[float, Field(default=1e-8, description='A parameter.')]

    content = Documented().to_toml()
    assert '# See scipy.optimize.least_squares' in content, \
        'api_doc was not emitted as a plain-text comment above the section'
    assert ':func:' not in content, \
        'Sphinx role markup leaked into the TOML comment, where it is read literally'


def test_api_doc_in_rst_keeps_the_role():
    """The rst table keeps the role, so that Sphinx can resolve it to a link."""
    class Documented(ParSet):
        api_doc = ':func:`scipy.optimize.least_squares`'
        xtol: Annotated[float, Field(default=1e-8, description='A parameter.')]

    content = '\n'.join(Documented.to_rst_table())
    assert ':func:`scipy.optimize.least_squares`' in content, \
        'api_doc role was stripped in rst output, where it should resolve to a link'


def test_api_doc_is_optional():
    """A parameter set without an api_doc emits no reference line."""
    content = ExampleTemplatePar().to_toml()
    assert '# See ' not in content, \
        'A reference comment was emitted for a parameter set that declares no api_doc'


def test_info_runs(capsys):
    """info() prints something for every parameter."""
    ExampleDC3Par().info()
    out = capsys.readouterr().out
    assert 'velscale_ratio' in out, 'info() did not report a nested parameter'
    assert 'Description:' in out, 'info() did not report parameter descriptions'
