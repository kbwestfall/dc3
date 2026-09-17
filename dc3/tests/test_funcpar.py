"""
Tests for :class:`~dc3.par.funcpar.FuncPar`.

Several of these run against the *real* third-party functions ``dc3`` wraps,
rather than against stand-ins.  That is deliberate: the point of
:class:`~dc3.par.funcpar.FuncPar` is to track upstream signatures, so a test
that only exercises a local stub would not notice the thing the class exists to
catch.
"""

import inspect
from typing import Any

from ppxf import ppxf_util
import pytest
import scipy.optimize

from dc3.par.funcpar import FuncPar
from dc3.par.parset import ParSet
from dc3.pkg.exceptions import DC3CodingError

from .test_parset import check_declaration


# ----------------------------------------------------------------------
# Example wrappers
# ----------------------------------------------------------------------
class VarsmoothPar(FuncPar):
    """Exposes the one varsmooth keyword dc3 lets a user set."""

    func = ppxf_util.varsmooth
    kw_subset = ['oversample']
    default_key = 'varsmooth'
    card_prefix = 'VSM'
    api_doc = 'https://pypi.org/project/ppxf/'


class LogRebinPar(FuncPar):
    """Exposes log_rebin's keywords except the one dc3 controls itself."""

    func = ppxf_util.log_rebin
    omitted_keys = ['velscale']
    default_key = 'log_rebin'
    card_prefix = 'LRB'
    api_doc = 'https://pypi.org/project/ppxf/'


class LeastSquaresPar(FuncPar):
    """A wrapper over a scipy function, which publishes an object inventory."""

    func = scipy.optimize.least_squares
    kw_subset = ['ftol', 'xtol', 'gtol', 'max_nfev']
    default_key = 'least_squares'
    card_prefix = 'LSQ'
    api_doc = ':func:`scipy.optimize.least_squares`'


# ----------------------------------------------------------------------
# Field derivation
# ----------------------------------------------------------------------
def test_fields_come_from_the_signature():
    """The collected keywords and defaults match the wrapped function."""
    signature = inspect.signature(ppxf_util.varsmooth)
    assert VarsmoothPar.keys() == ['oversample'], \
        'kw_subset did not restrict the collected keywords to exactly those requested'
    assert VarsmoothPar().oversample == signature.parameters['oversample'].default, \
        'Default was not taken from the wrapped function signature'


def test_kw_subset_and_omitted_keys_are_complements():
    """omitted_keys removes a keyword that would otherwise be collected."""
    all_kwargs = {
        key for key, par in inspect.signature(ppxf_util.log_rebin).parameters.items()
        if par.default is not inspect.Parameter.empty
    }
    assert 'velscale' in all_kwargs, \
        'Test assumes log_rebin has a velscale keyword; the upstream signature has changed'
    assert set(LogRebinPar.keys()) == all_kwargs - {'velscale'}, \
        'omitted_keys did not remove exactly the excluded keyword'


def test_unknown_kw_subset_fails_at_class_creation():
    """
    Naming a keyword upstream does not have is a coding error, raised at import.

    This is the mechanism that makes an upstream rename fail loudly rather than
    silently dropping a parameter.
    """
    with pytest.raises(DC3CodingError, match='are not keyword arguments'):
        class Broken(FuncPar):
            func = ppxf_util.varsmooth
            kw_subset = ['no_such_keyword']


def test_positional_only_function_is_rejected():
    """
    A function with no keyword arguments cannot be wrapped.

    ``losvd_rfft`` is the real case: all eight of its parameters are positional,
    so it needs a hand-written ParSet rather than a FuncPar.
    """
    has_kwargs = any(
        par.default is not inspect.Parameter.empty
        for par in inspect.signature(ppxf_util.losvd_rfft).parameters.values()
    )
    assert not has_kwargs, \
        'Test assumes losvd_rfft is entirely positional; the upstream signature has changed'

    with pytest.raises(DC3CodingError, match='no keyword arguments'):
        class Broken(FuncPar):
            func = ppxf_util.losvd_rfft


def test_func_is_not_bound_to_the_instance():
    """
    The wrapped function must not become a bound method.

    Assigning a plain function as a class attribute would bind the instance as
    its first argument, so it is stored as a staticmethod.
    """
    assert VarsmoothPar.func is ppxf_util.varsmooth, \
        'Class access to func did not return the wrapped function'
    assert VarsmoothPar().func is ppxf_util.varsmooth, \
        'Instance access to func returned a bound method rather than the function'


def test_module_and_name():
    """The wrapped function's identity is reported."""
    p = VarsmoothPar()
    assert p.name == 'varsmooth', 'Wrapped function name was not reported'
    assert p.module == 'ppxf.ppxf_util', 'Wrapped function module was not reported'


# ----------------------------------------------------------------------
# Typing: what the annotations actually recover
# ----------------------------------------------------------------------
def test_unannotated_upstream_yields_no_validation():
    """
    An unannotated keyword is typed Any, so nothing is validated.

    This is not a defect to fix but the measured reality of the functions dc3
    wraps: ppxf, scipy and numpy annotate none of their keyword arguments.  The
    test pins the behaviour so that it is a documented property rather than a
    surprise.
    """
    assert VarsmoothPar.model_fields['oversample'].annotation is Any, \
        'Expected Any for an unannotated upstream keyword'
    # Consequently a nonsensical value is accepted
    assert VarsmoothPar(oversample='not-a-number').oversample == 'not-a-number', \
        'A parameter typed Any should accept any value; validation cannot come from nowhere'


def test_generated_descriptions_name_the_function():
    """
    Every collected parameter gets a description naming the wrapped function.

    They are deliberately uninformative -- the meaning lives upstream -- but they
    must exist, so that a FuncPar passes the same declaration check as any other
    parameter set.
    """
    for cls in [VarsmoothPar, LogRebinPar, LeastSquaresPar]:
        check_declaration(cls)
        for key, field in cls.model_fields.items():
            assert cls.func.__name__ in field.description, \
                f'Description for {key} does not name the wrapped function'


# ----------------------------------------------------------------------
# Integration with ParSet
# ----------------------------------------------------------------------
def test_is_a_parset():
    """A FuncPar is a ParSet, so all of the base-class machinery applies."""
    assert issubclass(VarsmoothPar, ParSet), 'FuncPar subclasses must be ParSet subclasses'


def test_toml_round_trip(tmp_path):
    """A FuncPar round-trips through a TOML file like any other ParSet."""
    p = LeastSquaresPar(xtol=1e-10)
    f = tmp_path / 'fit.toml'
    p.to_toml(cfg_file=f)
    assert LeastSquaresPar.from_toml(f) == p, \
        'FuncPar did not survive a round trip through a TOML file'


def test_header_round_trip():
    """A FuncPar round-trips through a FITS header like any other ParSet."""
    p = VarsmoothPar(oversample=4)
    assert VarsmoothPar.from_header(p.to_header()) == p, \
        'FuncPar did not survive a FITS header round trip'


def test_api_doc_is_emitted():
    """
    The documentation pointer reaches both outputs.

    For a FuncPar this is the only real documentation a user gets, since the
    generated per-parameter descriptions say nothing beyond naming the function.
    """
    toml = LeastSquaresPar().to_toml()
    assert '# See scipy.optimize.least_squares' in toml, \
        'api_doc was not emitted as a plain-text comment in the TOML output'

    rst = '\n'.join(LeastSquaresPar.to_rst_table())
    assert ':func:`scipy.optimize.least_squares`' in rst, \
        'api_doc role was not emitted intact in the rst output'


def test_url_api_doc_passes_through():
    """A bare URL is emitted unchanged, since it carries no role markup."""
    toml = VarsmoothPar().to_toml()
    assert '# See https://pypi.org/project/ppxf/' in toml, \
        'A URL api_doc was altered on its way into the TOML comment'
