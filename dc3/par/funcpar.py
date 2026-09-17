"""
A :class:`~dc3.par.parset.ParSet` subclass that collects the keyword arguments
of a function.

.. note::

    The design follows ``pypeit/par/funcpar.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause); see
    ``licenses/README.rst``.  The implementation differs because ``dc3``'s
    :class:`~dc3.par.parset.ParSet` is built on pydantic: fields are injected by
    a metaclass before pydantic collects them, rather than by
    ``__init_subclass__`` building a ``parameters`` dictionary.

Wrap **third-party functions only**
-----------------------------------

:class:`FuncPar` deliberately carries no ``options``, no hand-written
descriptions, and a type only as good as the wrapped function's own
annotations.  That is the design, not a shortfall, and it dictates where the
class may be used:

+---------------------+----------------------+-----------------------------------------+
| Function            | Parameter set        | Why                                     |
+=====================+======================+=========================================+
| Internal to ``dc3`` | hand-written         | ``dc3`` owns the parameter, so ``dc3``  |
|                     | :class:`ParSet`      | owns its type, options and description. |
+---------------------+----------------------+-----------------------------------------+
| Third-party         | :class:`FuncPar`     | The upstream package owns all of that,  |
|                     |                      | and ``dc3`` **depends on its            |
|                     |                      | documentation rather than reproducing   |
|                     |                      | it**.                                   |
+---------------------+----------------------+-----------------------------------------+

Reproducing a dependency's parameter documentation guarantees it will drift, and
drifted documentation is worse than a pointer to the authoritative source.  So a
:class:`FuncPar` states which keywords ``dc3`` exposes and sends the user
upstream for what they mean, via :attr:`~dc3.par.parset.ParSet.api_doc`.

What ``dc3`` *does* own in that case is the **restriction**: :attr:`kw_subset`
and :attr:`omitted_keys` are an editorial decision about which knobs are
appropriate here, and that choice should be documented even though the
parameters themselves are not.

How little the type inference recovers
--------------------------------------

Types come from the wrapped function's annotations, and in practice there
usually are none.  Measured over the functions ``dc3`` wraps (see
``prototypes/annotation_survey.py``), only 17% of their keyword arguments are
annotated: ``ppxf`` 0 of 5, ``scipy`` 0 of 44, ``numpy`` 0 of 6, with only
:func:`astropy.stats.sigma_clip` fully annotated.

An unannotated parameter is typed :obj:`typing.Any`, which means **no validation at
all**.  What :class:`FuncPar` still buys is worth being clear about:

- the keyword list and defaults, tracked automatically against upstream, so they
  cannot silently rot; and
- **loud failure**: naming a keyword in :attr:`kw_subset` that upstream has
  renamed or removed raises at class creation, i.e. at import, rather than at
  run time.

It also means :attr:`~dc3.par.parset.ParSet.api_doc` is load-bearing rather than
a courtesy: for most wrapped functions it is the *only* documentation a user
gets, since the generated per-parameter descriptions say nothing beyond naming
the function.

.. include:: ../include/links.rst
"""

import inspect
import types
from typing import Any, ClassVar, Union, get_args, get_origin, get_type_hints

from pydantic import Field
from pydantic._internal._model_construction import ModelMetaclass

from ..pkg.exceptions import DC3CodingError
from .parset import ParSet


__all__ = ['FuncPar']


def _unwrap(func):
    """
    Return the underlying function, unwrapping a :class:`staticmethod` if present.

    Parameters
    ----------
    func : callable, staticmethod
        The object to unwrap.

    Returns
    -------
    callable
        The underlying function.
    """
    return func.__func__ if isinstance(func, staticmethod) else func


def _get_module(func):
    """
    Get the parent module for the provided function.

    Parameters
    ----------
    func : callable
        The callable to inspect.

    Returns
    -------
    str, None
        The name of the parent module of ``func``, or None if
        :func:`inspect.getmodule` is unsuccessful.
    """
    module = inspect.getmodule(func)
    return None if module is None else module.__name__


def _default_kwargs(func, kw_subset, omitted_keys):
    """
    Extract the keyword arguments of a function and their default values.

    Parameters
    ----------
    func : callable
        The callable whose keyword arguments are collected.
    kw_subset : list, None
        A subset of keyword arguments to extract.  If None, all keywords are
        included.  Naming a keyword that is *not* in the function signature
        raises, because that means upstream has changed and ``dc3`` must be
        updated to match; it is not user error.
    omitted_keys : list, None
        Keyword arguments to exclude.  Any entry that is not in the signature is
        ignored, since omitting something that has already gone away is
        harmless.

    Returns
    -------
    dict
        The keyword parameters and their default values.

    Raises
    ------
    DC3CodingError
        Raised if ``kw_subset`` names a keyword that the function does not have.
    """
    kwargs = {
        key: par.default for key, par in inspect.signature(func).parameters.items()
        if par.default is not inspect.Parameter.empty
    }

    if kw_subset is not None:
        missing = [key for key in kw_subset if key not in kwargs]
        if len(missing) > 0:
            raise DC3CodingError(
                f'{missing} are not keyword arguments of {func.__name__}.  Its keywords are '
                f'{sorted(kwargs.keys())}.  If the upstream signature has changed, the '
                'parameter set must be updated to match.'
            )
        kwargs = {key: value for key, value in kwargs.items() if key in kw_subset}

    if omitted_keys is not None:
        for key in omitted_keys:
            kwargs.pop(key, None)

    return kwargs


def _type_hints(func):
    """
    Resolve the type hints for the keyword arguments of a function.

    Uses :func:`typing.get_type_hints` rather than reading
    `inspect.Parameter.annotation` directly, so that postponed annotations
    (strings, as produced when a module uses ``from __future__ import
    annotations``) are evaluated back into real type objects.

    Parameters
    ----------
    func : callable
        The callable to inspect.

    Returns
    -------
    dict
        Mapping of parameter name to its resolved annotation.  Empty if the
        annotations cannot be resolved, which is the common case: most
        third-party scientific code is unannotated.
    """
    try:
        return get_type_hints(func)
    except (NameError, TypeError):
        # NameError: a postponed annotation names something that cannot be
        # resolved in the function's module namespace.  TypeError: documented by
        # typing.get_type_hints for objects that cannot carry annotations.
        return {}


def _annotation(hint):
    """
    Convert a resolved annotation into one usable as a pydantic field type.

    Parameters
    ----------
    hint : object
        The resolved annotation, or None if the parameter is unannotated.

    Returns
    -------
    object
        The annotation to declare.  :obj:`typing.Any` is returned whenever the hint
        is absent or cannot be cleanly translated, which disables validation for
        that parameter.
    """
    if hint is None:
        return Any
    if isinstance(hint, type):
        return hint
    # Allow typing.Optional[T]/typing.Union[...] and the T | U syntax
    if get_origin(hint) in (Union, types.UnionType):
        args = [a for a in get_args(hint) if a is not type(None)]
        if len(args) > 0 and all(isinstance(a, type) for a in args):
            return hint
    return Any


def _inherited(namespace, bases, key):
    """
    Look up a class attribute in the namespace being built, then in its bases.

    Parameters
    ----------
    namespace : dict
        The class namespace under construction.
    bases : tuple
        The base classes.
    key : str
        The attribute name.

    Returns
    -------
    object
        The attribute value, or None if it is not set anywhere.
    """
    if key in namespace:
        return namespace[key]
    for base in bases:
        value = getattr(base, key, None)
        if value is not None:
            return value
    return None


class FuncParMeta(ModelMetaclass):
    """
    Metaclass that injects a wrapped function's keyword arguments as fields.

    The work must happen here rather than in ``__init_subclass__`` because
    pydantic collects fields from the class namespace as the class is created;
    by the time ``__init_subclass__`` runs, that collection is already done.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Build the field declarations before pydantic collects them."""
        func = _inherited(namespace, bases, 'func')

        # FuncPar itself declares no function; nothing to inject.
        if func is None:
            return super().__new__(mcs, name, bases, namespace, **kwargs)

        _func = _unwrap(func)
        func_kwargs = _default_kwargs(
            _func,
            _inherited(namespace, bases, 'kw_subset'),
            _inherited(namespace, bases, 'omitted_keys'),
        )
        if len(func_kwargs) == 0:
            raise DC3CodingError(
                f'{name} wraps {_func.__name__}, which exposes no keyword arguments to collect.  '
                'A function whose arguments are all positional cannot be wrapped by FuncPar; '
                'declare a ParSet by hand instead.'
            )

        hints = _type_hints(_func)
        descr = f'Parameter for {_func.__name__} in {_get_module(_func)}.'
        annotations = namespace.setdefault('__annotations__', {})
        for key, default in func_kwargs.items():
            annotations[key] = _annotation(hints.get(key))
            namespace[key] = Field(default=default, description=descr)

        # Wrap in staticmethod so that accessing it through an instance returns
        # the function itself rather than binding the instance as its first
        # argument.
        namespace['func'] = staticmethod(_func)

        return super().__new__(mcs, name, bases, namespace, **kwargs)


class FuncPar(ParSet, metaclass=FuncParMeta):
    """
    A :class:`~dc3.par.parset.ParSet` whose parameters are a function's keywords.

    This is an abstract base class; it should not be instantiated directly.  A
    subclass declares the function it wraps, and optionally which of its
    keywords to expose:

    .. code-block:: python

        class VarsmoothPar(FuncPar):
            func = ppxf_util.varsmooth
            kw_subset = ['oversample']
            default_key = 'varsmooth'
            card_prefix = 'VSM'
            api_doc = 'https://pypi.org/project/ppxf/'

    The fields are then derived from the signature at class-creation time:
    defaults come from the signature, and types from the function's own
    annotations where it has them.

    .. warning::

        Use this for **third-party functions only**, and always set
        :attr:`~dc3.par.parset.ParSet.api_doc`.  See the module documentation
        for why.

    Parameters
    ----------
    **kwargs
        Initial values for the parameters.  Keywords must be among those
        collected from the wrapped function.

    Attributes
    ----------
    module : str
        The module in which the wrapped function is defined.
    name : str
        The name of the wrapped function.
    """

    func: ClassVar = None
    """
    The callable whose keyword arguments are collected.  Stored as a
    :class:`staticmethod` so that it is not bound to the instance.
    """

    kw_subset: ClassVar[list | None] = None
    """
    The subset of keyword arguments to expose, or None for all of them.

    This is ``dc3``'s editorial decision about which knobs of the wrapped
    function are appropriate here, and it is the one thing about a
    :class:`FuncPar` that ``dc3`` owns rather than defers upstream.  Naming a
    keyword the function does not have raises at import.
    """

    omitted_keys: ClassVar[list | None] = None
    """
    Keyword arguments to exclude; the complement of :attr:`kw_subset`.

    Unlike :attr:`kw_subset`, naming a keyword the function does not have is
    ignored: omitting something that has already gone away is harmless.
    """

    @property
    def module(self):
        """The module in which the wrapped function is defined."""
        return _get_module(type(self).func)

    @property
    def name(self):
        """The name of the wrapped function."""
        return type(self).func.__name__
