"""
The base class used to hold runtime parameters.

:class:`~dc3.par.parset.ParSet` is a thin layer over `pydantic.BaseModel` that
adds the four capabilities ``dc3`` needs and pydantic does not provide:
reStructuredText table generation, a FITS header round-trip, a layered
default/file/command-line merge, and TOML configuration output that preserves
parameter descriptions as comments.

Everything else -- type declaration and coercion, value options, defaults,
nested parameter sets, and validation -- is pydantic's.

.. note::

    The *interface* here follows ``pypeit/par/parset.py`` in `PypeIt
    <https://github.com/pypeit/PypeIt>`__ (BSD 3-Clause), and the rst-table and
    FITS-header idioms are adapted from it; see ``licenses/README.rst``.  The
    implementation is not: PypeIt's ``ParSet`` is hand-rolled, where this is
    built on pydantic.  See ``prototypes/`` for the comparison that settled
    that choice.

Two differences from PypeIt's ``ParSet`` are worth knowing when reading code
written against the other:

- **Access is by attribute, not by key.** ``par.velscale_ratio``, not
  ``par['velscale_ratio']``.  Item access is retained as a compatibility
  shim, but attribute access is the idiom.
- **A parameter cannot silently be None.**  PypeIt's ``__setitem__`` always
  permits None regardless of the declared type; here a parameter is nullable
  only if its annotation says so.

.. include:: ../include/links.rst
"""

import shutil
import textwrap
import tomllib
import types
from pathlib import Path
from typing import ClassVar, Literal, Union, get_args, get_origin

from astropy.io import fits
import numpy as np
from pydantic import BaseModel, ConfigDict

from ..pkg.exceptions import DC3CodingError, DC3ParameterError


__all__ = ['ParSet']


def _is_parset(obj):
    """Return True if ``obj`` is a :class:`ParSet` subclass."""
    return isinstance(obj, type) and issubclass(obj, ParSet)


def _field_options(annotation):
    """
    Return the allowed values for a field, or None.

    Options are declared using `typing.Literal`; this recovers them for
    documentation.  Unions of Literals are flattened.

    Parameters
    ----------
    annotation : object
        The field annotation, as reported by ``model_fields[...].annotation``.

    Returns
    -------
    list, None
        The allowed values, or None if the field is not constrained to a
        literal set.
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return list(get_args(annotation))
    if origin in (Union, types.UnionType):
        options = []
        for arg in get_args(annotation):
            if get_origin(arg) is Literal:
                options += list(get_args(arg))
        return None if len(options) == 0 else options
    return None


def _type_name(annotation):
    """
    Render a field annotation as a human-readable type name.

    Parameters
    ----------
    annotation : object
        The field annotation.

    Returns
    -------
    str
        A comma-separated list of type names.  A `typing.Literal` is reported
        as the type of its members, since that is what a user must supply.
    """
    if annotation is None or annotation is type(None):
        return 'NoneType'
    origin = get_origin(annotation)
    if origin is Literal:
        members = get_args(annotation)
        return 'str' if len(members) == 0 else type(members[0]).__name__
    if origin in (Union, types.UnionType):
        return ', '.join(_type_name(a) for a in get_args(annotation) if a is not type(None))
    if origin is not None:
        return getattr(origin, '__name__', str(origin))
    return getattr(annotation, '__name__', str(annotation))


def _toml_value(value):
    """
    Render a value in TOML syntax.

    Parameters
    ----------
    value : object
        The value to render.

    Returns
    -------
    str
        The TOML representation.

    Raises
    ------
    DC3CodingError
        Raised if the value is None, which the caller is responsible for
        omitting.
    DC3ParameterError
        Raised if the value has no TOML representation.
    """
    # NOTE: bool must be tested before int, because bool IS a subclass of int.
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (str, Path)):
        return f'"{value}"'
    if isinstance(value, (list, tuple, np.ndarray)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    if value is None:
        # TOML has no null.  A None-valued parameter is omitted from the output
        # by the caller, so reaching here means the caller did not.
        raise DC3CodingError('TOML has no representation for None.')
    raise DC3ParameterError(f'No TOML representation for type {type(value).__name__}.')


class ParSet(BaseModel):
    """
    Base class used to collect runtime parameters.

    This is an abstract base class; it should not be instantiated directly.
    Parameters are declared as annotated pydantic fields, so a subclass looks
    like:

    .. code-block:: python

        class TemplatePar(ParSet):
            default_key = 'template'
            card_prefix = 'TPL'

            velscale_ratio: Annotated[int, Field(
                default=1, ge=1,
                description='Prepared-template pixels per galaxy pixel.'
            )]

    Every parameter **must** carry a ``description``; it is what the generated
    documentation and the commented TOML output are built from.  This is
    enforced by the test suite, not at run time, since violating it is a coding
    error rather than something a user can do; see
    ``dc3.tests.test_parset.check_declaration``.

    Parameters
    ----------
    **kwargs
        Initial values for the parameters.  Keywords must match declared
        fields; an unrecognized keyword raises a validation error rather than
        being silently absorbed.
    """

    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        arbitrary_types_allowed=True,
        validate_default=True,
    )

    default_key: ClassVar[str | None] = None
    """The default key to use for this parameter set in a configuration file."""

    default_comment: ClassVar[str | None] = None
    """The default comment to use for this parameter set in a configuration file."""

    card_prefix: ClassVar[str | None] = None
    """
    The header card prefix used when writing this parameter set to a FITS
    header.  Must be at most 7 characters, since a suffix character is appended
    and FITS keywords are limited to 8.
    """

    doc_url: ClassVar[str | None] = None
    """
    A pointer to authoritative documentation for these parameters.

    This is None for parameter sets that ``dc3`` owns, which document themselves
    through their ``description`` fields.  It is set for
    :class:`~dc3.par.funcpar.FuncPar` subclasses, which wrap a third-party
    function and deliberately defer to *its* documentation rather than
    reproducing it.  See :class:`~dc3.par.funcpar.FuncPar`.
    """

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @classmethod
    def keys(cls):
        """
        Return the list of parameter names.

        Returns
        -------
        list
            The declared field names, in declaration order.
        """
        return list(cls.model_fields.keys())

    @classmethod
    def nested(cls):
        """
        Return the names of the parameters that are themselves parameter sets.

        Returns
        -------
        list
            Field names whose value is a :class:`ParSet`.
        """
        return [k for k, f in cls.model_fields.items() if _is_parset(f.annotation)]

    @classmethod
    def field_default(cls, key):
        """
        Return a display value for a parameter's default.

        Parameters
        ----------
        key : str
            The parameter name.

        Returns
        -------
        object, str
            The default value, or a descriptive string naming the
            ``default_factory`` when the default is computed at instantiation.
        """
        f = cls.model_fields[key]
        if f.default_factory is not None:
            name = getattr(f.default_factory, '__qualname__', repr(f.default_factory))
            return f'{name}()'
        return f.default

    # ------------------------------------------------------------------
    # Compatibility shims.  Attribute access is the idiom; these exist so that
    # code and habits carried over from pypeit's ParSet keep working.
    # ------------------------------------------------------------------
    # NOTE: model_fields is accessed via type(self), not self.  Accessing it on
    # an instance is deprecated in pydantic 2.11 and is removed in 3.0.
    def __getitem__(self, key):
        """Get a parameter value by name."""
        if key not in type(self).model_fields:
            raise KeyError(f'{key} is not a valid parameter for {type(self).__name__}!')
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set a parameter value by name, with validation."""
        if key not in type(self).model_fields:
            raise KeyError(f'{key} is not a valid parameter for {type(self).__name__}!')
        setattr(self, key, value)

    def __len__(self):
        """Return the number of parameters."""
        return len(type(self).model_fields)

    # ------------------------------------------------------------------
    # Dictionaries
    # ------------------------------------------------------------------
    def to_dict(self):
        """
        Return the contents of the parameter set as a dictionary.

        Nested parameter sets are recursively converted.

        Returns
        -------
        dict
            The contents in dictionary form.
        """
        return self.model_dump(mode='json')

    @classmethod
    def from_dict(cls, cfg):
        """
        Instantiate from a dictionary.

        Nested parameter sets are handled recursively.

        Parameters
        ----------
        cfg : dict
            Dictionary used to instantiate the parameter set.

        Returns
        -------
        ParSet
            The instantiated parameter set.
        """
        return cls.model_validate(cfg)

    # ------------------------------------------------------------------
    # Layered merge
    # ------------------------------------------------------------------
    @classmethod
    def from_layers(cls, file_cfg=None, cli_cfg=None):
        """
        Build a parameter set by layering configuration sources.

        Precedence runs defaults < configuration file < command line.  A
        command-line value of None means "not supplied", so it does not override
        a value from the file; this is what makes an ``argparse`` namespace with
        ``default=None`` usable directly.

        Nested parameter sets are merged key by key, so a configuration file
        that sets one member of a subsection does not discard the others.

        Parameters
        ----------
        file_cfg : dict, optional
            Values read from a configuration file.
        cli_cfg : dict, optional
            Values supplied on the command line.

        Returns
        -------
        ParSet
            The merged parameter set.
        """
        cfg = {}
        for layer in (file_cfg, cli_cfg):
            if layer is None:
                continue
            for key, value in layer.items():
                if value is None:
                    # Not supplied at this layer
                    continue
                if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                    cfg[key] = {**cfg[key], **{k: v for k, v in value.items() if v is not None}}
                else:
                    cfg[key] = value
        return cls.model_validate(cfg)

    # ------------------------------------------------------------------
    # TOML
    # ------------------------------------------------------------------
    def config_lines(self, section_name=None, exclude_defaults=False, include_descr=True):
        """
        Generate the lines of a TOML configuration file for this parameter set.

        Nested parameter sets become dotted subsections
        (``[section.subsection]``), which is TOML's idiom for nesting.

        Parameters
        ----------
        section_name : str, optional
            Name of the top-level section.  If None, use :attr:`default_key`.
        exclude_defaults : bool, optional
            Omit parameters whose value is unchanged from the default.
        include_descr : bool, optional
            Include each parameter's description as a comment.

        Returns
        -------
        list
            The lines to write to a configuration file.

        Raises
        ------
        DC3ParameterError
            Raised if no section name is available.
        """
        _section = self.default_key if section_name is None else section_name
        if _section is None:
            raise DC3ParameterError(
                f'No section name available for {type(self).__name__}; provide section_name or '
                'set the default_key class attribute.'
            )

        lines = []
        if include_descr and self.default_comment is not None:
            lines += _comment_lines(self.default_comment)
        if include_descr and self.doc_url is not None:
            lines += _comment_lines(f'See {self.doc_url}')
        lines += [f'[{_section}]']

        nested = type(self).nested()
        for key, f in type(self).model_fields.items():
            if key in nested:
                continue
            value = getattr(self, key)
            if value is None:
                # TOML has no null; an unset parameter is simply absent.
                continue
            if exclude_defaults and value == f.default:
                continue
            if include_descr and f.description is not None:
                lines += _comment_lines(f.description)
            lines += [f'{key} = {_toml_value(value)}']

        for key in nested:
            value = getattr(self, key)
            if value is None:
                continue
            lines += ['']
            descr = type(self).model_fields[key].description
            if include_descr and descr is not None:
                lines += _comment_lines(descr)
            lines += value.config_lines(
                section_name=f'{_section}.{key}', exclude_defaults=exclude_defaults,
                include_descr=include_descr
            )
        return lines

    def to_toml(self, cfg_file=None, section_name=None, exclude_defaults=False,
                include_descr=True):
        """
        Write the parameter set to a TOML configuration file.

        Parameters
        ----------
        cfg_file : str, `pathlib.Path`, optional
            File to write.  If None, the lines are returned without writing.
        section_name : str, optional
            Name of the top-level section.  If None, use :attr:`default_key`.
        exclude_defaults : bool, optional
            Omit parameters whose value is unchanged from the default.
        include_descr : bool, optional
            Include each parameter's description as a comment.

        Returns
        -------
        str
            The configuration file contents, whether or not a file is written.
        """
        content = '\n'.join(
            self.config_lines(
                section_name=section_name, exclude_defaults=exclude_defaults,
                include_descr=include_descr
            )
        ) + '\n'
        if cfg_file is not None:
            Path(cfg_file).absolute().write_text(content)
        return content

    @classmethod
    def from_toml(cls, cfg_file, section_name=None):
        """
        Instantiate from a TOML configuration file.

        Parameters
        ----------
        cfg_file : str, `pathlib.Path`
            The file to read.
        section_name : str, optional
            The top-level section to read.  If None, use :attr:`default_key`;
            if that is also None, the whole document is used.

        Returns
        -------
        ParSet
            The instantiated parameter set.

        Raises
        ------
        DC3ParameterError
            Raised if the requested section is not in the file.
        """
        with open(cfg_file, 'rb') as f:
            doc = tomllib.load(f)
        _section = cls.default_key if section_name is None else section_name
        if _section is None:
            return cls.model_validate(doc)
        if _section not in doc:
            raise DC3ParameterError(
                f'Configuration file {cfg_file} has no [{_section}] section.  '
                f'Sections found: {list(doc.keys())}.'
            )
        return cls.model_validate(doc[_section])

    # ------------------------------------------------------------------
    # FITS headers
    # ------------------------------------------------------------------
    @classmethod
    def class_header_card(cls):
        """
        Return the header card naming the parameter-set class.

        Returns
        -------
        str
            The header keyword.

        Raises
        ------
        DC3CodingError
            Raised if :attr:`card_prefix` is not defined.
        """
        if cls.card_prefix is None:
            raise DC3CodingError(
                f'card_prefix for {cls.__name__} has not been defined!'
            )
        return f'{cls.card_prefix.upper()}C'

    @classmethod
    def dict_header_card(cls):
        """
        Return the header card holding the parameter-set contents.

        Returns
        -------
        str
            The header keyword.

        Raises
        ------
        DC3CodingError
            Raised if :attr:`card_prefix` is not defined.
        """
        if cls.card_prefix is None:
            raise DC3CodingError(
                f'card_prefix for {cls.__name__} has not been defined!'
            )
        return f'{cls.card_prefix.upper()}D'

    def to_header(self, hdr=None):
        """
        Include the parameter set in a FITS header.

        Two cards are added: the fully qualified class name, and the parameter
        values as a JSON document.  JSON is used rather than a dictionary
        ``repr`` so that reading the header back is a parse rather than an
        ``eval``, and so that the types survive the round trip exactly.

        Long values use the FITS long-string convention, which `astropy.io.fits`
        applies automatically.

        Parameters
        ----------
        hdr : `astropy.io.fits.Header`, optional
            Header to add the parameter set to.  If None, a new header is
            created.

        Returns
        -------
        `astropy.io.fits.Header`
            The header including the parameter set.
        """
        if hdr is None:
            hdr = fits.Header()
        hdr[self.class_header_card()] = f'{type(self).__module__}.{type(self).__name__}'
        hdr[self.dict_header_card()] = self.model_dump_json()
        return hdr

    @classmethod
    def from_header(cls, hdr):
        """
        Instantiate a parameter set from a FITS header.

        Parameters
        ----------
        hdr : `astropy.io.fits.Header`
            The header to read.

        Returns
        -------
        ParSet
            The instantiated parameter set.

        Raises
        ------
        DC3ParameterError
            Raised if :attr:`card_prefix` is not defined, if the expected cards
            are absent, or if the class recorded in the header does not match.
        """
        expected = hdr.get(cls.class_header_card())
        if expected is None:
            raise DC3ParameterError(
                f'Header does not include the card naming the ParSet class: '
                f'{cls.class_header_card()}.'
            )
        if expected != f'{cls.__module__}.{cls.__name__}':
            raise DC3ParameterError(
                f'Expected {cls.__module__}.{cls.__name__} in header card '
                f'{cls.class_header_card()}, but found {expected}.'
            )
        contents = hdr.get(cls.dict_header_card())
        if contents is None:
            raise DC3ParameterError(
                f'Header does not include the card with the ParSet contents: '
                f'{cls.dict_header_card()}.'
            )
        return cls.model_validate_json(contents)

    # ------------------------------------------------------------------
    # Documentation
    # ------------------------------------------------------------------
    @classmethod
    def to_rst_table(cls, parsets_listed=None):
        """
        Construct a reStructuredText table describing the parameter set.

        Works recursively for nested parameter sets, emitting one table per
        class and skipping any class already documented.

        Parameters
        ----------
        parsets_listed : list, optional
            Names of :class:`ParSet` subclasses that already have a table in the
            output, so that they are not repeated.

        Returns
        -------
        list
            Lines that can be written to an ``*.rst`` file.
        """
        if parsets_listed is None:
            parsets_listed = []
        parsets_listed.append(cls.__name__)

        new_parsets = []
        rows = [['Key', 'Type', 'Options', 'Default', 'Description']]
        for key in sorted(cls.model_fields.keys()):
            f = cls.model_fields[key]
            if _is_parset(f.annotation):
                if f.annotation.__name__ not in parsets_listed:
                    new_parsets.append(f.annotation)
                rows.append([
                    f'``{key}``',
                    f':class:`~{f.annotation.__module__}.{f.annotation.__name__}`',
                    '..',
                    f'`{f.annotation.__name__} Parameters`_',
                    '..' if f.description is None else f.description,
                ])
                continue
            options = _field_options(f.annotation)
            default = cls.field_default(key)
            rows.append([
                f'``{key}``',
                _type_name(f.annotation),
                '..' if options is None else ', '.join(f'``{o}``' for o in options),
                '..' if default is None else f'``{default}``',
                '..' if f.description is None else f.description,
            ])

        output = [f'.. _{cls.__name__.lower()}:', '']
        output += [f'{cls.__name__} Parameters', '-' * len(f'{cls.__name__} Parameters'), '']
        output += [f'Class Instantiation: :class:`~{cls.__module__}.{cls.__name__}`', '']
        if cls.doc_url is not None:
            output += [
                f'Parameters are those of the wrapped function; see {cls.doc_url} for their '
                'meaning.', ''
            ]
        output += [_rst_table(rows), '']
        for sub in new_parsets:
            output += ['----', '']
            output += sub.to_rst_table(parsets_listed=parsets_listed)
        return output

    def info(self, basekey=None):
        """
        Print a long-form description of the parameter set.

        Parameters
        ----------
        basekey : str, optional
            A prefix applied to each parameter name, used when recursing into
            nested parameter sets.
        """
        tcols = int(0.9 * shutil.get_terminal_size(fallback=(80, 25)).columns)
        for key, f in type(self).model_fields.items():
            value = getattr(self, key)
            if isinstance(value, ParSet):
                value.info(basekey=key if basekey is None else f'{basekey}:{key}')
                continue
            print(f'{key}' if basekey is None else f'{basekey}:{key}')
            options = _field_options(f.annotation)
            _wrap_print('        Value: ', value, tcols)
            _wrap_print('      Default: ', type(self).field_default(key), tcols)
            _wrap_print(
                '      Options: ',
                'None' if options is None else ', '.join(str(o) for o in options),
                tcols
            )
            _wrap_print('  Valid Types: ', _type_name(f.annotation), tcols)
            _wrap_print('  Description: ', f.description, tcols)
            print(' ')


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _comment_lines(comment, full_width=78):
    """
    Wrap a description into TOML comment lines.

    Parameters
    ----------
    comment : str
        The text to wrap.
    full_width : int, optional
        Maximum width of each output line.

    Returns
    -------
    list
        The comment lines, each prefixed with ``'# '``.
    """
    return [f'# {line}' for line in textwrap.wrap(str(comment), full_width - 2)]


def _rst_table(rows):
    """
    Format a list of rows as a simple reStructuredText table.

    Parameters
    ----------
    rows : list
        List of rows, each a list of strings.  The first row is the header.

    Returns
    -------
    str
        The formatted table.
    """
    width = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    bar = '  '.join('=' * w for w in width)
    out = [bar, '  '.join(rows[0][j].ljust(width[j]) for j in range(len(width))), bar]
    out += ['  '.join(r[j].ljust(width[j]) for j in range(len(width))) for r in rows[1:]]
    out += [bar]
    return '\n'.join(out)


def _wrap_print(head, output, tcols):
    """
    Print a wrapped, hanging-indented line.

    Parameters
    ----------
    head : str
        The inline header for the output.  May be empty, but not None.
    output : object
        The text to write.
    tcols : int
        The allowed width.
    """
    tail = ' ' * len(head)
    if tcols is None:
        print(head + f'{output}')
        return
    lines = textwrap.wrap(f'{output}', tcols - len(head))
    if len(lines) == 0:
        print(f'{head}None')
        return
    _head = [head] + [tail] * (len(lines) - 1)
    print('\n'.join(h + l for h, l in zip(_head, lines)))
