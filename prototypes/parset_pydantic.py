"""
Prototype: TemplatePar on pydantic v2, exercising the four capabilities the port
plan says pydantic does not give for free, plus FuncPar-equivalent behaviour.

Run:  python proto_pydantic.py
"""
import ast
import inspect
import types
import typing
from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, create_model
from astropy.io import fits


# --------------------------------------------------------------------------
# 1. The parameter set itself.  This is the part pydantic makes cheap.
# --------------------------------------------------------------------------
class TemplatePar(BaseModel):
    """Parameters governing template preparation."""
    model_config = {'extra': 'forbid', 'validate_assignment': True}

    velscale_ratio: Annotated[int, Field(
        default=1, ge=1,
        description='Integer number of prepared-template pixels per galaxy pixel.'
    )]
    epsilon_sigma: Annotated[float, Field(
        default=0.1, ge=0.1,
        description='Target for the minimum dispersion of the preparation kernel, in pixels.'
    )]
    sigma_floor: Annotated[float, Field(
        default=0.0, ge=0.0,
        description='Largest pedestal allowed to accommodate template regions of lower '
                    'resolution than the galaxy; sets the most negative dvar_inst.'
    )]
    mask_unmatched_sres: Annotated[bool, Field(
        default=False,
        description='Mask template regions that cannot reach the target resolution.'
    )]


# --------------------------------------------------------------------------
# 2. to_rst_table -- must be written.
# --------------------------------------------------------------------------
def _type_name(ann):
    """Render an annotation as a type-name string."""
    ann = _strip_annotated(ann)
    if get_origin(ann) in (typing.Union, types.UnionType):
        return ', '.join(
            a.__name__ for a in get_args(ann) if a is not type(None)
        )
    if get_origin(ann) is Literal:
        return 'str'
    return getattr(ann, '__name__', str(ann))


def _strip_annotated(ann):
    return get_args(ann)[0] if get_origin(ann) is Annotated else ann


def _options(ann):
    """Extract Literal options, if any."""
    ann = _strip_annotated(ann)
    return list(get_args(ann)) if get_origin(ann) is Literal else None


def to_rst_table(model_cls):
    """Build an rst table from the model fields."""
    rows = [['Key', 'Type', 'Options', 'Default', 'Description']]
    for name, f in model_cls.model_fields.items():
        opts = _options(f.annotation)
        rows.append([
            f'``{name}``',
            _type_name(f.annotation),
            '..' if opts is None else ', '.join(f'``{o}``' for o in opts),
            f'``{f.default}``',
            f.description or '..',
        ])
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    bar = '  '.join('=' * w[j] for j in range(len(w)))
    out = [bar, '  '.join(rows[0][j].ljust(w[j]) for j in range(len(w))), bar]
    out += ['  '.join(r[j].ljust(w[j]) for j in range(len(w))) for r in rows[1:]]
    out += [bar]
    return '\n'.join(out)


# --------------------------------------------------------------------------
# 3. to_header / from_header -- must be written.
# --------------------------------------------------------------------------
CARD_PREFIX = 'TPL'


def to_header(model, hdr=None):
    if hdr is None:
        hdr = fits.Header()
    hdr[f'{CARD_PREFIX}C'] = f'{type(model).__module__}.{type(model).__name__}'
    # JSON round-trip rather than repr + ast.literal_eval
    hdr[f'{CARD_PREFIX}D'] = model.model_dump_json()
    return hdr


def from_header(model_cls, hdr):
    return model_cls.model_validate_json(hdr[f'{CARD_PREFIX}D'])


# --------------------------------------------------------------------------
# 4. Layered default -> file -> CLI merge -- must be written.
# --------------------------------------------------------------------------
def merge(model_cls, file_cfg=None, cli_cfg=None):
    cfg = {}
    if file_cfg:
        cfg.update(file_cfg)
    if cli_cfg:
        cfg.update({k: v for k, v in cli_cfg.items() if v is not None})
    return model_cls(**cfg)


# --------------------------------------------------------------------------
# 5. TOML emitter with descriptions as comments -- must be written.
# --------------------------------------------------------------------------
def to_toml(model, section):
    lines = [f'[{section}]']
    for name, f in type(model).model_fields.items():
        if f.description:
            lines.append(f'# {f.description}')
        v = getattr(model, name)
        lines.append(f'{name} = {str(v).lower() if isinstance(v, bool) else v!r}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# 6. FuncPar equivalent: derive a model from a third-party function signature.
# --------------------------------------------------------------------------
def funcpar(func, kw_subset=None, omitted_keys=None, name=None):
    """pydantic create_model equivalent of pypeit's FuncPar."""
    sig = inspect.signature(func)
    kwargs = {
        k: v.default for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    if kw_subset is not None:
        bad = [k for k in kw_subset if k not in kwargs]
        if bad:
            raise KeyError(f'CODING ERROR: {bad} are not valid kwargs for {func.__name__}!')
        kwargs = {k: v for k, v in kwargs.items() if k in kw_subset}
    for k in (omitted_keys or []):
        kwargs.pop(k, None)
    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError):
        hints = {}
    descr = f'Parameter for {func.__name__} in {inspect.getmodule(func).__name__}.'
    fields = {}
    for k, v in kwargs.items():
        ann = hints.get(k, typing.Any)
        fields[k] = (ann, Field(default=v, description=descr))
    return create_model(name or f'{func.__name__.title()}Par', **fields)


# ==========================================================================
if __name__ == '__main__':
    print('=' * 74)
    print('1. Construction, defaults, validation')
    print('=' * 74)
    p = TemplatePar()
    print('defaults:', p.model_dump())

    for bad, why in [
        ({'velscale_ratio': 0}, 'below ge=1'),
        ({'epsilon_sigma': 0.05}, 'below the varsmooth 0.1 px floor'),
        ({'velscale_ratio': 'two'}, 'wrong type'),
        ({'bogus_key': 1}, 'unknown key'),
    ]:
        try:
            TemplatePar(**bad)
            print(f'  NOT CAUGHT: {bad}  ({why})')
        except ValidationError as e:
            print(f'  caught {bad}: {why} -> {e.errors()[0]["type"]}')

    # coercion
    print('  coercion  :', TemplatePar(velscale_ratio=2, epsilon_sigma=1).model_dump())
    # None is NOT silently allowed (contrast with ParSet.__setitem__)
    try:
        TemplatePar(velscale_ratio=None)
        print('  None allowed for velscale_ratio (like ParSet)')
    except ValidationError:
        print('  None REJECTED for velscale_ratio (unlike ParSet)')

    print()
    print('=' * 74)
    print('2. to_rst_table')
    print('=' * 74)
    print(to_rst_table(TemplatePar))

    print()
    print('=' * 74)
    print('3. FITS header round-trip')
    print('=' * 74)
    hdr = to_header(TemplatePar(velscale_ratio=4, mask_unmatched_sres=True))
    print('card len:', len(hdr['TPLD']), '| value:', hdr['TPLD'])
    back = from_header(TemplatePar, hdr)
    print('round-trip:', back.model_dump())

    print()
    print('=' * 74)
    print('4. Layered merge + 5. TOML')
    print('=' * 74)
    merged = merge(TemplatePar, file_cfg={'velscale_ratio': 2},
                   cli_cfg={'velscale_ratio': 8, 'epsilon_sigma': None})
    print('merged:', merged.model_dump())
    print(to_toml(merged, 'template'))

    print()
    print('=' * 74)
    print('6. FuncPar equivalent on a real third-party function')
    print('=' * 74)
    from ppxf import ppxf_util
    VarsmoothPar = funcpar(ppxf_util.varsmooth, kw_subset=['oversample'])
    vp = VarsmoothPar()
    print('varsmooth kwargs ->', vp.model_dump())
    print('field annotation:', VarsmoothPar.model_fields['oversample'].annotation)
    print('descr           :', VarsmoothPar.model_fields['oversample'].description)
    # does it type-check?  depends entirely on upstream annotations
    try:
        VarsmoothPar(oversample='not-a-number')
        print('  NO type checking (upstream is unannotated -> typing.Any)')
    except ValidationError:
        print('  type checking active')
    print()
    print('varsmooth signature:', inspect.signature(ppxf_util.varsmooth))
    print('varsmooth hints    :', typing.get_type_hints(ppxf_util.varsmooth))
