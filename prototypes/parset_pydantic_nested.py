"""
Prototype, part 2: the pydantic features a flat four-parameter set does not
exercise, and which carry the real risk in replacing pypeit's ParSet.

Specifically:
  - nested parameter sets, and recursive rst/TOML output
  - default_factory (a default knowable only at instantiation)
  - cross-parameter validation (ParSet.validate's job)
  - round-tripping a nested set through a FITS header
  - the `h3/h4 requires matched resolution` rule of science question 5, which is
    the concrete cross-parameter constraint the port actually needs

Run:  python parset_pydantic_nested.py
"""
from pathlib import Path
from typing import Annotated, Literal

from astropy.io import fits
from pydantic import BaseModel, Field, ValidationError, model_validator


class TemplatePar(BaseModel):
    """Parameters governing template preparation."""
    model_config = {'extra': 'forbid', 'validate_assignment': True}

    velscale_ratio: Annotated[int, Field(
        default=1, ge=1,
        description='Integer prepared-template pixels per galaxy pixel.')]
    epsilon_sigma: Annotated[float, Field(
        default=0.1, ge=0.1,
        description='Target minimum dispersion of the preparation kernel, in pixels.  The '
                    'floor of 0.1 matches a clip inside ppxf_util.varsmooth; a smaller value '
                    'would make dvar_inst wrong by the difference.')]
    sigma_floor: Annotated[float, Field(
        default=0.0, ge=0.0,
        description='Largest pedestal allowed for template regions of lower resolution than '
                    'the galaxy; sets the most negative dvar_inst.')]
    mask_unmatched_sres: Annotated[bool, Field(
        default=False,
        description='Mask template regions that cannot reach the target resolution.')]


class FitPar(BaseModel):
    """Parameters governing the kinematic fit."""
    model_config = {'extra': 'forbid', 'validate_assignment': True}

    method: Annotated[Literal['lsq', 'de', 'mcmc'], Field(
        default='lsq', description='Optimizer/sampler backend.')]
    moments: Annotated[int, Field(
        default=2, ge=2, le=6,
        description='Number of LOSVD moments to fit.  2 is V and sigma; >2 adds h3, h4, ...')]
    solve_norm: Annotated[bool, Field(
        default=True, description='Solve the amplitude analytically.')]


class DC3Par(BaseModel):
    """Top-level parameter set."""
    model_config = {'extra': 'forbid', 'validate_assignment': True}

    template: Annotated[TemplatePar, Field(
        default_factory=TemplatePar, description='Template preparation parameters.')]
    fit: Annotated[FitPar, Field(
        default_factory=FitPar, description='Fitting parameters.')]
    # default_factory: a default knowable only at instantiation time
    output_dir: Annotated[Path, Field(
        default_factory=Path.cwd, description='Directory for output products.')]

    @model_validator(mode='after')
    def _check_moments_require_matched_resolution(self):
        """
        Science question 5: the van der Marel & Franx parameterization is defined
        relative to the fitted Gaussian, so h3/h4 are only interpretable when the
        prepared templates are matched (dvar_inst ~ 0).  This must FAIL, not warn.
        """
        if self.fit.moments > 2 and self.template.sigma_floor > 0:
            raise ValueError(
                'Gauss-Hermite moments (fit.moments > 2) require matched template resolution, '
                'but template.sigma_floor > 0 permits a non-zero dvar_inst.'
            )
        return self

    def to_toml(self, indent=''):
        """Recursive TOML output, with descriptions as comments."""
        return '\n'.join(self._toml_lines('dc3'))

    def _toml_lines(self, section):
        lines, sub = [f'[{section}]'], []
        for name, f in type(self).model_fields.items():
            v = getattr(self, name)
            if isinstance(v, BaseModel):
                sub.append((f'{section}.{name}', v, f.description))
                continue
            if f.description:
                lines.append(f'# {f.description}')
            lines.append(f'{name} = {_toml_value(v)}')
        for sec, model, descr in sub:
            lines.append('')
            if descr:
                lines.append(f'# {descr}')
            lines.append(f'[{sec}]')
            for name, f in type(model).model_fields.items():
                if f.description:
                    lines.append(f'# {f.description}')
                lines.append(f'{name} = {_toml_value(getattr(model, name))}')
        return lines


def _toml_value(v):
    """Render a value as TOML.  Note bool must precede int: bool IS an int."""
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, Path):
        return f'"{v}"'
    if isinstance(v, list):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    return f'"{v}"'


if __name__ == '__main__':
    print('=' * 74)
    print('1. Nesting and default_factory')
    print('=' * 74)
    p = DC3Par()
    print('nested defaults resolve :', p.template.velscale_ratio, p.fit.method)
    print('default_factory (cwd)   :', p.output_dir)
    print('independent instances   :',
          DC3Par().template is not DC3Par().template)

    print()
    print('=' * 74)
    print('2. Nested construction from a plain dict (what TOML parsing yields)')
    print('=' * 74)
    cfg = {'template': {'velscale_ratio': 4}, 'fit': {'method': 'de', 'moments': 2}}
    q = DC3Par(**cfg)
    print('from dict:', q.template.velscale_ratio, q.fit.method, q.fit.moments)
    try:
        DC3Par(template={'velscale_ratio': 0})
        print('  NOT CAUGHT: nested validation did not fire')
    except ValidationError as e:
        print('  nested validation fires:', e.errors()[0]['loc'], e.errors()[0]['type'])
    try:
        DC3Par(fit={'method': 'amoeba'})
        print('  NOT CAUGHT: Literal options not enforced')
    except ValidationError as e:
        print('  Literal options enforced:', e.errors()[0]['type'])

    print()
    print('=' * 74)
    print('3. Cross-parameter validation (science question 5)')
    print('=' * 74)
    try:
        DC3Par(fit={'moments': 4}, template={'sigma_floor': 5.0})
        print('  NOT CAUGHT: h3/h4 allowed with unmatched resolution')
    except ValidationError as e:
        print('  FAILS as required:', e.errors()[0]['msg'][:88])
    print('  h3/h4 with matched resolution is fine:',
          DC3Par(fit={'moments': 4}).fit.moments)

    print()
    print('=' * 74)
    print('4. Nested FITS header round-trip')
    print('=' * 74)
    hdr = fits.Header()
    hdr['DC3C'] = f'{DC3Par.__module__}.{DC3Par.__name__}'
    hdr['DC3D'] = q.model_dump_json()
    print('card length:', len(hdr['DC3D']))
    back = DC3Par.model_validate_json(hdr['DC3D'])
    print('round-trip equal:', back == q)
    print('  (note Path survives:', type(back.output_dir).__name__, ')')

    print()
    print('=' * 74)
    print('5. Recursive TOML output')
    print('=' * 74)
    print(q.to_toml())

    print()
    print('=' * 74)
    print('6. Re-reading that TOML with the stdlib')
    print('=' * 74)
    import tomllib
    parsed = tomllib.loads(q.to_toml())
    r = DC3Par(**parsed['dc3'])
    print('parsed keys   :', list(parsed['dc3'].keys()))
    print('round-trips   :', r == q)
