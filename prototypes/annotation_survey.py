"""
Survey: how much type information can FuncPar actually recover from the
third-party functions dc3 intends to wrap?

FuncPar infers a parameter's dtype from the wrapped function's own type
annotations.  This measures what that yields in practice.  The answer determines
how much of FuncPar's value is type checking versus keyword tracking, and it is
the same answer for a hand-rolled ParSet and for a pydantic create_model, since
both read the identical annotations.

Run:  python annotation_survey.py
"""
import inspect
import typing


def survey(func, label):
    """Report the annotated fraction of a function's keyword arguments."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        print(f'{label:<46} SIGNATURE UNAVAILABLE')
        return None
    kwargs = [
        k for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    ]
    try:
        hints = typing.get_type_hints(func)
    except (NameError, TypeError):
        hints = {}
    annotated = [k for k in kwargs if k in hints]
    n, m = len(annotated), len(kwargs)
    frac = f'{n}/{m}' if m else '0/0'
    flag = 'OK  ' if m and n == m else ('some' if n else 'NONE')
    print(f'{label:<46} {flag}  {frac:>7}  annotated kwargs')
    return n, m


if __name__ == '__main__':
    targets = []

    from ppxf import ppxf_util
    targets += [
        (ppxf_util.varsmooth, 'ppxf_util.varsmooth'),
        (ppxf_util.losvd_rfft, 'ppxf_util.losvd_rfft'),
        (ppxf_util.log_rebin, 'ppxf_util.log_rebin'),
    ]

    import scipy.optimize
    import scipy.signal
    targets += [
        (scipy.optimize.least_squares, 'scipy.optimize.least_squares'),
        (scipy.optimize.differential_evolution, 'scipy.optimize.differential_evolution'),
        (scipy.optimize.nnls, 'scipy.optimize.nnls'),
        (scipy.signal.windows.tukey, 'scipy.signal.windows.tukey'),
    ]

    import numpy.polynomial.legendre as L
    targets += [(L.Legendre.fit, 'numpy.polynomial.legendre.Legendre.fit')]

    from astropy.stats import sigma_clip
    targets += [(sigma_clip, 'astropy.stats.sigma_clip')]

    print('=' * 74)
    print('Annotated keyword arguments, by wrapped function')
    print('=' * 74)
    tot_n = tot_m = 0
    for f, label in targets:
        r = survey(f, label)
        if r:
            tot_n += r[0]
            tot_m += r[1]
    print('-' * 74)
    print(f'{"TOTAL":<46} {tot_n}/{tot_m} '
          f'({100 * tot_n / tot_m:.0f}% annotated)' if tot_m else 'none')

    print()
    print('=' * 74)
    print('What FuncPar recovers for the two critical-path ppxf functions')
    print('=' * 74)
    for f, label in targets[:2]:
        print(f'\n{label}')
        print(f'  signature : {inspect.signature(f)}')
        print(f'  hints     : {typing.get_type_hints(f)}')
        print(f'  docstring : {"present" if f.__doc__ else "ABSENT"}')
