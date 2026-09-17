#!/usr/bin/env python3
"""
Minimal reproduction of an off-by-one in ``ppxf_util.varsmooth``.

When the kernel width in pixels, ``sig_x/np.gradient(x)``, is *exactly* uniform,
``varsmooth`` builds its internal stretched grid with one sample fewer than the
input, and the interpolation onto that shortened grid and back broadens the
result.  A requested 0.5-pixel kernel is applied as 0.87 pixels.

The defect is independent of the kernel width, and is not a sampling effect: a
perturbation of one part in 1e-12 to a single element of ``sig_x`` -- which
changes nothing physical -- restores the correct answer exactly.

Depends only on ``numpy`` and ``ppxf``.

Run:  python ppxf_varsmooth_offbyone.py
"""

import numpy as np
from ppxf import ppxf_util


NPIX = 800
SIGMA_IN = 4.0          # width of the probe line, in pixels; comfortably resolved
INDEX = np.arange(NPIX, dtype=float)


def probe_line():
    """
    Return a well-resolved Gaussian to convolve.

    A resolved line rather than a delta function, so that the measurement below
    is not dominated by interpolation ringing.

    Returns
    -------
    numpy.ndarray
        The probe spectrum.
    """
    return np.exp(-0.5 * np.square((INDEX - NPIX / 2) / SIGMA_IN))


def realised_width(convolved):
    """
    Measure the kernel width that was actually applied.

    The second moment of the output, differenced in quadrature against the
    input, which is the definition the kernel is supposed to satisfy.

    Parameters
    ----------
    convolved : numpy.ndarray
        The convolved spectrum.

    Returns
    -------
    float
        The realised kernel width, in pixels.
    """
    mean = np.sum(convolved * INDEX) / np.sum(convolved)
    width = np.sqrt(np.sum(convolved * np.square(INDEX - mean)) / np.sum(convolved))
    return np.sqrt(max(width ** 2 - SIGMA_IN ** 2, 0.0))


def internal_grid_size(x, sig_x, oversample=1):
    """
    Reproduce the internal grid size that ``varsmooth`` computes.

    These four lines are quoted from ``ppxf_util.varsmooth`` solely to show
    where the count goes wrong; they are not a reimplementation of the routine.

    Parameters
    ----------
    x : numpy.ndarray
        The abscissa.
    sig_x : numpy.ndarray
        The kernel width, in the units of ``x``.
    oversample : int, optional
        As passed to ``varsmooth``.

    Returns
    -------
    tuple
        The span of the stretched coordinate, the internal sample count ``n``
        that ``varsmooth`` uses, and whether the pixel-space kernel is exactly
        uniform.
    """
    sig = sig_x / np.gradient(x)
    sig = sig.clip(0.1)
    sig_max = np.max(sig) * oversample
    xs = np.cumsum(sig_max / sig)
    span = xs[-1] - xs[0]
    return span, int(np.ceil(span)), bool(np.all(sig == sig[0]))


def report(label, x, sig_x, want, oversample=1):
    """
    Convolve, measure, and print one case.

    Parameters
    ----------
    label : str
        Description of the case.
    x : numpy.ndarray
        The abscissa.
    sig_x : numpy.ndarray, float
        The kernel width, in the units of ``x``.
    want : float
        The kernel width that should result, in pixels.
    oversample : int, optional
        As passed to ``varsmooth``.
    """
    out = ppxf_util.varsmooth(x, probe_line(), sig_x, oversample=oversample)
    got = realised_width(out)
    if np.isscalar(sig_x):
        span, n, uniform = float('nan'), NPIX, True
        grid = f'{"n/a (scalar branch)":>25}'
    else:
        span, n, uniform = internal_grid_size(x, sig_x, oversample)
        grid = f'{str(uniform):>5}  {span - (NPIX - 1) * oversample:12.3e} {n:6d}'
    flag = 'ok ' if np.isclose(got, want, rtol=0.02) else 'BUG'
    print(f'  {flag}  {label:<44} {grid}   want {want:.3f}, got {got:.3f} px')


def main():
    """Run every case and print the diagnosis."""
    print(__doc__.strip().splitlines()[0])
    print(
        f'\nInput length {NPIX}.  "unif" is whether sig_x/gradient(x) is exactly '
        'constant;\nthe next column is how far the stretched span exceeds '
        '(N-1)*oversample, which is\nwhat decides whether ceil() rounds up to the '
        'correct sample count.'
    )
    print(f'\n  {"":4}{"case":<46}{"unif":>5}  {"span - (N-1)k":>12} {"n":>6}')
    print('  ' + '-' * 104)

    # An abscissa whose numpy.gradient is exactly 1.0, so the pixel-space
    # kernel equals sig_x exactly and is therefore exactly uniform.
    x_exact = INDEX.copy()
    # A realistic logarithmic abscissa, whose gradient carries floating-point
    # noise, so the pixel-space kernel is not exactly uniform.
    x_log = np.log10(3800.0) + 1.09e-5 * INDEX

    print('\n  Well above the 0.1-pixel clip, so undersampling is not in play:')
    report('uniform 0.5 px, exactly uniform abscissa', x_exact, np.full(NPIX, 0.5), 0.5)

    perturbed = np.full(NPIX, 0.5)
    perturbed[NPIX // 2] *= 1 - 1e-12
    report('  ... one interior element changed by 1e-12', x_exact, perturbed, 0.5)
    report('  ... or the same request on a log abscissa', x_log, np.full(NPIX, 0.5 * 1.09e-5), 0.5)

    print('\n  Oversampling does not help; it trades one error for another:')
    for oversample in [2, 4, 8]:
        report(
            f'uniform 0.5 px, oversample={oversample}', x_exact, np.full(NPIX, 0.5), 0.5,
            oversample=oversample
        )

    print('\n  Below the clip, which forces exact uniformity and so always triggers it:')
    for requested in [0.001, 0.05, 0.09]:
        report(f'uniform {requested} px on a log abscissa', x_log,
               np.full(NPIX, requested * 1.09e-5), requested)
    report('uniform 0.1 px on a log abscissa (at the clip)', x_log,
           np.full(NPIX, 0.1 * 1.09e-5), 0.1)

    print('\n  Control: the scalar sig_x branch builds no stretched grid at all:')
    report('scalar sig_x = 0.5', x_exact, 0.5, 0.5)

    print(
        '\n'
        'Diagnosis\n'
        '---------\n'
        '  xs      = np.cumsum(sig_max/sig)        has len(x) entries\n'
        '  span    = xs[-1] - xs[0]                covers len(x) - 1 unit intervals\n'
        '  n       = int(np.ceil(span))            <-- one short\n'
        '\n'
        '  Because sig_max/sig >= 1 by construction, span >= len(x) - 1, with\n'
        '  equality if and only if sig is exactly uniform.  In that case ceil()\n'
        '  returns len(x) - 1, so x_new holds one sample fewer than the input and\n'
        '  the round trip through interp() loses resolution.\n'
        '\n'
        '  A range spanning S unit intervals needs S + 1 samples, so:\n'
        '\n'
        '      n = int(np.ceil(xs[-1] - xs[0])) + 1\n'
        '\n'
        '  In the generic, non-uniform case this adds one sample to a grid that is\n'
        '  already denser than the input, which costs nothing; in the uniform case\n'
        '  it restores the identity mapping, which is what the rows marked "ok"\n'
        '  above demonstrate.\n'
        '\n'
        '  The 0.1-pixel clip is only the trigger: it replaces every smaller value\n'
        '  with the same literal, making sig exactly uniform.  That is why any\n'
        '  request below 0.1 px is applied as ~0.71 px rather than as 0.1 px.'
    )


if __name__ == '__main__':
    main()
