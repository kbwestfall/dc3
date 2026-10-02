r"""
Characterization 2: what Step 1, resolution matching, does to the pre-pixelized width.

Step 1 convolves each template with a Gaussian kernel of known, varying width,
by :func:`dc3.core.resolution.apply_kernel`, which wraps the vector path of
``ppxf_util.varsmooth``.  The ``mangadap`` assumption is that this changes the
pre-pixelized dispersion by exactly the kernel, in quadrature,
:math:`\sigma_{\rm out}^2 = \sigma_{\rm in}^2 + k^2`.  This measures the
departure from that assumption, on one grid so that no resampling is involved,
and compares it with an exact prediction and with two convolutions that do not
interpolate.

**The convolutions compared.**

``varsmooth, oversample = m``
    ``apply_kernel`` as production uses it, at :math:`m` of 2 and 4.  It
    stretches the coordinate so that the kernel becomes uniform, interpolates
    the spectrum linearly onto the stretched grid, convolves by FFT with the
    analytic Gaussian transform, and interpolates back.  :math:`m = 1`, which
    production refuses, is run through ``varsmooth`` directly in the same way,
    for comparison.
``varsmooth scalar``
    ``ppxf_util.varsmooth`` given a scalar dispersion: one FFT convolution, with
    no stretch and no interpolation.  Only possible for a uniform kernel, and
    never used in production.
``gaussian_filter1d``
    ``ppxf_util.gaussian_filter1d``: a direct sum over a Gaussian sampled at
    whole-pixel offsets.  Deprecated upstream.

Each is run on a comb of pixel-integrated lines, of pre-pixelized width
:math:`\sigma_{\rm in}` = 0.5, 1 and 2 pixels, with a **uniform** kernel at a
range of widths and with a **varying** kernel that rises geometrically from
0.1 to 10 pixels along the spectrum.  The line spacing is incommensurate with
the pixels, so each comb samples every pixel phase.

**The measurements.**  The excess variance beyond quadrature addition, in
pixels squared, is measured three ways:

- from the pre-pixelized width of a pixel-integrated Gaussian fit to each
  output line, :math:`\sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k^2)`;
- from the second moment of the samples, the change in their variance less
  :math:`k^2`;
- from a kinematic-style fit, which convolves the input line with a Gaussian of
  free dispersion :math:`s` by its analytic transform and fits the output line,
  :math:`s^2 - k^2`.

The shift of each line is measured from the kinematic-style fit and from the
first moment.

**The prediction.**  Both convolutions without interpolation act on the
samples as a discrete kernel, so the second moment gains that kernel's
variance exactly.  For ``varsmooth``'s vector path the discrete kernel is the
hat function of linear interpolation convolved with the Gaussian and sampled at
whole pixels, plus the return interpolation from a stretched grid of spacing
:math:`1/(mD)` pixels, where :math:`D = k_{\rm max}/k` is the widest kernel in
the spectrum relative to the local one.  By Poisson summation its variance
exceeds :math:`k^2` by

.. math::

    \frac{1}{6} - \frac{1}{\pi^2}\sum_{p \geq 1} \frac{e^{-2\pi^2 p^2 k^2}}{p^2}
    + \frac{1}{6 (mD)^2}.

For ``gaussian_filter1d`` the discrete kernel is the sampled, normalized
Gaussian, whose variance falls short of :math:`k^2` when it is undersampled.  A
kernel varying along the spectrum spreads a line more on its wider side,
shifting its first moment by :math:`2kk'`.

These results were first reached in the exploratory scripts
``explore_convolution_methods.py``, ``explore_kernel_dynamic_range.py`` and
``explore_kernel_fitting.py``.
"""

from astropy.table import Table, vstack
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np
from ppxf import ppxf_util
from scipy import fft, optimize

from dc3.core import lsf, resolution

import characterize_common


SIGMA_IN = [0.5, 1.0, 2.0]
"""The input pre-pixelized widths, in pixels."""

UNIFORM_K = np.geomspace(0.1, 10.0, 13)
"""The uniform kernel widths, in pixels."""

KERNEL_RANGE = (0.1, 10.0)
"""The range of the varying kernel, in pixels."""

VARYING_NPIX = 60000
"""The length of the spectrum carrying the varying kernel, in pixels."""

METHODS = ['varsmooth, oversample = 1', 'varsmooth, oversample = 2',
           'varsmooth, oversample = 4', 'varsmooth scalar', 'gaussian_filter1d']
"""The convolutions compared, in plotting order."""


# ----------------------------------------------------------------------
# The convolutions
# ----------------------------------------------------------------------
def convolve(flux, kernel, method):
    """
    Convolve a spectrum by one of the methods compared.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The spectrum.
    kernel : :class:`numpy.ndarray`
        The kernel dispersion at each pixel, in pixels.
    method : str
        One of :data:`METHODS`.

    Returns
    -------
    :class:`numpy.ndarray`
        The convolved spectrum, or None if the method does not apply to this
        kernel.
    """
    if method.startswith('varsmooth, oversample'):
        m = int(method.split('=')[1])
        if m >= 2:
            match = resolution.ResolutionMatch(
                kernel, 0.0, np.zeros(kernel.size, dtype=bool), 1.0, 0.1
            )
            return resolution.apply_kernel(flux, match, oversample=m)
        # Below the production minimum: varsmooth directly, in pixel
        # coordinates and with apply_kernel's workaround, so the comparison is
        # like for like
        sig = resolution._break_kernel_uniformity(kernel, oversample=m)
        return ppxf_util.varsmooth(np.arange(flux.size, dtype=float), flux, sig, oversample=m)
    if method == 'varsmooth scalar':
        if not np.all(kernel == kernel[0]):
            return None
        return ppxf_util.varsmooth(np.arange(flux.size, dtype=float), flux, float(kernel[0]))
    return ppxf_util.gaussian_filter1d(flux, kernel)


# ----------------------------------------------------------------------
# The predictions
# ----------------------------------------------------------------------
def predicted_sampled_gaussian_excess(k):
    """
    Return the predicted excess variance of ``gaussian_filter1d``.

    The variance of a Gaussian sampled at whole pixels and normalized to unit
    sum, less ``k**2``.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        The kernel dispersion, in pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The excess, in pixels squared; negative where the Gaussian is
        undersampled.
    """
    n = np.arange(-200, 201, dtype=float)[:, None]
    w = np.exp(-0.5 * np.square(n / np.atleast_1d(k)))
    return np.sum(np.square(n) * w, axis=0) / np.sum(w, axis=0) - np.square(k)


def predicted_shift(k):
    r"""
    Return the predicted first-moment shift of a line under the varying kernel.

    :math:`2kk'`, with :math:`k' = k \ln(k_{\rm max}/k_{\rm min})/(N - 1)`
    for the geometric kernel of :data:`KERNEL_RANGE` over :data:`VARYING_NPIX`
    pixels.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        The local kernel dispersion, in pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The shift, in pixels, towards the wider kernel.
    """
    kmin, kmax = KERNEL_RANGE
    return 2 * np.square(k) * np.log(kmax / kmin) / (VARYING_NPIX - 1)


# ----------------------------------------------------------------------
# The measurements
# ----------------------------------------------------------------------
def moments(flux, centers, halfwindow):
    """
    Return the first and second central moments of the samples about each line.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The spectrum.
    centers : :class:`numpy.ndarray`
        The line centres, in pixels.
    halfwindow : :class:`numpy.ndarray`
        The half-width of the window over which each moment is taken, in
        pixels.

    Returns
    -------
    tuple
        The centroid and the variance of each line, in pixels and pixels
        squared.
    """
    x = np.arange(flux.size, dtype=float)
    mean = np.empty(centers.size, dtype=float)
    var = np.empty(centers.size, dtype=float)
    for i, (c, h) in enumerate(zip(centers, halfwindow)):
        window = np.absolute(x - c) <= h
        y = flux[window]
        mean[i] = np.sum(y * x[window]) / np.sum(y)
        var[i] = np.sum(y * np.square(x[window] - mean[i])) / np.sum(y)
    return mean, var


def broaden(template, sigma, shift, npad):
    r"""
    Convolve a template with a shifted Gaussian, by its analytic transform.

    Parameters
    ----------
    template : :class:`numpy.ndarray`
        The template segment.
    sigma : float
        The dispersion of the Gaussian, in pixels.
    shift : float
        The shift, in pixels, positive towards larger index.
    npad : int
        The length of the zero-padded transform, at least twice the segment.

    Returns
    -------
    :class:`numpy.ndarray`
        The broadened template, the same length as ``template``.
    """
    omega = 2 * np.pi * fft.rfftfreq(npad)
    kernel = np.exp(-0.5 * np.square(omega * sigma) - 1j * omega * shift)
    return fft.irfft(fft.rfft(template, npad) * kernel, npad)[:template.size]


def _kernel_residuals(par, template, observed, npad):
    """
    Return the residuals of the broadened template from the observation.

    Parameters
    ----------
    par : :class:`numpy.ndarray`
        The amplitude, dispersion and shift.
    template, observed : :class:`numpy.ndarray`
        The template and observed segments.
    npad : int
        The length of the zero-padded transform.

    Returns
    -------
    :class:`numpy.ndarray`
        The residuals.
    """
    return par[0] * broaden(template, par[1], par[2], npad) - observed


def fit_kernel(observed, template, center, sigma_guess, halfwindow):
    """
    Fit the Gaussian kernel that best relates a template line to an observed one.

    This is how a kinematic fit measures a broadening: it convolves a template
    with a Gaussian and adjusts the Gaussian, rather than fitting a profile to
    the observed line.  The analytic transform applies a discrete kernel of
    variance exactly ``sigma**2``.

    Parameters
    ----------
    observed, template : :class:`numpy.ndarray`
        The observed and template spectra, on the same grid.
    center : float
        The line centre, in pixels.
    sigma_guess : float
        The initial guess for the kernel dispersion, in pixels.
    halfwindow : int
        The half-width of the fitted segment, in pixels.

    Returns
    -------
    tuple
        The fitted kernel dispersion and shift, in pixels.
    """
    c = int(np.round(center))
    lo, hi = max(c - halfwindow, 0), min(c + halfwindow + 1, observed.size)
    seg_t, seg_o = template[lo:hi], observed[lo:hi]
    npad = fft.next_fast_len(2 * seg_t.size)
    result = optimize.least_squares(
        _kernel_residuals, [1.0, max(sigma_guess, 0.05), 0.0], args=(seg_t, seg_o, npad),
        bounds=([0.0, 0.0, -halfwindow / 2], [np.inf, np.inf, halfwindow / 2]),
        x_scale='jac'
    )
    return result.x[1], result.x[2]


def measure(npix, kernel, sigma_in, case):
    """
    Convolve a comb by every method and measure each line.

    Parameters
    ----------
    npix : int
        The length of the spectrum.
    kernel : :class:`numpy.ndarray`
        The kernel dispersion at each pixel, in pixels.
    sigma_in : float
        The input pre-pixelized width, in pixels.
    case : str
        ``'uniform'`` or ``'varying'``, recorded in the table.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line and method.
    """
    # Wide enough that the broadest lines do not reach their neighbours'
    # windows, and incommensurate with the pixels
    spacing = max(61.37, 18.37 * np.sqrt(sigma_in ** 2 + np.amax(kernel) ** 2))
    centers = characterize_common.comb_centers(npix, spacing, 2.5 * spacing)
    flux = lsf.gaussian_comb(npix, centers, sigma_in)
    k = np.interp(centers, np.arange(npix), kernel)
    expected = np.sqrt(sigma_in ** 2 + k ** 2)
    # Eight dispersions of the output line hold all but a negligible part of it
    halfwindow = np.minimum(8 * expected + 4, spacing / 2 - 1)
    centroid_in, var_in = moments(flux, centers, halfwindow)
    tables = []
    for method in METHODS:
        convolved = convolve(flux, kernel, method)
        if convolved is None:
            continue
        _, sigma_out = lsf.measure_comb(convolved, centers, expected)
        centroid_out, var_out = moments(convolved, centers, halfwindow)
        kernel_fit = np.array([
            fit_kernel(convolved, flux, c, kk, int(h))
            for c, kk, h in zip(centers, k, halfwindow)
        ])
        tables.append(Table({
            'case': np.full(centers.size, case),
            'method': np.full(centers.size, method),
            'sigma_in': np.full(centers.size, sigma_in),
            'kernel': k,
            'excess': np.square(sigma_out) - np.square(expected),
            'moment_excess': var_out - var_in - np.square(k),
            'kernel_fit_excess': np.square(kernel_fit[:, 0]) - np.square(k),
            'kernel_fit_shift': kernel_fit[:, 1],
            'centroid_shift': centroid_out - centroid_in,
        }))
    return vstack(tables)


def run():
    """
    Measure every method, for uniform kernels and for the varying kernel.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, method and configuration.
    """
    tables = []
    for sigma_in in SIGMA_IN:
        for k in UNIFORM_K:
            tables.append(measure(4000, np.full(4000, k), sigma_in, 'uniform'))
        tables.append(measure(
            VARYING_NPIX, np.geomspace(*KERNEL_RANGE, VARYING_NPIX), sigma_in, 'varying'
        ))
    return vstack(tables)


# ----------------------------------------------------------------------
# The figures
# ----------------------------------------------------------------------
def summarize(x, values, uniform, edges):
    """
    Return the median and 16th/84th percentiles of a measurement.

    Parameters
    ----------
    x : :class:`numpy.ndarray`
        The abscissa of each line.
    values : :class:`numpy.ndarray`
        The measurement for each line.
    uniform : bool
        Each configuration has a single abscissa, so group by its distinct
        values rather than binning.
    edges : :class:`numpy.ndarray`
        The bin edges, used if ``uniform`` is False.

    Returns
    -------
    :class:`astropy.table.Table`
        Columns ``x``, ``median``, ``p16`` and ``p84``.
    """
    if not uniform:
        return characterize_common.binned_stats(x, values, edges)
    groups = np.unique(x)
    return Table({
        'x': groups,
        'median': [np.median(values[x == v]) for v in groups],
        'p16': [np.percentile(values[x == v], 16) for v in groups],
        'p84': [np.percentile(values[x == v], 84) for v in groups],
    })


def _format_axis(ax, xaxis):
    """
    Set the scale, ticks and label of a panel's abscissa.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The panel.
    xaxis : str
        ``'k'`` for the kernel width, ``'D'`` for the local dynamic range.
    """
    ax.set_xscale('log')
    if xaxis == 'k':
        ax.set_xticks([0.1, 0.3, 1, 3, 10], labels=['0.1', '0.3', '1', '3', '10'])
        ax.set_xlabel('Kernel dispersion, $k$ (pixels)')
    else:
        ax.set_xticks([1, 3, 10, 30, 100], labels=['1', '3', '10', '30', '100'])
        ax.set_xlabel(r'Local dynamic range, $D = k_{\rm max}/k$')


def plot(table):
    r"""
    Plot the excess variance of each method.

    Three rows: the uniform kernels against the kernel width, and the varying
    kernel against the kernel width and against the local dynamic range
    :math:`D = k_{\rm max}/k`.  Solid lines, with their 16th-84th percentile
    bands, are the Gaussian-fit excess; dotted lines the second-moment excess;
    dashed lines the predictions.  The ``varsmooth`` prediction is drawn only
    for the varying kernel, since a uniform kernel lines the stretched grid up
    with the pixels.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    rows = [('uniform', 'k'), ('varying', 'k'), ('varying', 'D')]
    fig, axes = pyplot.subplots(
        len(rows), len(SIGMA_IN), figsize=(10.0, 9.5), sharey=True, layout='constrained'
    )
    kmax = KERNEL_RANGE[1]
    edges = {
        'k': np.geomspace(*KERNEL_RANGE, 21),
        'D': np.geomspace(1.0, kmax / KERNEL_RANGE[0], 21),
    }
    kgrid = np.geomspace(*KERNEL_RANGE, 300)
    dynamic_range = kmax / kgrid
    for row, (case, xaxis) in enumerate(rows):
        for col, sigma_in in enumerate(SIGMA_IN):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                indx = (
                    (table['case'] == case) & (table['method'] == method)
                    & (table['sigma_in'] == sigma_in)
                )
                if not np.any(indx):
                    continue
                kernel = np.asarray(table['kernel'][indx])
                x = kernel if xaxis == 'k' else kmax / kernel
                fit = summarize(x, np.asarray(table['excess'][indx]), case == 'uniform',
                                edges[xaxis])
                moment = summarize(x, np.asarray(table['moment_excess'][indx]),
                                   case == 'uniform', edges[xaxis])
                ax.fill_between(
                    fit['x'], fit['p16'], fit['p84'], color=f'C{i}', alpha=0.15, lw=0
                )
                ax.plot(fit['x'], fit['median'], color=f'C{i}', label=method)
                ax.plot(moment['x'], moment['median'], color=f'C{i}', ls=':', lw=1.5)
                xgrid = kgrid if xaxis == 'k' else dynamic_range
                if case == 'varying' and method.startswith('varsmooth, oversample'):
                    m = int(method.split('=')[1])
                    ax.plot(
                        xgrid, resolution.varsmooth_excess(kgrid, m, dynamic_range),
                        color=f'C{i}', ls='--', lw=1.0
                    )
                elif method == 'gaussian_filter1d':
                    ax.plot(
                        xgrid, predicted_sampled_gaussian_excess(kgrid), color=f'C{i}',
                        ls='--', lw=1.0
                    )
            ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
            if xaxis == 'k':
                ax.axhline(1 / 6, color='0.5', ls='-.', lw=1.0, zorder=0)
            _format_axis(ax, xaxis)
            if row == 0:
                ax.set_title(
                    rf'$\sigma_{{\rm in}}$ = {sigma_in:g} pixel{"s" if sigma_in > 1 else ""}'
                )
            if col == 0:
                ax.set_ylabel(
                    f'{case} kernel\n'
                    r'$\sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k^2)$ (pixels$^2$)'
                )
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color='k', ls='-', label='Gaussian fit'),
        Line2D([], [], color='k', ls=':', lw=1.5, label='second moment'),
        Line2D([], [], color='k', ls='--', lw=1.0, label='prediction'),
        Line2D([], [], color='0.5', ls='-.', lw=1.0, label=r'$\Delta^2/6$'),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(
        'Step 1: excess variance beyond quadrature addition: Gaussian fit (median and '
        '16th-84th percentiles over pixel phase),\nsecond moment of the samples (median), '
        'and the predicted excess of the second moment',
        fontsize=9
    )
    return fig


def plot_shift(table):
    r"""
    Plot the shift of each line under the varying kernel.

    A uniform kernel shifts nothing, so only the varying kernel is shown,
    against the kernel width and against the local dynamic range.  Solid lines,
    with their 16th-84th percentile bands, are the kinematic-style fit's shift;
    dotted lines the first-moment shift; the black dashed line the predicted
    first-moment shift, :math:`2kk'`.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    rows = ['k', 'D']
    fig, axes = pyplot.subplots(
        len(rows), len(SIGMA_IN), figsize=(10.0, 6.5), sharey=True, layout='constrained'
    )
    kmax = KERNEL_RANGE[1]
    edges = {
        'k': np.geomspace(*KERNEL_RANGE, 21),
        'D': np.geomspace(1.0, kmax / KERNEL_RANGE[0], 21),
    }
    kgrid = np.geomspace(*KERNEL_RANGE, 300)
    for row, xaxis in enumerate(rows):
        for col, sigma_in in enumerate(SIGMA_IN):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                indx = (
                    (table['case'] == 'varying') & (table['method'] == method)
                    & (table['sigma_in'] == sigma_in)
                )
                if not np.any(indx):
                    continue
                kernel = np.asarray(table['kernel'][indx])
                x = kernel if xaxis == 'k' else kmax / kernel
                fit = summarize(x, np.asarray(table['kernel_fit_shift'][indx]), False,
                                edges[xaxis])
                centroid = summarize(x, np.asarray(table['centroid_shift'][indx]), False,
                                     edges[xaxis])
                ax.fill_between(
                    fit['x'], fit['p16'], fit['p84'], color=f'C{i}', alpha=0.15, lw=0
                )
                ax.plot(fit['x'], fit['median'], color=f'C{i}', label=method)
                ax.plot(centroid['x'], centroid['median'], color=f'C{i}', ls=':', lw=1.5)
            ax.plot(
                kgrid if xaxis == 'k' else kmax / kgrid, predicted_shift(kgrid), color='k',
                ls='--', lw=1.0
            )
            ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
            _format_axis(ax, xaxis)
            if row == 0:
                ax.set_title(
                    rf'$\sigma_{{\rm in}}$ = {sigma_in:g} pixel{"s" if sigma_in > 1 else ""}'
                )
            if col == 0:
                ax.set_ylabel('shift of the line centre (pixels)')
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color='k', ls='-', label='kernel fit'),
        Line2D([], [], color='k', ls=':', lw=1.5, label='first moment'),
        Line2D([], [], color='k', ls='--', lw=1.0, label=r"prediction, $2kk'$"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(
        'Step 1, varying kernel: shift of the line centre, from the kinematic-style kernel '
        'fit (median and 16th-84th percentiles\nover pixel phase) and the first moment of the '
        'samples (median); positive towards larger pixel index, where the kernel is wider',
        fontsize=9
    )
    return fig


def main():
    """Run the characterization and write its products."""
    args = characterize_common.get_parser(__doc__.split('\n')[1]).parse_args()
    characterize_common.apply_style()
    table = run()
    for case in ['uniform', 'varying']:
        for method in METHODS:
            for sigma_in in SIGMA_IN:
                indx = (
                    (table['case'] == case) & (table['method'] == method)
                    & (table['sigma_in'] == sigma_in)
                )
                if not np.any(indx):
                    continue
                parts = []
                for label, (lo, hi) in [('k ~ 0.2', (0.18, 0.22)), ('k ~ 1', (0.9, 1.1)),
                                        ('k ~ 9', (8.0, 10.0))]:
                    sel = indx & (table['kernel'] > lo) & (table['kernel'] < hi)
                    if not np.any(sel):
                        continue
                    parts.append(
                        f'{label}: fit {np.median(table["excess"][sel]):+.4f}, kernel fit '
                        f'{np.median(table["kernel_fit_excess"][sel]):+.4f}, moment '
                        f'{np.median(table["moment_excess"][sel]):+.4f}'
                    )
                print(f'{case:8s} {method:26s} sigma_in = {sigma_in:g}: ' + '; '.join(parts))
    characterize_common.save(plot(table), table, 'matching', args)
    shifts = table[table['case'] == 'varying']
    shifts.keep_columns(
        ['method', 'sigma_in', 'kernel', 'kernel_fit_shift', 'centroid_shift']
    )
    characterize_common.save(plot_shift(table), shifts, 'matching_shift', args)


if __name__ == '__main__':
    main()
