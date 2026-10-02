r"""
Exploration: is the excess width of Step 1 specific to ``varsmooth``?

Characterization 2 finds that convolving with ``varsmooth`` broadens a line by
about :math:`\Delta^2/6` in variance beyond quadrature addition, even when the
line and the kernel are both well sampled.  This compares three convolutions of
the same comb of pixel-integrated lines:

``varsmooth``
    :func:`dc3.core.resolution.apply_kernel`, as production uses it: the
    vector path of ``ppxf_util.varsmooth``, which stretches the coordinate,
    interpolates onto the stretched grid, convolves by FFT with the analytic
    Gaussian transform, and interpolates back.  Run at ``oversample`` of 1
    and 4.  Production requires at least 2, a decision this exploration
    informed; 1 is run through ``varsmooth`` directly, as
    :func:`varsmooth_vector` describes.
``varsmooth scalar``
    ``ppxf_util.varsmooth`` given a scalar dispersion: a single FFT
    convolution with the analytic transform, with no stretch and no
    interpolation.  Only possible for a uniform kernel.
``gaussian_filter1d``
    ``ppxf_util.gaussian_filter1d``: a direct sum over a Gaussian sampled at
    the pixel offsets, with no interpolation.  Deprecated upstream in favour of
    ``varsmooth``.

Each is run with a **uniform** kernel at a range of widths, where all three
apply, and with the **varying** kernel of characterization 2, which rises
geometrically from 0.1 to 10 pixels.  The excess is the change in variance
beyond quadrature addition, in pixels squared, measured two ways: from the
pre-pixelized width of a pixel-integrated Gaussian fit,
:math:`\sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k^2)`, and from the second
moment of the samples, the change in their variance less :math:`k^2`.

**Predictions.** Both convolutions without interpolation act on the samples as
a discrete kernel, and so add that kernel's variance to the second moment
exactly.  For ``varsmooth``'s vector path the discrete kernel is the hat
function of linear interpolation, :math:`\Lambda`, convolved with the Gaussian
and sampled at whole pixels; by Poisson summation its variance exceeds
:math:`k^2` by

.. math::

    \frac{1}{6} - \frac{1}{\pi^2}\sum_{p \geq 1} \frac{e^{-2\pi^2 p^2 k^2}}{p^2}
    + \frac{1}{6 (mD)^2},

the last term being the return interpolation from a stretched grid of spacing
:math:`1/(mD)` pixels, with :math:`m` the oversampling and :math:`D =
k_{\rm max}/k`.  The sum takes the excess to zero as :math:`k \to 0`, where
sampling the hat function at whole pixels returns a delta function.  For
``gaussian_filter1d`` the discrete kernel is the Gaussian sampled at whole
pixels and normalized, whose variance falls short of :math:`k^2` once the
Gaussian is undersampled.  Neither prediction describes ``varsmooth`` with a
uniform kernel, where the stretched grid lines up with the pixels.

This is an exploratory script, not one of the characterizations: it is not run
by ``make figures``, and writes to ``doc/figures/exploration/`` by default.
"""

from pathlib import Path

from astropy.table import Table, vstack
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np
from ppxf import ppxf_util

from dc3.core import lsf, resolution

import characterize_common


SIGMA_IN = [0.5, 1.0, 2.0]
"""The input pre-pixelized widths, in pixels."""

UNIFORM_K = np.geomspace(0.1, 10.0, 13)
"""The uniform kernel widths, in pixels."""

KERNEL_RANGE = (0.1, 10.0)
"""The range of the varying kernel, in pixels."""

METHODS = ['varsmooth, oversample = 1', 'varsmooth, oversample = 4', 'varsmooth scalar',
           'gaussian_filter1d']
"""The convolutions compared, in plotting order."""


def predicted_varsmooth_excess(k, oversample, dynamic_range):
    r"""
    Return the predicted excess variance of ``varsmooth``'s vector path.

    See the module docstring for the derivation.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        The local kernel dispersion, in pixels.
    oversample : int
        The ``varsmooth`` oversampling, :math:`m`.
    dynamic_range : :class:`numpy.ndarray`
        The local dynamic range, :math:`D = k_{\rm max}/k`.

    Returns
    -------
    :class:`numpy.ndarray`
        The excess of the second moment over quadrature addition, in pixels
        squared.
    """
    p = np.arange(1, 101, dtype=float)[:, None]
    aliased = np.sum(np.exp(-2 * np.square(np.pi * p * np.atleast_1d(k))) / np.square(p), axis=0)
    return 1 / 6 - aliased / np.pi ** 2 + 1 / (6 * np.square(oversample * dynamic_range))


def predicted_sampled_gaussian_excess(k):
    """
    Return the predicted excess variance of ``gaussian_filter1d``.

    The variance of a Gaussian of dispersion ``k`` sampled at whole pixels and
    normalized to unit sum, less ``k**2``.

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


def second_moment(flux, centers, halfwindow):
    """
    Return the variance of the samples about each line.

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
    :class:`numpy.ndarray`
        The second central moment of the samples about each line, in pixels
        squared.
    """
    x = np.arange(flux.size, dtype=float)
    out = np.empty(centers.size, dtype=float)
    for i, (c, h) in enumerate(zip(centers, halfwindow)):
        window = np.absolute(x - c) <= h
        y = flux[window]
        mean = np.sum(y * x[window]) / np.sum(y)
        out[i] = np.sum(y * np.square(x[window] - mean)) / np.sum(y)
    return out


def match(kernel):
    """
    Return a :class:`~dc3.core.resolution.ResolutionMatch` applying a kernel.

    The velocity scale is one, so the kernel in km/s is the kernel in pixels.

    Parameters
    ----------
    kernel : :class:`numpy.ndarray`
        The kernel dispersion at each pixel, in pixels.

    Returns
    -------
    :class:`~dc3.core.resolution.ResolutionMatch`
        The match.
    """
    return resolution.ResolutionMatch(kernel, 0.0, np.zeros(kernel.size, dtype=bool), 1.0, 0.1)


def varsmooth_vector(flux, kernel, oversample):
    """
    Convolve by ``varsmooth``'s vector path, as production does, at any oversampling.

    For an oversampling of 2 or more this is
    :func:`dc3.core.resolution.apply_kernel`.  That refuses 1, which production
    does not allow; for 1 this calls ``varsmooth`` directly, in pixel
    coordinates and with the same workaround for its uniform-kernel defect, so
    the comparison the explorations draw with an oversampling of 1 is like for
    like.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The spectrum.
    kernel : :class:`numpy.ndarray`
        The kernel dispersion at each pixel, in pixels.
    oversample : int
        The oversampling.

    Returns
    -------
    :class:`numpy.ndarray`
        The convolved spectrum.
    """
    if oversample >= 2:
        return resolution.apply_kernel(flux, match(kernel), oversample=oversample)
    sig = resolution._break_kernel_uniformity(kernel, oversample=oversample)
    return ppxf_util.varsmooth(
        np.arange(flux.size, dtype=float), flux, sig, oversample=oversample
    )


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
        return varsmooth_vector(flux, kernel, int(method.split('=')[1]))
    if method == 'varsmooth scalar':
        if not np.all(kernel == kernel[0]):
            return None
        return ppxf_util.varsmooth(np.arange(flux.size, dtype=float), flux, float(kernel[0]))
    return ppxf_util.gaussian_filter1d(flux, kernel)


def measure(npix, kernel, sigma_in, case):
    """
    Convolve a comb by every method and measure the excess of each line.

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
    # fitting windows, and incommensurate with the pixels
    spacing = max(61.37, 18.37 * np.sqrt(sigma_in ** 2 + np.amax(kernel) ** 2))
    centers = characterize_common.comb_centers(npix, spacing, 2.5 * spacing)
    flux = lsf.gaussian_comb(npix, centers, sigma_in)
    k = np.interp(centers, np.arange(npix), kernel)
    expected = np.sqrt(sigma_in ** 2 + k ** 2)
    # The moments are taken over 8 dispersions of the output line, which holds
    # all but a negligible part of it, and stay clear of the neighbouring lines
    halfwindow = np.minimum(8 * expected + 4, spacing / 2 - 1)
    moment_in = second_moment(flux, centers, halfwindow)
    tables = []
    for method in METHODS:
        convolved = convolve(flux, kernel, method)
        if convolved is None:
            continue
        _, sigma_out = lsf.measure_comb(convolved, centers, expected)
        tables.append(Table({
            'case': np.full(centers.size, case),
            'method': np.full(centers.size, method),
            'sigma_in': np.full(centers.size, sigma_in),
            'kernel': k,
            'excess': np.square(sigma_out) - np.square(expected),
            'moment_excess': (
                second_moment(convolved, centers, halfwindow) - moment_in - np.square(k)
            ),
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
        npix = 60000
        tables.append(measure(npix, np.geomspace(*KERNEL_RANGE, npix), sigma_in, 'varying'))
    return vstack(tables)


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


def plot(table, fit_label='Gaussian fit'):
    r"""
    Plot the excess variance of each method.

    Three rows: the uniform kernels against the kernel width; the varying
    kernel against the kernel width; and the varying kernel against the local
    dynamic range, :math:`D = k_{\rm max}/k`, the widest kernel in the
    spectrum relative to the one at each line, which is what sets
    ``varsmooth``'s internal sample spacing there.  A uniform kernel has
    :math:`D = 1` everywhere, so only the varying kernel has a third row.

    Solid lines, with their 16th-84th percentile bands, are the excess measured
    by the fit in the ``excess`` column; dotted lines the excess of the second
    moment; dashed lines the predictions of the module docstring.  The
    ``varsmooth`` prediction is drawn only for the varying kernel.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`, or a table with the same columns.
    fit_label : str, optional
        What the ``excess`` column was measured by, for the legend and title.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    # Each row: the kernel case, and whether it is plotted against the kernel
    # width ('k') or the local dynamic range ('D')
    rows = [('uniform', 'k'), ('varying', 'k'), ('varying', 'D')]
    fig, axes = pyplot.subplots(
        len(rows), len(SIGMA_IN), figsize=(10.0, 9.5), sharey=True, layout='constrained'
    )
    kmax = KERNEL_RANGE[1]
    edges = {
        'k': np.geomspace(*KERNEL_RANGE, 21),
        'D': np.geomspace(1.0, kmax / KERNEL_RANGE[0], 21),
    }
    # The kernel widths at which the predictions are drawn, and their local
    # dynamic range in the varying kernel
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

                # The predictions, against whichever variable the row uses
                xgrid = kgrid if xaxis == 'k' else dynamic_range
                if case == 'varying' and method.startswith('varsmooth, oversample'):
                    m = int(method.split('=')[1])
                    ax.plot(
                        xgrid, predicted_varsmooth_excess(kgrid, m, dynamic_range),
                        color=f'C{i}', ls='--', lw=1.0
                    )
                elif method == 'gaussian_filter1d':
                    ax.plot(
                        xgrid, predicted_sampled_gaussian_excess(kgrid),
                        color=f'C{i}', ls='--', lw=1.0
                    )
            ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
            ax.set_xscale('log')
            if xaxis == 'k':
                ax.axhline(1 / 6, color='0.5', ls='-.', lw=1.0, zorder=0)
                ax.set_xticks([0.1, 0.3, 1, 3, 10], labels=['0.1', '0.3', '1', '3', '10'])
                ax.set_xlabel('Kernel dispersion, $k$ (pixels)')
            else:
                ax.set_xticks([1, 3, 10, 30, 100], labels=['1', '3', '10', '30', '100'])
                ax.set_xlabel(r'Local dynamic range, $D = k_{\rm max}/k$')
            if row == 0:
                ax.set_title(
                    rf'$\sigma_{{\rm in}}$ = {sigma_in:g} pixel{"s" if sigma_in > 1 else ""}'
                )
            if col == 0:
                ax.set_ylabel(
                    f'{case} kernel\n'
                    r'$\sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k^2)$ (pixels$^2$)'
                )
    # The methods by color, and the kinds of curve by line style
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color='k', ls='-', label=fit_label),
        Line2D([], [], color='k', ls=':', lw=1.5, label='second moment'),
        Line2D([], [], color='k', ls='--', lw=1.0, label='prediction'),
        Line2D([], [], color='0.5', ls='-.', lw=1.0, label=r'$\Delta^2/6$'),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(
        f'Excess variance of three convolutions beyond quadrature addition: {fit_label} '
        '(median and 16th-84th percentiles over pixel phase),\n'
        'second moment of the samples (median), and the predicted excess of the second moment',
        fontsize=9
    )
    return fig


def main():
    """Run the exploration and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.set_defaults(outdir=Path(__file__).resolve().parents[1] / 'figures' / 'exploration')
    args = parser.parse_args()
    characterize_common.apply_style()
    table = run()
    for case in ['uniform', 'varying']:
        for method in METHODS:
            for sigma_in in SIGMA_IN:
                indx = (
                    (table['case'] == case) & (table['method'] == method)
                    & (table['sigma_in'] == sigma_in) & (table['kernel'] > 1.0)
                )
                if not np.any(indx):
                    continue
                print(
                    f'{case:8s} {method:27s} sigma_in = {sigma_in:g}: median excess for k > 1 px '
                    f'= {np.median(table["excess"][indx]):+.4f} px^2'
                )
    characterize_common.save(plot(table), table, 'convolution_methods', args)


if __name__ == '__main__':
    main()
