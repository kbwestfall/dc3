r"""
Exploration: the Step-1 excess as a kinematic fit would see it.

The characterizations measure the width of a prepared line by fitting it with a
pixel-integrated Gaussian.  That assumes the effective kernel of the
preparation is Gaussian, which ``explore_convolution_methods.py`` shows it is
not: the excess in the second moment is exactly predictable, but the excess a
Gaussian fit reports depends on the width of the line.

A kinematic fit does something different.  It does not fit a profile to the
observed line; it convolves a *template* line with a Gaussian broadening kernel
and adjusts the kernel until the result matches the observation, so it
measures the kernel that relates the two.  This mimics that.  For each line,
:func:`fit_kernel` takes the line before the convolution under test as the
template and the line after it as the observation, and fits

.. math::

    {\rm model} = a \left[ T \otimes G(s) \right](x - v),

with amplitude :math:`a`, dispersion :math:`s` and shift :math:`v`.  The
convolution is done as a kinematic fit with an analytic broadening function
would do it: the template's discrete Fourier transform is multiplied by the
analytic transform of the shifted Gaussian,
:math:`\exp(-\omega^2 s^2/2 - i\omega v)`.  The discrete kernel this applies
has a variance of exactly :math:`s^2` whatever :math:`s`, so the excess
reported, :math:`s^2 - k^2`, is the variance of the convolution under test
beyond the nominal kernel, as far as a Gaussian broadening of the template can
represent it.

The tests, the predictions and the second moments are those of
``explore_convolution_methods.py``, which this imports; only the fit differs.

A second figure shows the **shift** of each line, which a uniform kernel leaves
at zero but a varying one need not: the fitted :math:`v`, and the change in the
first moment of the samples, both in pixels and positive towards larger index.
A convolution whose kernel dispersion :math:`k(x)` changes along the spectrum
spreads a line more on the side where the kernel is wider; to first order in
the gradient this moves its first moment by

.. math::

    \delta x = 2 k k',

whatever the method, which is drawn as the prediction.  For the varying kernel,
geometric from :math:`k_{\rm min}` to :math:`k_{\rm max}` over :math:`N`
pixels, :math:`k' = k \ln(k_{\rm max}/k_{\rm min})/(N - 1)`.

This is an exploratory script, not one of the characterizations: it is not run
by ``make figures``, and writes to ``doc/figures/exploration/`` by default.
"""

from pathlib import Path

from astropy.table import Table, vstack
from matplotlib import pyplot
from matplotlib.lines import Line2D
import numpy as np
from scipy import fft, optimize

from dc3.core import lsf

import characterize_common
import explore_convolution_methods


VARYING_NPIX = 60000
"""The length of the spectrum carrying the varying kernel, in pixels."""


def predicted_shift(k):
    r"""
    Return the predicted first-moment shift of a line under the varying kernel.

    :math:`2 k k'`, with the gradient of the geometric kernel of
    :data:`explore_convolution_methods.KERNEL_RANGE` over :data:`VARYING_NPIX`
    pixels; see the module docstring.

    Parameters
    ----------
    k : :class:`numpy.ndarray`
        The local kernel dispersion, in pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The shift, in pixels, towards the wider kernel.
    """
    kmin, kmax = explore_convolution_methods.KERNEL_RANGE
    return 2 * np.square(k) * np.log(kmax / kmin) / (VARYING_NPIX - 1)


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
        The length of the zero-padded transform, at least twice the segment,
        so that the convolution does not wrap around.

    Returns
    -------
    :class:`numpy.ndarray`
        The broadened template, the same length as ``template``.
    """
    omega = 2 * np.pi * fft.rfftfreq(npad)
    kernel = np.exp(-0.5 * np.square(omega * sigma) - 1j * omega * shift)
    return fft.irfft(fft.rfft(template, npad) * kernel, npad)[:template.size]


def _residuals(par, template, observed, npad):
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

    Parameters
    ----------
    observed : :class:`numpy.ndarray`
        The observed spectrum.
    template : :class:`numpy.ndarray`
        The template spectrum, on the same grid.
    center : float
        The line centre, in pixels.
    sigma_guess : float
        The initial guess for the kernel dispersion, in pixels.
    halfwindow : int
        The half-width of the fitted segment, in pixels; it must enclose the
        observed line and exclude its neighbours.

    Returns
    -------
    tuple
        The fitted amplitude, kernel dispersion and shift, the last two in
        pixels.
    """
    c = int(np.round(center))
    lo, hi = max(c - halfwindow, 0), min(c + halfwindow + 1, observed.size)
    seg_t, seg_o = template[lo:hi], observed[lo:hi]
    npad = fft.next_fast_len(2 * seg_t.size)
    result = optimize.least_squares(
        _residuals, [1.0, max(sigma_guess, 0.05), 0.0], args=(seg_t, seg_o, npad),
        bounds=([0.0, 0.0, -halfwindow / 2], [np.inf, np.inf, halfwindow / 2]),
        x_scale='jac'
    )
    return tuple(result.x)


def first_moment(flux, centers, halfwindow):
    """
    Return the centroid of the samples about each line.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The spectrum.
    centers : :class:`numpy.ndarray`
        The line centres, in pixels.
    halfwindow : :class:`numpy.ndarray`
        The half-width of the window over which each centroid is taken, in
        pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The flux-weighted mean position of each line, in pixels.
    """
    x = np.arange(flux.size, dtype=float)
    out = np.empty(centers.size, dtype=float)
    for i, (c, h) in enumerate(zip(centers, halfwindow)):
        window = np.absolute(x - c) <= h
        out[i] = np.sum(flux[window] * x[window]) / np.sum(flux[window])
    return out


def measure(npix, kernel, sigma_in, case):
    """
    Convolve a comb by every method and fit the kernel relating each line.

    The comb, the window and the second moments follow
    :func:`explore_convolution_methods.measure`.

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
    spacing = max(61.37, 18.37 * np.sqrt(sigma_in ** 2 + np.amax(kernel) ** 2))
    centers = characterize_common.comb_centers(npix, spacing, 2.5 * spacing)
    flux = lsf.gaussian_comb(npix, centers, sigma_in)
    k = np.interp(centers, np.arange(npix), kernel)
    expected = np.sqrt(sigma_in ** 2 + k ** 2)
    halfwindow = np.minimum(8 * expected + 4, spacing / 2 - 1)
    moment_in = explore_convolution_methods.second_moment(flux, centers, halfwindow)
    centroid_in = first_moment(flux, centers, halfwindow)
    tables = []
    for method in explore_convolution_methods.METHODS:
        convolved = explore_convolution_methods.convolve(flux, kernel, method)
        if convolved is None:
            continue
        fits = np.array([
            fit_kernel(convolved, flux, c, kk, int(h))
            for c, kk, h in zip(centers, k, halfwindow)
        ])
        moment_out = explore_convolution_methods.second_moment(convolved, centers, halfwindow)
        tables.append(Table({
            'case': np.full(centers.size, case),
            'method': np.full(centers.size, method),
            'sigma_in': np.full(centers.size, sigma_in),
            'kernel': k,
            'excess': np.square(fits[:, 1]) - np.square(k),
            'shift': fits[:, 2],
            'centroid_shift': first_moment(convolved, centers, halfwindow) - centroid_in,
            'moment_excess': moment_out - moment_in - np.square(k),
        }))
    return vstack(tables)


def plot_shift(table):
    r"""
    Plot the shift of each line, in the layout of the variance figure.

    Three rows: the uniform kernels against the kernel width, and the varying
    kernel against the kernel width and against the local dynamic range,
    :math:`D = k_{\rm max}/k`.  Solid lines, with their 16th-84th percentile
    bands, are the shift fitted by :func:`fit_kernel`; dotted lines the change
    in the first moment of the samples; the black dashed line, in the rows for
    the varying kernel, the predicted first-moment shift, :func:`predicted_shift`.

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
    sigma_in = explore_convolution_methods.SIGMA_IN
    fig, axes = pyplot.subplots(
        len(rows), len(sigma_in), figsize=(10.0, 9.5), sharey=True, layout='constrained'
    )
    kernel_range = explore_convolution_methods.KERNEL_RANGE
    kmax = kernel_range[1]
    edges = {
        'k': np.geomspace(*kernel_range, 21),
        'D': np.geomspace(1.0, kmax / kernel_range[0], 21),
    }
    for row, (case, xaxis) in enumerate(rows):
        for col, s in enumerate(sigma_in):
            ax = axes[row, col]
            for i, method in enumerate(explore_convolution_methods.METHODS):
                indx = (
                    (table['case'] == case) & (table['method'] == method)
                    & (table['sigma_in'] == s)
                )
                if not np.any(indx):
                    continue
                kernel = np.asarray(table['kernel'][indx])
                x = kernel if xaxis == 'k' else kmax / kernel
                fit = explore_convolution_methods.summarize(
                    x, np.asarray(table['shift'][indx]), case == 'uniform', edges[xaxis]
                )
                centroid = explore_convolution_methods.summarize(
                    x, np.asarray(table['centroid_shift'][indx]), case == 'uniform',
                    edges[xaxis]
                )
                ax.fill_between(
                    fit['x'], fit['p16'], fit['p84'], color=f'C{i}', alpha=0.15, lw=0
                )
                ax.plot(fit['x'], fit['median'], color=f'C{i}', label=method)
                ax.plot(centroid['x'], centroid['median'], color=f'C{i}', ls=':', lw=1.5)
            if case == 'varying':
                kgrid = np.geomspace(*kernel_range, 300)
                ax.plot(
                    kgrid if xaxis == 'k' else kmax / kgrid, predicted_shift(kgrid),
                    color='k', ls='--', lw=1.0
                )
            ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
            ax.set_xscale('log')
            if xaxis == 'k':
                ax.set_xticks([0.1, 0.3, 1, 3, 10], labels=['0.1', '0.3', '1', '3', '10'])
                ax.set_xlabel('Kernel dispersion, $k$ (pixels)')
            else:
                ax.set_xticks([1, 3, 10, 30, 100], labels=['1', '3', '10', '30', '100'])
                ax.set_xlabel(r'Local dynamic range, $D = k_{\rm max}/k$')
            if row == 0:
                ax.set_title(rf'$\sigma_{{\rm in}}$ = {s:g} pixel{"s" if s > 1 else ""}')
            if col == 0:
                ax.set_ylabel(f'{case} kernel\nshift of the line centre (pixels)')
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color='k', ls='-', label='kernel fit'),
        Line2D([], [], color='k', ls=':', lw=1.5, label='first moment'),
        Line2D([], [], color='k', ls='--', lw=1.0, label=r"prediction, $2kk'$"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=7)
    fig.suptitle(
        'Shift of the line centre by three convolutions: kernel fit (median and 16th-84th '
        'percentiles over pixel phase)\nand first moment of the samples (median); positive '
        'towards larger pixel index',
        fontsize=9
    )
    return fig


def run():
    """
    Fit every method, for uniform kernels and for the varying kernel.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, method and configuration.
    """
    tables = []
    for sigma_in in explore_convolution_methods.SIGMA_IN:
        for k in explore_convolution_methods.UNIFORM_K:
            tables.append(measure(4000, np.full(4000, k), sigma_in, 'uniform'))
        tables.append(measure(
            VARYING_NPIX,
            np.geomspace(*explore_convolution_methods.KERNEL_RANGE, VARYING_NPIX), sigma_in,
            'varying'
        ))
    return vstack(tables)


def main():
    """Run the exploration and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.set_defaults(outdir=Path(__file__).resolve().parents[1] / 'figures' / 'exploration')
    args = parser.parse_args()
    characterize_common.apply_style()
    table = run()
    for case in ['uniform', 'varying']:
        for method in explore_convolution_methods.METHODS:
            for sigma_in in explore_convolution_methods.SIGMA_IN:
                indx = (
                    (table['case'] == case) & (table['method'] == method)
                    & (table['sigma_in'] == sigma_in) & (table['kernel'] > 1.0)
                )
                if not np.any(indx):
                    continue
                print(
                    f'{case:8s} {method:27s} sigma_in = {sigma_in:g}: k > 1 px, median kernel-fit '
                    f'excess {np.median(table["excess"][indx]):+.4f} px^2, second moment '
                    f'{np.median(table["moment_excess"][indx]):+.4f} px^2, largest |shift| '
                    f'{np.amax(np.absolute(table["shift"][indx])):.1e} px'
                )
    fig = explore_convolution_methods.plot(table, fit_label='kernel fit')
    characterize_common.save(fig, table, 'kernel_fitting', args)
    characterize_common.save(plot_shift(table), table, 'kernel_fitting_shift', args)


if __name__ == '__main__':
    main()
