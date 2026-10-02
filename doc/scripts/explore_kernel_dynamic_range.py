r"""
Exploration: does the dynamic range of the kernel change the Step-1 excess?

``varsmooth`` stretches the coordinate so that each pixel spans
:math:`m\,k_{\rm max}/k` samples of its internal grid, :math:`m` being the
oversampling and :math:`k_{\rm max}` the widest kernel *anywhere in the
spectrum*.  If the excess broadening it adds comes from interpolating onto and
back from that grid, it should depend on the local kernel :math:`k` only
through the local sample spacing, :math:`1/(mD)` pixels with
:math:`D = k_{\rm max}/k` -- and so on how much wider the kernel is somewhere
else, not on the local kernel itself.

This separates the two.  The kernel is held uniform at :math:`k_p` across the
middle of the spectrum, where the lines are measured, and rises (or falls)
smoothly to :math:`k_{\rm end}` at both ends, where there are no lines.  The
ratio :math:`k_{\rm end}/k_p` is varied from 0.1 to 100; below 1 the widest
kernel is the local one, which tests whether a narrower kernel elsewhere
matters.  Lines are 2 pixels wide, well sampled, as are all but the smallest
:math:`k_p`.

Overlaid, for each oversampling, is the hypothesis

.. math::

    \sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k_p^2)
    = \frac{\Delta^2}{6}\left[1 + \frac{1}{(mD)^2}\right],
    \qquad D = \max(1, k_{\rm end}/k_p),

a constant :math:`\Delta^2/6` from interpolating the input samples onto the
stretched grid, and :math:`(1/mD)^2/6` from interpolating back from it.
``ppxf_util.gaussian_filter1d``, which does not interpolate, is the
reference.

This is an exploratory script, not one of the characterizations: it is not run
by ``make figures``, and writes to ``doc/figures/exploration/`` by default.
"""

from pathlib import Path

from astropy.table import Table, vstack
from matplotlib import pyplot
import numpy as np
from ppxf import ppxf_util

from dc3.core import lsf

import characterize_common
import explore_convolution_methods


SIGMA_IN = 2.0
"""The input pre-pixelized width, in pixels."""

K_PLATEAU = [0.3, 1.0, 3.0]
"""The uniform kernel where the lines are measured, in pixels."""

RATIOS = np.concatenate([[0.1, 0.3], np.geomspace(1.0, 100.0, 13)])
r"""The ratios :math:`k_{\rm end}/k_p` sampled."""

OVERSAMPLES = [1, 2, 4]
"""The ``varsmooth`` oversamplings."""

NPIX = 24000
"""The length of the spectrum, in pixels."""


def plateau_kernel(k_plateau, k_end):
    """
    Return a kernel uniform in the middle of the spectrum and different at its ends.

    The middle third is uniform at ``k_plateau`` and the outer sixths at
    ``k_end``; between them the kernel changes geometrically.  Lines are
    placed only well inside the middle third.

    Parameters
    ----------
    k_plateau : float
        The kernel in the middle, in pixels.
    k_end : float
        The kernel at the ends, in pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The kernel at each pixel, in pixels.
    """
    x = np.arange(NPIX, dtype=float)
    # 0 in the outer sixths, 1 in the middle third, linear in between
    weight = np.clip(np.minimum(x - NPIX / 6, NPIX * 5 / 6 - x) / (NPIX / 6), 0.0, 1.0)
    return np.exp(weight * np.log(k_plateau) + (1 - weight) * np.log(k_end))


def measure(k_plateau, ratio):
    """
    Convolve a comb in the uniform region and measure the excess of each line.

    Parameters
    ----------
    k_plateau : float
        The kernel where the lines are, in pixels.
    ratio : float
        The ratio of the kernel at the ends to ``k_plateau``.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line and method.
    """
    kernel = plateau_kernel(k_plateau, k_plateau * ratio)
    width = np.sqrt(SIGMA_IN ** 2 + k_plateau ** 2)
    spacing = max(61.37, 18.37 * width)
    centers = np.arange(NPIX / 3 + 500, NPIX * 2 / 3 - 500, spacing)
    flux = lsf.gaussian_comb(NPIX, centers, SIGMA_IN)
    # An oversampling of 1, below the production minimum, is run through
    # varsmooth directly; see explore_convolution_methods.varsmooth_vector
    outputs = {
        f'varsmooth, oversample = {m}': explore_convolution_methods.varsmooth_vector(
            flux, kernel, m
        )
        for m in OVERSAMPLES
    }
    outputs['gaussian_filter1d'] = ppxf_util.gaussian_filter1d(flux, kernel)
    tables = []
    for method, convolved in outputs.items():
        _, sigma_out = lsf.measure_comb(convolved, centers, width)
        tables.append(Table({
            'method': np.full(centers.size, method),
            'k_plateau': np.full(centers.size, k_plateau),
            'ratio': np.full(centers.size, ratio),
            'excess': np.square(sigma_out) - width ** 2,
        }))
    return vstack(tables)


def run():
    """
    Measure every plateau kernel and ratio.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, method and configuration.
    """
    tables = []
    for k_plateau in K_PLATEAU:
        for ratio in RATIOS:
            # Keep the widest kernel at 100 pixels or less, and the narrowest at
            # or above varsmooth's clip
            if k_plateau * ratio > 100.0 or k_plateau * ratio < 0.1:
                continue
            tables.append(measure(k_plateau, ratio))
    return vstack(tables)


def plot(table):
    r"""
    Plot the excess against the local dynamic range.

    The local dynamic range is :math:`D = \max(1, k_{\rm end}/k_p)`, the widest
    kernel in the spectrum relative to the one at the lines.  Every
    configuration whose ends are *narrower* than the plateau has :math:`D = 1`;
    those are plotted there as squares, which shows that their excess is not
    set by :math:`D` alone.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, axes = pyplot.subplots(
        1, len(K_PLATEAU), figsize=(10.0, 3.8), sharey=True, layout='constrained'
    )
    methods = [f'varsmooth, oversample = {m}' for m in OVERSAMPLES] + ['gaussian_filter1d']
    for ax, k_plateau in zip(axes, K_PLATEAU):
        sub = table[table['k_plateau'] == k_plateau]
        for i, method in enumerate(methods):
            rows = sub[sub['method'] == method]
            ratios = np.unique(rows['ratio'])
            excess = [rows['excess'][rows['ratio'] == r] for r in ratios]
            median = np.array([np.median(e) for e in excess])
            # The ends wider than, or as wide as, the plateau: D = k_end/k_p
            wider = ratios >= 1
            ax.fill_between(
                ratios[wider], [np.percentile(e, 16) for e, w in zip(excess, wider) if w],
                [np.percentile(e, 84) for e, w in zip(excess, wider) if w],
                color=f'C{i}', alpha=0.15, lw=0
            )
            ax.plot(ratios[wider], median[wider], color=f'C{i}')
            ax.scatter(ratios[wider], median[wider], color=f'C{i}', lw=0, s=16, label=method)
            # The ends narrower than the plateau, all of which have D = 1
            ax.scatter(
                np.ones(np.sum(~wider)), median[~wider], color=f'C{i}', lw=0, s=30, marker='s'
            )
            if method.startswith('varsmooth'):
                m = OVERSAMPLES[i]
                d = np.geomspace(1.0, 100.0, 300)
                ax.plot(d, (1 + 1 / np.square(m * d)) / 6, color=f'C{i}', ls='--', lw=1.0)
        ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
        ax.set_xscale('log')
        ax.set_xlabel(r'Local dynamic range, $D = \max(1, k_{\rm end}/k_p)$')
        ax.set_title(rf'$k_p$ = {k_plateau:g} pixels')
    axes[0].set_ylabel(r'$\sigma_{\rm out}^2 - (\sigma_{\rm in}^2 + k_p^2)$ (pixels$^2$)')
    axes[0].legend(fontsize=7)
    fig.suptitle(
        rf'Excess variance at a uniform local kernel, $\sigma_{{\rm in}}$ = {SIGMA_IN:g} pixels, '
        'against the local dynamic range\n(dashed: the hypothesis for each oversampling; '
        'squares: ends narrower than the plateau)',
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
    for k_plateau in K_PLATEAU:
        for ratio in np.unique(table['ratio'][table['k_plateau'] == k_plateau]):
            indx = (table['k_plateau'] == k_plateau) & (table['ratio'] == ratio)
            values = []
            for method in np.unique(table['method'][indx]):
                select = indx & (table['method'] == method)
                values.append(f'{method.replace("varsmooth, oversample = ", "m=")}: '
                              f'{np.median(table["excess"][select]):+.4f}')
            print(f'k_p = {k_plateau:g}, k_end/k_p = {ratio:7.3f}: ' + ', '.join(values))
    characterize_common.save(plot(table), table, 'kernel_dynamic_range', args)


if __name__ == '__main__':
    main()
