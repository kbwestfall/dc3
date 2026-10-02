"""
Characterization 1: the pre- and post-pixelized widths of a line.

Establishes the instrument the other characterizations measure with.  A line of
known pre-pixelized width is integrated over pixels at a range of pixel phases
and then fit two ways:

- with a Gaussian evaluated at the pixel centres, which returns the
  post-pixelized width.  This reproduces Figure 4 of Law et al. (2021, AJ 161,
  52), whose polynomial fit is overplotted as an independent check;
- with a Gaussian integrated over each pixel, which should return the
  pre-pixelized width exactly.

If the second fit does not recover its input, nothing measured with it means
anything; if the first did not reproduce the published relation, the setup
would be suspect.
"""

from astropy.table import Table
from matplotlib import pyplot
import numpy as np

from dc3.core import lsf

import characterize_common


# Law et al. (2021), Figure 4: omega_POST / omega_PRE as a polynomial in
# omega_POST, fit over 0.8 <= omega_POST <= 2.0 pixels.
LAW2021_COEFFICIENTS = [1.5612, -1.2153, 1.0567, -0.4199, 0.0633]
LAW2021_RANGE = (0.8, 2.0)


def run():
    """
    Measure both widths over a grid of input widths and pixel phases.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per input width and phase.
    """
    sigma_pre = np.linspace(0.6, 2.2, 33)
    # Phase 1 repeats phase 0 a pixel along, so a plot against phase closes
    phases = np.arange(11) / 10
    rows = []
    x = np.arange(101, dtype=float)
    for s in sigma_pre:
        for phase in phases:
            center = 50.0 + phase
            y = lsf.pixelated_gaussian(x, center=center, sigma=s)
            _, _, recovered = lsf.fit_line(y, center, s, fix_center=True)
            _, _, post = lsf.fit_line(y, center, s, pixelated=False, fix_center=True)
            rows.append((s, phase, post, recovered))
    return Table(rows=rows, names=['sigma_pre', 'phase', 'sigma_post', 'sigma_recovered'])


def plot(table):
    """
    Plot the post-pixelized bias and the pre-pixelized recovery.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, (top, bottom) = pyplot.subplots(
        2, 1, figsize=(6.5, 6.0), height_ratios=[2, 1], layout='constrained'
    )

    # scatter and plot advance separate color cycles, so the default cycle's
    # colors are named to keep the three series distinct
    ratio = table['sigma_post'] / table['sigma_pre']
    top.scatter(
        table['sigma_post'], ratio, color='C0', lw=0,
        label='Measured: Gaussian at pixel centres (11 pixel phases)'
    )
    post = np.linspace(*LAW2021_RANGE, 200)
    top.plot(
        post, np.polynomial.polynomial.polyval(post, LAW2021_COEFFICIENTS), color='C1',
        label='Law et al. (2021), Fig. 4 fit'
    )
    post = np.linspace(np.amin(table['sigma_post']), np.amax(table['sigma_post']), 200)
    top.plot(
        post, post / np.sqrt(np.square(post) - 1 / 12), color='C2',
        label=r'Second moment: $\sigma_{\rm post}^2 = \sigma_{\rm pre}^2 + 1/12$'
    )
    top.set_xlabel(r'$\sigma_{\rm post}$ (pixels)')
    top.set_ylabel(r'$\sigma_{\rm post} / \sigma_{\rm pre}$')
    top.set_title('A Gaussian fit at the pixel centres overestimates the pre-pixelized width')
    top.legend()

    bottom.scatter(
        table['sigma_pre'], table['sigma_recovered'] / table['sigma_pre'] - 1, color='C0', lw=0
    )
    bottom.set_ylim(-5e-15, 5e-15)
    bottom.set_xlabel(r'$\sigma_{\rm pre}$ (pixels)')
    bottom.set_ylabel(r'$\sigma_{\rm fit} / \sigma_{\rm pre} - 1$')
    bottom.set_title('A pixel-integrated Gaussian recovers it to round-off')
    return fig


def plot_phase(table, sigma_pre=(0.6, 0.7, 0.8, 0.9)):
    """
    Plot the departure from the second-moment prediction against pixel phase.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The measurements from :func:`run`.
    sigma_pre : tuple, optional
        The input pre-pixelized widths, in pixels, whose measurements are
        plotted, one series each.

    Returns
    -------
    tuple
        The :class:`matplotlib.figure.Figure` and the
        :class:`astropy.table.Table` of the rows it plots.
    """
    select = np.any([np.isclose(table['sigma_pre'], s) for s in sigma_pre], axis=0)
    rows = table[select]
    fig, ax = pyplot.subplots(figsize=(6.5, 3.5), layout='constrained')
    for i, s in enumerate(sigma_pre):
        series = rows[np.isclose(rows['sigma_pre'], s)]
        predicted = np.sqrt(np.square(series['sigma_pre']) + 1 / 12)
        ratio = series['sigma_post'] / predicted
        ax.plot(series['phase'], ratio, color=f'C{i}')
        ax.scatter(
            series['phase'], ratio, color=f'C{i}', lw=0,
            label=rf'$\sigma_{{\rm pre}}$ = {s} pixels'
        )
    ax.set_xlabel('Pixel phase of the line centre')
    ax.set_ylabel(r'$\sigma_{\rm post} / \sqrt{\sigma_{\rm pre}^2 + 1/12}$')
    ax.legend()
    return fig, rows


def main():
    """Run the characterization and write its products."""
    args = characterize_common.get_parser(__doc__.split('\n')[1]).parse_args()
    characterize_common.apply_style()
    table = run()

    # The agreement with the published relation, over the range it was fit
    indx = (table['sigma_post'] >= LAW2021_RANGE[0]) & (table['sigma_post'] <= LAW2021_RANGE[1])
    law = np.polynomial.polynomial.polyval(table['sigma_post'][indx], LAW2021_COEFFICIENTS)
    measured = table['sigma_post'][indx] / table['sigma_pre'][indx]
    table.meta['law2021_max_fractional_difference'] = float(
        np.amax(np.absolute(measured / law - 1))
    )
    table.meta['recovery_max_fractional_error'] = float(
        np.amax(np.absolute(table['sigma_recovered'] / table['sigma_pre'] - 1))
    )
    print(
        'Largest departure from Law et al. (2021) Fig. 4 over its range: '
        f'{table.meta["law2021_max_fractional_difference"]:.2%}'
    )
    print(
        'Largest error in the recovered pre-pixelized width: '
        f'{table.meta["recovery_max_fractional_error"]:.1e}'
    )
    characterize_common.save(plot(table), table, 'pixelization', args)
    fig, rows = plot_phase(table)
    characterize_common.save(fig, rows, 'pixelization_phase', args)


if __name__ == '__main__':
    main()
