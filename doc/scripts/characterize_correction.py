r"""
Characterization 6: the whole preparation pipeline, with the excess variance corrected for.

Repeats Characterization 4 -- MILES-like templates prepared for MaNGA-like data,
at ``velscale_ratio = 2`` and ``varsmooth_oversample = 2``, over ten positions
of a line comb -- with ``correct_lsf_excess`` on, the default.  ``prepare`` then
chooses the matching kernel so that the two steps together bring the templates
to the target resolution in the second moment:

- Step 1's own excess, :math:`E_1` of Characterization 2, is accounted for by
  :func:`~dc3.core.resolution.match_resolution`, which chooses the kernel whose
  variance *with* that excess makes up the difference;
- Step 2's, of Characterization 3, is subtracted from the target passed to the
  matching, by ``prepare``.  It is predicted by either of the two
  ``resample_excess_method`` options, ``'expectation'`` and ``'local'``, both of
  which are run.

Each line is measured by a pixel-integrated Gaussian fit and compared with the
dispersion the prepared templates *report*, ``PreparedTemplates.idsp``: the
target, offset by ``dvar_inst``, wherever the matching succeeds; and what the
templates actually carry where it cannot, below about 4500 angstroms here, where
MILES with the excess of both steps is broader than the galaxy.  The excess is
reported in units of the native template pixel squared,
:math:`\Delta_{\rm tpl}^2`, and as a fractional excess in the dispersion.  Lines
where the matching fails are marked.
"""

import warnings

from astropy.table import Table, vstack
from matplotlib import pyplot
import numpy as np

from dc3 import templates
from dc3.core import lsf
from dc3.core.velocity import SPEED_OF_LIGHT

import characterize_common
import characterize_preparation


METHODS = ['local', 'expectation']
"""The ``resample_excess_method`` options, in plotting order."""


def run(method, shift=0.0):
    r"""
    Prepare the comb with the corrections, and measure each line against what is reported.

    Parameters
    ----------
    method : str
        Passed to :func:`~dc3.templates.prepare` as ``resample_excess_method``.
    shift : float, optional
        Shift of the comb, in native pixels; see
        :func:`characterize_preparation.make_library`.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, with the reported and measured dispersions and the
        excess, in units of :math:`\Delta_{\rm tpl}^2` and as a fraction.
    """
    library, line_wave = characterize_preparation.make_library(shift)
    galaxy = characterize_preparation.make_galaxy()
    with warnings.catch_warnings():
        # The out-of-range and unmatched-pixel warnings are expected; the
        # latter is what the marked lines show.
        warnings.simplefilter('ignore')
        prepared = templates.prepare(
            library, galaxy, velscale_ratio=2,
            varsmooth_oversample=characterize_preparation.VARSMOOTH_OVERSAMPLE,
            resample_excess_method=method
        )

    # The lines on the prepared grid, keeping those inside the galaxy's range
    # and with a full fitting window
    centers = (np.log10(line_wave) - prepared.log10lam0) / prepared.dloglam
    reported = np.interp(centers, np.arange(prepared.npix), prepared.idsp[0])
    guess = reported / prepared.velscale
    halfwindow = np.maximum(8 * guess, 5.5) + 2
    keep = (line_wave > galaxy.wave[0]) & (line_wave < galaxy.wave[-1]) \
        & (centers > halfwindow) & (centers < prepared.npix - 1 - halfwindow)
    _, sigma = lsf.measure_comb(prepared.flux[0], centers[keep], guess[keep])
    measured = sigma * prepared.velscale
    wave = line_wave[keep]
    reported = reported[keep]
    dv_tpl = SPEED_OF_LIGHT * characterize_preparation.MILES_DLAMBDA / wave
    unmatched = np.interp(wave, library.wave, prepared.match.unmatched.astype(float)) > 0

    table = Table({
        'method': np.full(wave.size, method),
        'shift': np.full(wave.size, shift),
        'wave': wave,
        'reported': reported,
        'measured': measured,
        'unmatched': unmatched,
        'dvar': (np.square(measured) - np.square(reported)) / np.square(dv_tpl),
        'fractional': measured / reported - 1,
    })
    table.meta['dvar_inst'] = float(prepared.dvar_inst)
    return table


def plot(table):
    r"""
    Plot the excess over the reported dispersion against wavelength, for each method.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The combined results of :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, axes = pyplot.subplots(
        2, len(METHODS), figsize=(9.0, 6.0), sharex=True, sharey='row', layout='constrained'
    )
    nshift = len(np.unique(table['shift']))
    for col, method in enumerate(METHODS):
        rows = table[table['method'] == method]
        unmatched = np.asarray(rows['unmatched'])
        for row, (column, scale, ylabel) in enumerate([
            ('dvar', 1.0, r'$(\sigma_{\rm measured}^2 - \sigma_{\rm reported}^2)\ /\ '
                          r'\Delta_{\rm tpl}^2$'),
            ('fractional', 100.0, r'$\sigma_{\rm measured}/\sigma_{\rm reported} - 1$ (%)'),
        ]):
            ax = axes[row, col]
            y = scale * np.asarray(rows[column])
            ax.scatter(rows['wave'][~unmatched], y[~unmatched], color='C0', s=4, lw=0,
                       label=f'Matched ({nshift} comb positions)')
            ax.scatter(rows['wave'][unmatched], y[unmatched], color='C3', s=4, lw=0,
                       label='Unmatched: MILES broader than the galaxy')
            ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
            if row == 0:
                ax.set_title(f"resample_excess_method = '{method}'", fontsize=9)
            else:
                ax.set_xlabel('Wavelength (A)')
            if col == 0:
                ax.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=7, loc='upper right')
    fig.suptitle(
        'MILES-like templates prepared for MaNGA-like data with the excess corrected for: '
        'velscale_ratio = 2,\nvarsmooth_oversample = 2; measured against the reported '
        'dispersion', fontsize=9
    )
    return fig


def main():
    """Run the characterization and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.add_argument(
        '--nshift', type=int, default=10,
        help='The number of comb positions, each shifted from the last by 1/nshift of the '
             'line spacing.'
    )
    args = parser.parse_args()
    characterize_common.apply_style()

    shifts = characterize_preparation.LINE_SPACING * np.arange(args.nshift) / args.nshift
    table = vstack(
        [run(method, shift) for method in METHODS for shift in shifts],
        metadata_conflicts='silent'
    )
    for method in METHODS:
        rows = table[table['method'] == method]
        for label, select in [
            ('matched', ~rows['unmatched']), ('unmatched', rows['unmatched'])
        ]:
            dvar = rows['dvar'][select]
            fractional = rows['fractional'][select]
            print(
                f'{method}, {label}: {len(dvar)} lines; excess over the reported dispersion '
                f'median {np.median(dvar):+.4f}, mean {np.mean(dvar):+.4f}, 16-84% '
                f'{np.percentile(dvar, 16):+.4f} to {np.percentile(dvar, 84):+.4f} '
                f'Delta_tpl^2; in sigma median {np.median(fractional):+.2%}, range '
                f'{np.amin(fractional):+.2%} to {np.amax(fractional):+.2%}'
            )
        print(f'    unmatched below {np.amax(rows["wave"][rows["unmatched"]]):.0f} A')
    characterize_common.save(plot(table), table, 'correction', args)


if __name__ == '__main__':
    main()
