r"""
Characterization 3: what Step 2, resampling, does to the pre-pixelized width.

Step 2 resamples each template onto the galaxy's logarithmic sampling, at an
integer ``velscale_ratio``, with :class:`~dc3.core.resample.Resample`.  The
``mangadap`` assumption is that this leaves the pre-pixelized dispersion
unchanged: the line-spread function is a property of the optics, not of the
sampling.

The spectrum being resampled is not the line-spread function, however, but the
line-spread function already integrated over its own pixels, and
``Resample`` treats each input pixel as flat before integrating it over the
output pixels.  This measures the result.  The change in the pre-pixelized
variance is expressed in units of the *input* pixel squared, :math:`\Delta_{\rm
in}^2`, against three variables:

- the ratio of output to input pixel size, :math:`s = \Delta_{\rm out}/\Delta_{\rm
  in}`;
- the offset of the output grid relative to the input grid, as a fraction of an
  output pixel -- which production does not control, since it follows from the
  templates' and the galaxy's starting wavelengths;
- the phase of each line within its pixel, sampled by an incommensurate comb.

The top panel overlays the prediction of :func:`predicted_dvar`, exact for the
second moment of a well-sampled line; the bottom panel overlays its range over
grid offset, :func:`predicted_range`, at rational :math:`s = k/12`, the only
values for which it holds.

The measurements take a few minutes.  Both are written as ECSV files beside the
figure, and ``--replot`` reads them back, if present, and only redraws the
figure; the prediction is recomputed either way.
"""

from fractions import Fraction

from astropy.table import Table, vstack
from matplotlib import pyplot, ticker
import numpy as np

from dc3.core import lsf, resample

import characterize_common


NPIX = 6000
LOG10LAM0 = np.log10(4000.0)
DLOGLAM = 2e-5


def predicted_dvar(ratio, offset):
    r"""
    Return the predicted change in pre-pixelized variance, in units of :math:`\Delta_{\rm in}^2`.

    ``Resample`` treats each input pixel as flat.  An output pixel whose
    borders fall a fraction :math:`\phi` of the way into an input pixel
    therefore collects :math:`1 - \phi` of the first input pixel it overlaps,
    all of those between, and :math:`\phi` of the last.  Its window is a box of
    the output pixel's width convolved with a two-point kernel, of weights
    :math:`1 - \phi` and :math:`\phi` one input pixel apart -- linear
    interpolation at :math:`\phi` -- whose variance is :math:`\phi(1 - \phi)`.

    For :math:`s = p/q` in lowest terms, :math:`\phi` takes :math:`q` equally
    spaced values across successive output pixels, and a line spanning several
    of them sees their average,

    .. math::

        \frac{\Delta\sigma^2_{\rm pre}}{\Delta_{\rm in}^2}
            = \frac{1}{6} - \frac{1 - 6\phi'(1 - \phi')}{6q^2},
        \qquad \phi' = {\rm frac}(p\,x),

    where :math:`x` is the grid offset as a fraction of an output pixel.  At
    integer :math:`s` this is :math:`\phi(1 - \phi)`, with amplitude 1/4 for
    any :math:`s`; as :math:`q` grows it tends to 1/6.  It is exact for the
    second moment; a Gaussian fit departs from it where the line is narrow on
    either grid.

    It holds only for a rational :math:`s`, and only for a line spanning many
    cycles of :math:`\phi`.  Near a simple ratio, :math:`\phi` drifts slowly
    from one output pixel to the next, so a line of finite width sees only part
    of a cycle, and an excess nearer :math:`\phi(1 - \phi)` at its local phase.

    Parameters
    ----------
    ratio : float
        Output pixel size in units of the input pixel, :math:`s`.  It must be
        a fraction with a small denominator; see :func:`lowest_terms`.
    offset : float or :class:`numpy.ndarray`
        Offset of the output grid, as a fraction of an output pixel.

    Returns
    -------
    float or :class:`numpy.ndarray`
        The predicted change in pre-pixelized variance.
    """
    p, q = lowest_terms(ratio)
    phase = np.mod(p * np.asarray(offset), 1.0)
    return 1 / 6 - (1 - 6 * phase * (1 - phase)) / (6 * q ** 2)


def predicted_range(ratio):
    r"""
    Return the predicted minimum, expectation and maximum over grid offset.

    With the grid offset :math:`x` uniform on [0, 1), :math:`\phi' = {\rm
    frac}(p\,x)` is uniform too, so :math:`\phi'(1 - \phi')` ranges from 0 to
    1/4 with mean 1/6.  In :func:`predicted_dvar` that gives

    .. math::

        \min = \frac{1}{6} - \frac{1}{6q^2},
        \qquad
        \langle \Delta\sigma^2_{\rm pre} \rangle = \frac{1}{6},
        \qquad
        \max = \frac{1}{6} + \frac{1}{12q^2},

    in units of :math:`\Delta_{\rm in}^2`.  The expectation is 1/6 at every
    :math:`s`; only the range depends on it.

    Parameters
    ----------
    ratio : float
        Output pixel size in units of the input pixel, :math:`s`.  It must be
        a fraction with a small denominator; see :func:`lowest_terms`.

    Returns
    -------
    tuple
        The minimum, the expectation and the maximum.
    """
    _, q = lowest_terms(ratio)
    return 1 / 6 - 1 / (6 * q ** 2), 1 / 6, 1 / 6 + 1 / (12 * q ** 2)


def lowest_terms(ratio, max_denominator=12):
    """
    Return a pixel-size ratio as a fraction in lowest terms.

    Parameters
    ----------
    ratio : float
        The ratio.
    max_denominator : int, optional
        The largest denominator tried.

    Returns
    -------
    tuple
        The numerator and the denominator.

    Raises
    ------
    ValueError
        Raised if the ratio is not, to within :math:`10^{-9}`, a fraction with
        a denominator of at most ``max_denominator``; the prediction does not
        hold for it.
    """
    fraction = Fraction(ratio).limit_denominator(max_denominator)
    if abs(float(fraction) - ratio) > 1e-9:
        raise ValueError(f'{ratio} is not a fraction with a denominator of at most '
                         f'{max_denominator}.')
    return fraction.numerator, fraction.denominator


def resample_comb(flux, centers, sigma_in, ratio, offset):
    r"""
    Resample a comb onto a new grid and measure its lines there.

    Parameters
    ----------
    flux : :class:`numpy.ndarray`
        The comb, on the input grid.
    centers : :class:`numpy.ndarray`
        Line centres, in input pixels.
    sigma_in : float
        Pre-pixelized dispersion of every line, in input pixels.
    ratio : float
        Output pixel size in units of the input pixel.
    offset : float
        Offset of the output grid, as a fraction of an output pixel.  At zero
        the first output pixel's lower border coincides with an input pixel
        border.

    Returns
    -------
    :class:`numpy.ndarray`
        The pre-pixelized width of each line on the output grid, in *input*
        pixels.
    """
    step = ratio * DLOGLAM
    # Five output pixels in from the lower edge, so no output pixel runs off
    # the input.
    first = LOG10LAM0 - DLOGLAM / 2 + (5 + 0.5 + offset) * step
    nout = int(NPIX / ratio) - 12
    resampled = resample.Resample(
        flux, x=10 ** (LOG10LAM0 + DLOGLAM * np.arange(NPIX)), inLog=True,
        newRange=[10 ** first, 10 ** (first + (nout - 1) * step)], newdx=step, newLog=True
    )
    out_centers = (LOG10LAM0 + DLOGLAM * centers - first) / step
    halfwindow = max(8 * sigma_in / ratio, 5.5) + 2
    keep = (out_centers > halfwindow) & (out_centers < nout - 1 - halfwindow)
    _, pre = lsf.measure_comb(resampled.outy, out_centers[keep], sigma_in / ratio)
    return pre * ratio


def summarize(sigma_in, ratio, offset, measured):
    """
    Reduce the per-line measurements of one configuration to a table row.

    Parameters
    ----------
    sigma_in : float
        Input pre-pixelized dispersion, in input pixels.
    ratio : float
        Output pixel size in units of the input pixel.
    offset : float
        Grid offset, as a fraction of an output pixel.
    measured : :class:`numpy.ndarray`
        The output of :func:`resample_comb`.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, with the change in pre-pixelized variance in units of
        the input pixel squared.
    """
    n = measured.size
    return Table({
        'sigma_in': np.full(n, sigma_in),
        'ratio': np.full(n, ratio),
        'offset': np.full(n, offset),
        'dvar_pre': np.square(measured) - sigma_in ** 2,
    })


def run():
    """
    Measure the resampled widths over the three variables.

    Both views sample the grid offset at the same 21 values, from 0 to 1 in
    steps of 0.05.

    Returns
    -------
    tuple
        Two :class:`astropy.table.Table` objects: the change against grid
        offset at a few ratios, one row per ratio and offset, reduced over line
        phase; and the change against ratio, one row per input width and ratio,
        reduced over grid offset and line phase by :func:`reduce_by_ratio`.
    """
    centers = characterize_common.comb_centers(NPIX, 61.37, 150)
    offsets = np.arange(21) / 20

    # Against grid offset, for a few ratios either side of one
    flux = lsf.gaussian_comb(NPIX, centers, 1.0)
    by_offset = []
    for ratio in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for offset in offsets:
            measured = summarize(1.0, ratio, offset, resample_comb(flux, centers, 1.0, ratio,
                                                                   offset))
            by_offset.append(reduce_rows(measured, ['sigma_in', 'ratio', 'offset'],
                                         ['dvar_pre']))

    # Against ratio, every line at every grid offset
    ratios = np.unique(np.concatenate([
        np.geomspace(0.25, 4.0, 25), [0.25, 1 / 3, 0.5, 1.0, 2.0, 3.0, 4.0]
    ]))
    by_ratio = []
    for sigma_in in [1.0, 2.0]:
        flux = lsf.gaussian_comb(NPIX, centers, sigma_in)
        for ratio in ratios:
            for offset in offsets:
                by_ratio.append(summarize(sigma_in, ratio, offset, resample_comb(
                    flux, centers, sigma_in, ratio, offset
                )))
    return vstack(by_offset), reduce_by_ratio(vstack(by_ratio))


def reduce_rows(table, keys, columns):
    """
    Reduce per-line rows to their extremes, mean, median and 16th/84th percentiles.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        Rows sharing the same values of ``keys``.
    keys : list
        The columns identifying the configuration.
    columns : list
        The measured columns to reduce.

    Returns
    -------
    :class:`astropy.table.Table`
        A single row.

    Raises
    ------
    ValueError
        Raised if there are no rows, or a measured value is not finite, which
        can only be a failed measurement.
    """
    if len(table) == 0:
        raise ValueError('No measurements to reduce.')
    row = {key: [table[key][0]] for key in keys}
    row['nlines'] = [len(table)]
    for column in columns:
        values = np.asarray(table[column])
        if not np.all(np.isfinite(values)):
            raise ValueError(f'Non-finite values in {column}: a measurement failed.')
        for label, pct in [('min', 0), ('p16', 16), ('median', 50), ('p84', 84),
                           ('max', 100)]:
            row[f'{column}_{label}'] = [np.percentile(values, pct)]
        row[f'{column}_mean'] = [np.mean(values)]
    return Table(row)


def reduce_by_ratio(by_ratio):
    """
    Reduce the per-line measurements against ratio.

    Parameters
    ----------
    by_ratio : :class:`astropy.table.Table`
        One row per line, at every input width, ratio and grid offset.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per input width and ratio, reduced over grid offset and line
        phase by :func:`reduce_rows`.
    """
    rows = []
    for sigma_in in np.unique(by_ratio['sigma_in']):
        for ratio in np.unique(by_ratio['ratio']):
            indx = (by_ratio['sigma_in'] == sigma_in) & (by_ratio['ratio'] == ratio)
            rows.append(reduce_rows(by_ratio[indx], ['sigma_in', 'ratio'], ['dvar_pre']))
    return vstack(rows)


def plot(by_offset, by_ratio):
    """
    Plot the change in pre-pixelized variance against grid offset and pixel ratio.

    Parameters
    ----------
    by_offset, by_ratio : :class:`astropy.table.Table`
        The measurements from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, (top, middle) = pyplot.subplots(2, 1, figsize=(6.5, 6.4), layout='constrained')

    # Measured medians as points, the second-moment prediction as lines
    offset = np.linspace(0.0, 1.0, 401)
    for i, ratio in enumerate([0.5, 1.0, 1.5, 2.0, 3.0]):
        indx = by_offset['ratio'] == ratio
        top.plot(offset, predicted_dvar(ratio, offset), color=f'C{i}', lw=1.0)
        top.scatter(by_offset['offset'][indx], by_offset['dvar_pre_median'][indx],
                    color=f'C{i}', lw=0, label=f'$s$ = {ratio:g}')
    top.set_xlabel('Offset of the output grid (fraction of an output pixel)')
    top.set_ylabel(r'$\Delta\sigma^2_{\rm pre}\ /\ \Delta_{\rm in}^2$')
    top.set_title(r'$\sigma_{\rm in}$ = 1 input pixel: median over line phase (points) '
                  'and prediction (lines)')
    top.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncols=5)

    for ax in [top, middle]:
        ax.axhline(0.0, color='0.5', lw=1.0, zorder=0)
        for level, label in [(1 / 12, '1/12'), (1 / 6, '1/6')]:
            ax.axhline(level, color='0.5', ls='--', lw=1.0, zorder=0)
            ax.text(1.005, level, label, transform=ax.get_yaxis_transform(), va='center',
                    fontsize=8)

    # Measured: the mean (solid) and extremes (dashed) over grid offset and line
    # phase, for each input width
    for i, sigma_in in enumerate([1.0, 2.0]):
        rows = by_ratio[by_ratio['sigma_in'] == sigma_in]
        middle.plot(rows['ratio'], rows['dvar_pre_mean'], color=f'C{i}',
                    label=rf'$\sigma_{{\rm in}}$ = {sigma_in:g} input pixel'
                          + ('s' if sigma_in > 1 else ''))
        for column in ['dvar_pre_min', 'dvar_pre_max']:
            middle.plot(rows['ratio'], rows[column], color=f'C{i}', ls='--', lw=1.0)

    # Predicted range over grid offset, independent of the input width, and only
    # for rational s, here at steps of 1/12.  Its expectation is 1/6 at every s,
    # which the reference line already shows.
    ratio = np.arange(3, 49) / 12
    predicted = np.array([predicted_range(r) for r in ratio])
    middle.vlines(ratio, predicted[:, 0], predicted[:, 2], color='k', lw=0.8,
                  label='Predicted range')
    middle.set_xscale('log')
    middle.set_xticks([0.25, 0.5, 1, 2, 4], labels=['1/4', '1/2', '1', '2', '4'])
    # Keep the minor ticks but not their labels, which would crowd the major ones
    middle.xaxis.set_minor_formatter(ticker.NullFormatter())
    middle.set_xlabel(r'$s = \Delta_{\rm out}/\Delta_{\rm in}$')
    middle.set_ylabel(r'$\Delta\sigma^2_{\rm pre}\ /\ \Delta_{\rm in}^2$')
    middle.set_title('Mean (solid) and extremes (dashed) over grid offset and phase')
    middle.legend(loc='lower left')

    fig.suptitle('Step 2: change in the pre-pixelized width caused by resampling', fontsize=10)
    return fig


def main():
    """Run the characterization and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.add_argument(
        '--replot', action='store_true',
        help='Read the measurements from the ECSV files a previous run wrote to the output '
             'directory, if both are present, and only redraw the figure.'
    )
    args = parser.parse_args()
    characterize_common.apply_style()

    ratio_file = args.outdir / 'resampling.ecsv'
    offset_file = args.outdir / 'resampling_by_offset.ecsv'
    if args.replot and ratio_file.exists() and offset_file.exists():
        print(f'Reading the measurements from {ratio_file} and {offset_file}')
        by_offset = Table.read(offset_file, format='ascii.ecsv')
        by_ratio = Table.read(ratio_file, format='ascii.ecsv')
    else:
        by_offset, by_ratio = run()

    for row in by_ratio:
        if row['ratio'] in [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]:
            low, mean, high = predicted_range(row['ratio'])
            print(
                f'sigma_in = {row["sigma_in"]:.0f} px, s = {row["ratio"]:.3g}: change in '
                f'pre-pixelized variance {row["dvar_pre_min"]:+.4f} to '
                f'{row["dvar_pre_max"]:+.4f}, mean {row["dvar_pre_mean"]:+.4f}, median '
                f'{row["dvar_pre_median"]:+.4f} Delta_in^2; predicted {low:+.4f} to '
                f'{high:+.4f}, mean {mean:+.4f}'
            )
    characterize_common.save(plot(by_offset, by_ratio), by_ratio, 'resampling', args)
    by_offset.write(offset_file, format='ascii.ecsv', overwrite=True)


if __name__ == '__main__':
    main()
