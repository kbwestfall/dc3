r"""
Exploration: the pixel-size ratio when resampling a linear spectrum to a logarithmic grid.

Most template libraries are sampled linearly in wavelength, and Step 2 of
template preparation resamples them onto the galaxy's logarithmic sampling.
Characterization 3 shows that the excess variance resampling adds depends on
the ratio of output to input pixel size, :math:`s = \Delta_{\rm out}/\Delta_{\rm
in}`, and on the offset between the grids.  On a linear-to-logarithmic
resampling :math:`s` is not one number: the input pixel is fixed in
wavelength, :math:`\Delta\lambda_{\rm in}`, while the output pixel grows with
wavelength, :math:`\Delta\lambda_{\rm out} \approx \lambda v/c` for an output
velocity scale :math:`v`, so

.. math::

    s(\lambda) \approx \frac{v}{c}\,\frac{\lambda}{\Delta\lambda_{\rm in}},

rising linearly along the spectrum.  Over an input of :math:`N` pixels starting
at :math:`\lambda_0` it spans a factor

.. math::

    \frac{s_{\rm max}}{s_{\rm min}} = \frac{\lambda_0 + N\Delta\lambda_{\rm in}}{\lambda_0},

whatever :math:`v` is; :math:`v` only moves the range up or down.  This
computes :math:`s` exactly, output pixel by output pixel, from the borders of
:class:`~dc3.core.sampling.SpectralGrid` objects, and shows:

1. the range of :math:`s` against the output velocity scale, for inputs of
   several lengths with MILES-like sampling (0.9 angstrom pixels from 3540.5
   angstroms);
2. the span less one, :math:`s_{\rm max}/s_{\rm min} - 1 \approx N\Delta\lambda_{\rm
   in}/\lambda_0`, against the input length, for several input pixel sizes,
   starting at 3540.5 angstroms;
3. :math:`s(\lambda)` across a full MILES-like spectrum, for the galaxy
   velocity scales of a high-resolution survey (7.5 km/s, as in the DiskMass
   data) and of MaNGA (69 km/s), at several ``velscale_ratio``.

This is an exploratory script, not one of the characterizations: it is not run
by ``make figures``, and writes to ``doc/figures/exploration/`` by default.
"""

from pathlib import Path

from astropy.table import Table, vstack
from matplotlib import pyplot, ticker
import numpy as np

from dc3.core import sampling

import characterize_common


LAMBDA0 = 3540.5
"""The first input pixel centre, in angstroms, as in the MILES library."""

DLAMBDA = 0.9
"""The MILES-like input pixel size, in angstroms."""

MILES_NPIX = 4300
"""The length of a MILES spectrum, in pixels."""


def pixel_ratio(lam0, dlam, npix, velscale):
    r"""
    Return :math:`s` for each output pixel of a linear-to-logarithmic resampling.

    The output grid is logarithmic, with the given velocity scale, and spans the
    input: its first pixel is centred on the first input pixel.  Only output
    pixels lying wholly within the input are kept.

    Parameters
    ----------
    lam0 : float
        The first input pixel centre, in angstroms.
    dlam : float
        The input pixel size, in angstroms.
    npix : int
        The number of input pixels.
    velscale : float
        The output velocity scale, in km/s per pixel.

    Returns
    -------
    tuple
        The output pixel centres, in angstroms, and :math:`s` at each.
    """
    linear = sampling.SpectralGrid.from_linear_spacing(lam0, dlam, npix)
    dloglam = sampling.dloglam_from_velscale(velscale)
    nout = int(np.floor(np.log10(linear.borders[-1] / linear.borders[0]) / dloglam)) - 1
    log = sampling.SpectralGrid.from_log_spacing(np.log10(lam0), dloglam, nout)
    width = np.diff(log.borders)
    inside = log.borders[1:] <= linear.borders[-1]
    return log.wave[inside], width[inside] / dlam


def run():
    r"""
    Compute the range of the pixel-size ratio over the cases shown.

    Returns
    -------
    tuple
        Three :class:`astropy.table.Table` objects: the range of :math:`s`
        against output velocity scale and input length; the span against input
        length and pixel size; and :math:`s(\lambda)` for a MILES-like input at
        several galaxy samplings.
    """
    velscales = np.geomspace(5.0, 150.0, 40)
    rows = []
    for npix in [500, 1000, 2000, MILES_NPIX, 10000]:
        for v in velscales:
            _, s = pixel_ratio(LAMBDA0, DLAMBDA, npix, v)
            rows.append({'npix': npix, 'velscale': v, 's_min': np.amin(s), 's_max': np.amax(s)})
    by_velscale = Table(rows=rows)

    rows = []
    for dlam in [0.2, 0.9, 2.5]:
        for npix in np.unique(np.geomspace(100, 30000, 40).astype(int)):
            _, s = pixel_ratio(LAMBDA0, dlam, npix, 50.0)
            rows.append({'dlam': dlam, 'npix': npix, 'span': np.amax(s) / np.amin(s)})
    by_length = Table(rows=rows)

    tables = []
    for galaxy_velscale in [7.5, 69.0]:
        for ratio in [1, 2, 4]:
            wave, s = pixel_ratio(LAMBDA0, DLAMBDA, MILES_NPIX, galaxy_velscale / ratio)
            tables.append(Table({
                'galaxy_velscale': np.full(wave.size, galaxy_velscale),
                'velscale_ratio': np.full(wave.size, ratio),
                'wave': wave,
                's': s,
            }))
    miles = vstack(tables)
    return by_velscale, by_length, miles


def plot(by_velscale, by_length, miles):
    """
    Plot the three views of the pixel-size ratio.

    Parameters
    ----------
    by_velscale, by_length, miles : :class:`astropy.table.Table`
        The results of :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, (top, middle, bottom) = pyplot.subplots(3, 1, figsize=(6.5, 10.0), layout='constrained')

    # The range of s against output velocity scale, one band per input length
    for i, npix in enumerate(np.unique(by_velscale['npix'])):
        rows = by_velscale[by_velscale['npix'] == npix]
        top.fill_between(rows['velscale'], rows['s_min'], rows['s_max'], color=f'C{i}',
                         alpha=0.25, lw=0)
        top.plot(rows['velscale'], rows['s_min'], color=f'C{i}', lw=1.0)
        top.plot(rows['velscale'], rows['s_max'], color=f'C{i}', lw=1.0,
                 label=f'{npix} pixels, to {LAMBDA0 + (npix - 1) * DLAMBDA:.0f} A')
    top.axhline(1.0, color='0.5', ls='--', lw=1.0, zorder=0)
    top.set_xscale('log')
    top.set_yscale('log')
    top.set_xlabel('Output velocity scale, $v$ (km/s per pixel)')
    top.set_ylabel(r'$s = \Delta_{\rm out}/\Delta_{\rm in}$')
    top.set_title(rf'Range of $s$ over the spectrum: {DLAMBDA} A input pixels '
                  rf'from {LAMBDA0} A')
    top.legend(fontsize=7, title='Input length', title_fontsize=7)

    # The span s_max / s_min against input length, which v does not affect,
    # less one: the fractional change in s across the spectrum
    for i, dlam in enumerate(np.unique(by_length['dlam'])):
        rows = by_length[by_length['dlam'] == dlam]
        middle.plot(rows['npix'], rows['span'] - 1, color=f'C{i}', label=f'{dlam:g} A pixels')
    middle.set_xscale('log')
    middle.set_yscale('log')
    middle.set_xlabel('Input length, $N$ (pixels)')
    middle.set_ylabel(r'$s_{\rm max} / s_{\rm min} - 1$')
    middle.set_title(rf'Span of $s$ less one, $\approx N\Delta\lambda_{{\rm in}}/\lambda_0$, '
                     rf'from {LAMBDA0} A, for any $v$')
    middle.legend(fontsize=7)

    # s(lambda) across a MILES-like spectrum, for two galaxy samplings
    styles = {1: '-', 2: '--', 4: ':'}
    for i, galaxy_velscale in enumerate(np.unique(miles['galaxy_velscale'])):
        for ratio in [1, 2, 4]:
            rows = miles[(miles['galaxy_velscale'] == galaxy_velscale)
                         & (miles['velscale_ratio'] == ratio)]
            bottom.plot(rows['wave'], rows['s'], color=f'C{i}', ls=styles[ratio],
                        label=f'galaxy {galaxy_velscale:g} km/s, velscale_ratio = {ratio}')
    bottom.axhline(1.0, color='0.5', ls='--', lw=1.0, zorder=0)
    bottom.set_yscale('log')
    bottom.yaxis.set_major_formatter(ticker.ScalarFormatter())
    bottom.yaxis.set_minor_formatter(ticker.NullFormatter())
    bottom.set_xlabel('Wavelength (A)')
    bottom.set_ylabel(r'$s = \Delta_{\rm out}/\Delta_{\rm in}$')
    bottom.set_title(f'A MILES-like spectrum ({MILES_NPIX} pixels of {DLAMBDA} A): '
                     r'$s(\lambda)$')
    bottom.legend(fontsize=7, ncols=2, loc='upper center', bbox_to_anchor=(0.5, -0.15))

    fig.suptitle('Resampling a linearly sampled spectrum onto a logarithmic grid: the '
                 'pixel-size ratio', fontsize=10)
    return fig


def main():
    """Run the exploration and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.set_defaults(outdir=Path(__file__).resolve().parents[1] / 'figures' / 'exploration')
    args = parser.parse_args()
    characterize_common.apply_style()
    by_velscale, by_length, miles = run()
    for galaxy_velscale in np.unique(miles['galaxy_velscale']):
        for ratio in [1, 2, 4]:
            rows = miles[(miles['galaxy_velscale'] == galaxy_velscale)
                         & (miles['velscale_ratio'] == ratio)]
            print(
                f'MILES-like input, galaxy {galaxy_velscale:g} km/s, velscale_ratio = {ratio}: '
                f's from {np.amin(rows["s"]):.3f} to {np.amax(rows["s"]):.3f} '
                f'(span {np.amax(rows["s"]) / np.amin(rows["s"]):.3f})'
            )
    characterize_common.save(plot(by_velscale, by_length, miles), by_velscale,
                             'linear_to_log', args)
    by_length.write(args.outdir / 'linear_to_log_span.ecsv', format='ascii.ecsv',
                    overwrite=True)
    miles.write(args.outdir / 'linear_to_log_miles.ecsv', format='ascii.ecsv', overwrite=True)


if __name__ == '__main__':
    main()
