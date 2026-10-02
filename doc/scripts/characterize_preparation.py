r"""
Characterization 4: the whole preparation pipeline, for MILES templates fit to MaNGA data.

A full-pipeline check of the predictions of Characterizations 2 and 3, on a
concrete case: the resolution of templates prepared exactly as production
prepares them, against what the two predictions say it should be.  A comb of
lines with the MILES library's sampling and resolution is prepared for
MaNGA-like galaxy spectra with
:func:`dc3.templates.prepare`, at ``velscale_ratio = 2`` and
``varsmooth_oversample = 2``.  The pre-pixelized width of each prepared line is
then measured by fitting a pixel-integrated Gaussian, and compared with the
dispersion Step 1 applied: the template's own, added in quadrature to the
matching kernel's.  That is the dispersion the prepared templates *report*,
``PreparedTemplates.idsp``, wherever the matching succeeds.  Where it cannot --
below about 4000 angstroms, where MILES is *broader* than the galaxy and, at
the default ``sigma_floor = 0``, the kernel is held at its floor -- ``idsp``
reports the galaxy's resolution rather than the template's, a bookkeeping gap
separate from the effects measured here.  Those lines are marked.

The experiment is repeated ten times, with the comb shifted by a tenth of the
line spacing each time, so the lines sample ten times as many positions
relative to both grids.

**The templates** are MILES-like: 4300 pixels of 0.9 angstroms from 3540.5
angstroms, linearly sampled, with a FWHM of 2.51 angstroms at every wavelength.

**The galaxy** is MaNGA-like, with two simplifications: a logarithmic sampling
of :math:`\Delta\log_{10}\lambda = 10^{-4}` from 3622 to 10354 angstroms, and an
instrumental dispersion that falls linearly from 80 km/s at 4000 angstroms to
55 km/s at 9000 angstroms.

**The sampling.**  ``velscale_ratio = 2`` is the ratio chosen from the MILES
library's own resolution; ``'auto'`` chooses 1, since it accounts for Step 1
broadening the templates to the galaxy's resolution first.  At 2, the
pixel-size ratio :math:`s` runs from about 0.46 to 0.94, so the output pixels
are smaller than the native ones throughout, and neighbouring output pixels are
correlated.  That is not all cost: Characterization 3 shows that for
:math:`s < 1` the Step 2 excess stays close to its expectation of 1/6 whatever
the grid offset, except near simple ratios, while near an integer :math:`s` it
ranges from 0 to 1/4.  The prepared widths are therefore more uniform, and a
correction for them more accurate.

**The prediction** is the sum of the two steps' excesses, in units of the native
template pixel squared, :math:`\Delta_{\rm tpl}^2`:

- Step 1, ``varsmooth``'s vector path (Characterization 2),

  .. math::

      E_1 = \frac{1}{6} - \frac{1}{\pi^2}\sum_{p \geq 1}
            \frac{e^{-2\pi^2p^2k^2}}{p^2} + \frac{1}{6(mD)^2},

  with :math:`k` the local kernel dispersion in native pixels, clipped at
  ``varsmooth``'s floor of 0.1, :math:`m` the oversampling and :math:`D =
  k_{\rm max}/k`;
- Step 2, resampling (Characterization 3): an expectation of 1/6 over the grid
  offset, at every pixel-size ratio :math:`s`.

``--velscale-ratio`` and ``--nshift`` change the configuration for exploration;
a ratio other than 2 is appended to the names of the products, so it does not
overwrite the committed figure.
"""

import warnings

from astropy.table import Table, vstack
from matplotlib import pyplot
import numpy as np

from dc3 import templates
from dc3.core import lsf, resolution, sampling
from dc3.core.velocity import SPEED_OF_LIGHT
from dc3.spectra import GalaxySpectra

import characterize_common


MILES_LAMBDA0 = 3540.5
"""The first MILES pixel centre, in angstroms."""

MILES_DLAMBDA = 0.9
"""The MILES pixel size, in angstroms."""

MILES_NPIX = 4300
"""The length of a MILES spectrum, in pixels."""

MILES_FWHM = 2.51
"""The MILES instrumental FWHM, in angstroms (Falcón-Barroso et al. 2011)."""

LINE_SPACING = 23.37
"""The comb's line spacing, in native pixels; incommensurate with the pixels."""

VARSMOOTH_OVERSAMPLE = 2
"""The nominal ``varsmooth_oversample``, the default of :class:`~dc3.templates.TemplatePar`."""


def make_library(shift=0.0):
    r"""
    Build the MILES-like template comb.

    Each line is a Gaussian of constant dispersion in angstroms integrated over
    the linear pixels, so its pre-pixelized width is known exactly.  The lines
    are spaced :data:`LINE_SPACING` pixels apart, a spacing incommensurate with
    the pixels, so the comb samples every pixel phase.

    Parameters
    ----------
    shift : float, optional
        Shift of the whole comb, in native pixels.

    Returns
    -------
    tuple
        The :class:`~dc3.templates.TemplateLibrary` and the line centres in
        angstroms.
    """
    sigma_pix = MILES_FWHM / resolution.SIGMA_TO_FWHM / MILES_DLAMBDA
    centers = characterize_common.comb_centers(MILES_NPIX, LINE_SPACING, 30) + shift
    grid = sampling.SpectralGrid.from_linear_spacing(MILES_LAMBDA0, MILES_DLAMBDA, MILES_NPIX)
    idsp = SPEED_OF_LIGHT * sigma_pix * MILES_DLAMBDA / grid.wave
    library = templates.TemplateLibrary(
        lsf.gaussian_comb(MILES_NPIX, centers, sigma_pix), grid, key='miles-comb', idsp=idsp
    )
    return library, MILES_LAMBDA0 + MILES_DLAMBDA * centers


def make_galaxy():
    """
    Build the MaNGA-like galaxy.

    Returns
    -------
    :class:`~dc3.spectra.GalaxySpectra`
        A flat spectrum carrying the MaNGA-like sampling and resolution.
    """
    log10lam0 = np.log10(3622.0)
    dloglam = 1e-4
    npix = int(np.floor((np.log10(10354.0) - log10lam0) / dloglam)) + 1
    grid = sampling.SpectralGrid.from_log_spacing(log10lam0, dloglam, npix)
    idsp = 80.0 + (55.0 - 80.0) * (grid.wave - 4000.0) / 5000.0
    return GalaxySpectra(np.ones(npix), grid, idsp=idsp)


def run(velscale_ratio=2, shift=0.0):
    r"""
    Prepare the comb, measure each line, and compute the prediction.

    Parameters
    ----------
    velscale_ratio : int or str, optional
        Passed to :func:`~dc3.templates.prepare`.
    shift : float, optional
        Shift of the comb, in native pixels; see :func:`make_library`.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line, with the reported and measured dispersions, the
        excess in units of :math:`\Delta_{\rm tpl}^2`, and the predicted
        excesses of each step.  The metadata record the preparation.
    """
    library, line_wave = make_library(shift)
    galaxy = make_galaxy()
    with warnings.catch_warnings():
        # The out-of-range and unmatched-pixel warnings are expected; the
        # latter is what the marked lines show.
        warnings.simplefilter('ignore')
        # Without the corrections, which is what this characterization tests;
        # characterize_correction.py repeats it with them
        prepared = templates.prepare(
            library, galaxy, velscale_ratio=velscale_ratio,
            varsmooth_oversample=VARSMOOTH_OVERSAMPLE, correct_lsf_excess=False
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

    # The native template pixel in velocity, the unit of the excess
    dv_tpl = SPEED_OF_LIGHT * MILES_DLAMBDA / wave

    # Step 1: the kernel at each line, in native pixels, as varsmooth clips it,
    # and the dispersion it applies, the template's own in quadrature with it
    kernel = np.maximum(prepared.match.kernel_sigma_pixels, resolution.VARSMOOTH_MIN_SIG)
    k = np.interp(wave, library.wave, kernel)
    applied = np.sqrt(
        np.square(np.interp(wave, library.wave, library.idsp[0])) + np.square(k * dv_tpl)
    )
    unmatched = np.interp(wave, library.wave, prepared.match.unmatched.astype(float)) > 0
    step1 = resolution.varsmooth_excess(k, VARSMOOTH_OVERSAMPLE, np.amax(kernel) / k)

    # Step 2: the local pixel-size ratio, and the expectation of 1/6
    dloglam = prepared.dloglam
    ratio = wave * (10 ** (dloglam / 2) - 10 ** (-dloglam / 2)) / MILES_DLAMBDA

    table = Table({
        'shift': np.full(wave.size, shift),
        'wave': wave,
        'reported': reported,
        'applied': applied,
        'unmatched': unmatched,
        'measured': measured,
        'dvar': (np.square(measured) - np.square(applied)) / np.square(dv_tpl),
        'fractional': measured / applied - 1,
        'fractional_reported': measured / reported - 1,
        'kernel': k,
        'ratio': ratio,
        'predicted_step1': step1,
        'predicted_step2': np.full(wave.size, 1 / 6),
        'dv_tpl': dv_tpl,
    })
    table.meta['velscale_ratio'] = prepared.velscale_ratio
    table.meta['velscale_ratio_requested'] = str(velscale_ratio)
    table.meta['velscale'] = float(prepared.velscale)
    table.meta['dvar_inst'] = float(prepared.dvar_inst)
    table.meta['varsmooth_oversample'] = VARSMOOTH_OVERSAMPLE
    return table


def plot(table):
    r"""
    Plot the measured and predicted excess against wavelength.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The combined results of :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, (top, bottom) = pyplot.subplots(
        2, 1, figsize=(6.5, 7.0), sharex=True, height_ratios=[3, 2], layout='constrained'
    )
    order = np.argsort(table['wave'])
    wave = np.asarray(table['wave'])
    step1 = np.asarray(table['predicted_step1'])
    total = step1 + np.asarray(table['predicted_step2'])
    unmatched = np.asarray(table['unmatched'])
    nshift = len(np.unique(table['shift']))

    # The variance excess over what Step 1 applied, in native template pixels
    # squared
    top.scatter(wave[~unmatched], table['dvar'][~unmatched], color='C0', s=4, lw=0,
                label=f'Measured, each line ({nshift} comb positions)')
    top.scatter(wave[unmatched], table['dvar'][unmatched], color='C3', s=4, lw=0,
                label='Measured, where MILES is broader than the galaxy')
    top.plot(wave[order], total[order], color='C1',
             label='Predicted: Step 1 + Step 2 expectation (1/6)')
    top.plot(wave[order], step1[order], color='C1', ls='--', lw=1.0,
             label='Predicted: Step 1 alone')
    top.axhline(0.0, color='0.5', lw=1.0, zorder=0)
    top.set_ylabel(r'$(\sigma_{\rm measured}^2 - \sigma_{\rm applied}^2)\ /\ '
                   r'\Delta_{\rm tpl}^2$')
    top.set_title(r'Excess pre-pixelized variance over what Step 1 applied, '
                  r'$\sigma_{\rm applied}^2 = \sigma_{\rm tpl}^2 + k^2$')
    top.legend(fontsize=7)

    # The same, as a fractional excess in the dispersion; and, where matching
    # fails, the excess over what the prepared templates report
    applied = np.asarray(table['applied'])
    predicted = np.sqrt(1 + total * np.square(np.asarray(table['dv_tpl']) / applied)) - 1
    bottom.scatter(wave[~unmatched], 100 * table['fractional'][~unmatched], color='C0', s=4,
                   lw=0, label='Over the applied dispersion (= reported)')
    bottom.scatter(wave[unmatched], 100 * table['fractional'][unmatched], color='C3', s=4,
                   lw=0, label='Over the applied dispersion')
    bottom.scatter(wave[unmatched], 100 * table['fractional_reported'][unmatched], color='C5',
                   s=4, lw=0, label='Over the reported dispersion')
    bottom.plot(wave[order], 100 * predicted[order], color='C1')
    bottom.axhline(0.0, color='0.5', lw=1.0, zorder=0)
    bottom.set_xlabel('Wavelength (A)')
    bottom.set_ylabel(r'$\sigma_{\rm measured}/\sigma - 1$ (%)')
    bottom.set_title('As a fractional excess in the dispersion')
    bottom.legend(fontsize=7)

    fig.suptitle(
        f"MILES-like templates prepared for MaNGA-like data: velscale_ratio = "
        f"{table.meta['velscale_ratio']}"
        + (" ('auto')" if table.meta['velscale_ratio_requested'] == 'auto' else '')
        + f", varsmooth_oversample = {table.meta['varsmooth_oversample']}", fontsize=10
    )
    return fig


def main():
    """Run the characterization and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.add_argument(
        '--velscale-ratio', default='2',
        help="The velscale_ratio passed to prepare: an integer, 2 by default, or 'auto'.  Any "
             "other than 2 is appended to the names of the products."
    )
    parser.add_argument(
        '--nshift', type=int, default=10,
        help='The number of comb positions, each shifted from the last by 1/nshift of the '
             'line spacing.'
    )
    args = parser.parse_args()
    characterize_common.apply_style()
    velscale_ratio = 'auto' if args.velscale_ratio == 'auto' else int(args.velscale_ratio)

    tables = []
    for shift in LINE_SPACING * np.arange(args.nshift) / args.nshift:
        result = run(velscale_ratio, shift)
        total = result['predicted_step1'] + result['predicted_step2']
        print(f'shift {shift:5.2f} px: {len(result)} lines, measured minus predicted median '
              f"{np.median(result['dvar'] - total):+.4f} Delta_tpl^2")
        tables.append(result)
    table = vstack(tables, metadata_conflicts='silent')

    total = table['predicted_step1'] + table['predicted_step2']
    print(f"velscale_ratio = {table.meta['velscale_ratio']}, prepared velscale = "
          f"{table.meta['velscale']:.2f} km/s, dvar_inst = {table.meta['dvar_inst']:.1f} "
          f"(km/s)^2")
    print(f"{len(table)} lines; s from {np.amin(table['ratio']):.3f} to "
          f"{np.amax(table['ratio']):.3f}; Step 1 kernel from {np.amin(table['kernel']):.3f} "
          f"to {np.amax(table['kernel']):.3f} native pixels")
    print(f"Excess: measured median {np.median(table['dvar']):.4f}, mean "
          f"{np.mean(table['dvar']):.4f}; predicted median {np.median(total):.4f} "
          f"Delta_tpl^2")
    print(f"Measured minus predicted: median {np.median(table['dvar'] - total):+.4f}, "
          f"mean {np.mean(table['dvar'] - total):+.4f}, 16-84% "
          f"{np.percentile(table['dvar'] - total, 16):+.4f} to "
          f"{np.percentile(table['dvar'] - total, 84):+.4f} Delta_tpl^2")
    print(f"Fractional excess in sigma over the applied dispersion: median "
          f"{np.median(table['fractional']):+.2%}, range {np.amin(table['fractional']):+.2%} "
          f"to {np.amax(table['fractional']):+.2%}")
    unmatched = np.asarray(table['unmatched'])
    if np.any(unmatched):
        print(f"{np.sum(unmatched)} lines below {np.amax(table['wave'][unmatched]):.0f} A are "
              f"unmatched; there the excess over the reported dispersion reaches "
              f"{np.amax(table['fractional_reported'][unmatched]):+.2%}")
    name = 'preparation' if velscale_ratio == 2 else f'preparation_r{velscale_ratio}'
    characterize_common.save(plot(table), table, name, args)


if __name__ == '__main__':
    main()
