r"""
Characterization 5: what a jump or a masked gap in the library does to the prepared lines.

A spliced library is smooth within each section, and differs where two meet:
its sampling or resolution may jump there, and the pixels between the sections
may be missing, padded and masked.  The resolution-matching convolution spreads
each pixel over the kernel's footprint, so it cannot treat any of these
correctly, nor carry the library's mask through.  :func:`dc3.templates.prepare`
therefore convolves the library as it is and masks the prepared templates
around each such region, grown by ``convolution_mask_growth`` dispersions of
the matching kernel either side, and flagged ``SAMP_JUMP``, ``RES_JUMP`` or
``TPL_MASKED``.  This measures how far from the region the prepared lines are
actually affected, to test whether the growth, at its default of three kernel
dispersions, is wide enough and not needlessly wide.

Three experiments, each isolating one kind of region at 4025 angstroms:

``sampling``
    A linear library whose pixel size changes from 0.02 to 0.03 angstroms, at
    a constant resolution.
``resolution``
    A linear library at 0.02 angstroms throughout, whose resolution steps from
    3 to 12 km/s.
``masked``
    A linear library at 0.02 angstroms and 3 km/s throughout, with a gap of
    0.5 angstroms whose pixels are set to zero and masked, as a gap between
    spliced sections padded with zeros would be.  The lines that fall in the
    gap itself are not measured.

Each line is a Gaussian in velocity integrated exactly over each pixel's own
borders, so it is the correct pixelized line on either side of a jump and
across it, and has a known pre-pixelized width: that of its section.  Many
combs, each offset by a fraction of the line spacing, sample the distance from
the region finely.

The galaxy's resolution is 20 km/s but for a dip to 15 km/s at its blue end.
The dip sets ``dvar_inst``, which leaves a matching kernel of about 13 km/s,
constant everywhere near the region; with constant resolutions throughout, the
kernel would be held at ``epsilon_sigma`` and the growth at its one-pixel
minimum, and with a kernel that changed across the region the preparation's
ordinary effect on the width would change with it.  Every line is measured
after preparation against the width ``prepare`` reports for it, and that
departure is compared with the median departure of the lines of its own
section that are clear of the region but near enough to share its kernel.
That removes the preparation's ordinary excess, the :math:`E_1 + 1/6` of
Characterizations 2 and 3, and leaves only the region's effect.  The width is
reported as a fractional departure, and the centre as a shift in km/s.

The combs lie on a zero background, so they show what a region does to the
lines' own profiles.  What a masked gap does to the *continuum* -- the deficit
of its zeros, spread over the kernel's footprint -- is measured separately, by
preparing a flat continuum of unit flux on the same grid, at the same
resolution and with the same mask, and taking the fractional error of the
prepared flux.  Since preparation is linear in the flux, that error adds to a
template's prepared flux whatever its lines.
"""

import warnings

from astropy.table import Table, vstack
from matplotlib import pyplot, ticker
import numpy as np
from scipy.special import ndtr

from dc3 import templates
from dc3.core import lsf, sampling
from dc3.core.velocity import SPEED_OF_LIGHT
from dc3.spectra import GalaxySpectra

import characterize_common


# The join, and the ranges of the library and the galaxy around it
JOIN = 4025.0
LIBRARY_RANGE = (3900.0, 4150.0)
GALAXY_LOG10LAM0 = np.log10(3950.0)
GALAXY_DLOGLAM = 1.09e-5
GALAXY_NPIX = 1480

# The masked gap of the third experiment, centred on the join, in angstroms
GAP = (JOIN - 0.25, JOIN + 0.25)

# The line spacing, in km/s, and the number of combs it is divided into: a
# step of 4 km/s in distance from the join, about 0.3 kernel dispersions
SPACING = 400.0
NCOMB = 100

# Lines between these distances from the join, in kernel dispersions, set each
# section's baseline departure
BASELINE = (12.0, 40.0)

CASES = {
    'sampling': ('SAMP_JUMP', 'Sampling jump: 0.02 to 0.03 A'),
    'resolution': ('RES_JUMP', 'Resolution jump: 3 to 12 km/s'),
    'masked': ('TPL_MASKED', 'Masked gap of zeros: 0.5 A'),
}
"""Each experiment's flag and figure title."""


def section_borders(edges, steps):
    """
    Return the borders of contiguous linear sections.

    Parameters
    ----------
    edges : list
        The wavelengths at which the sections begin and end.
    steps : list
        The pixel size of each section.

    Returns
    -------
    :class:`numpy.ndarray`
        The borders.
    """
    borders = [np.array([edges[0]])]
    for lo, hi, step in zip(edges[:-1], edges[1:], steps):
        npix = int(np.round((hi - lo) / step))
        borders.append(borders[-1][-1] + step * np.arange(1, npix + 1))
    return np.concatenate(borders)


def integrated_comb(borders, centers, sigma_kms):
    """
    Return lines Gaussian in velocity, integrated over each pixel.

    The flux density in each pixel is the fraction of each line's flux falling
    between the pixel's borders, divided by its width, so the comb is exact on
    any grid, including across a jump in the pixel size.

    Parameters
    ----------
    borders : :class:`numpy.ndarray`
        The pixel borders, in angstroms.
    centers : :class:`numpy.ndarray`
        The line centres, in angstroms.
    sigma_kms : :class:`numpy.ndarray`
        The pre-pixelized dispersion of each line, in km/s.

    Returns
    -------
    :class:`numpy.ndarray`
        The flux density in each pixel.
    """
    lnb = np.log(borders)
    flux = np.zeros(borders.size - 1, dtype=float)
    for center, sigma in zip(centers, sigma_kms):
        s = sigma / SPEED_OF_LIGHT
        lo = np.searchsorted(lnb, np.log(center) - 10 * s)
        hi = np.searchsorted(lnb, np.log(center) + 10 * s)
        lo, hi = max(lo - 1, 0), min(hi + 1, lnb.size)
        cdf = ndtr((lnb[lo:hi] - np.log(center)) / s)
        flux[lo:hi - 1] += np.diff(cdf)
    return flux / np.diff(borders)


def make_case(case, offset, continuum=False):
    """
    Build the library for one experiment and one comb offset.

    Parameters
    ----------
    case : str
        ``'sampling'``, ``'resolution'`` or ``'masked'``.
    offset : float
        The offset of the comb, as a fraction of the line spacing.
    continuum : bool, optional
        Replace the comb by a flat continuum of unit flux.

    Returns
    -------
    tuple
        The :class:`~dc3.templates.TemplateLibrary`, the line centres in
        angstroms, and their pre-pixelized dispersions in km/s.
    """
    if case == 'sampling':
        borders = section_borders([LIBRARY_RANGE[0], JOIN, LIBRARY_RANGE[1]], [0.02, 0.03])
    else:
        borders = section_borders(list(LIBRARY_RANGE), [0.02])
    wave = (borders[1:] + borders[:-1]) / 2
    if case == 'resolution':
        idsp = np.where(wave < JOIN, 3.0, 12.0)
    else:
        idsp = np.full(wave.size, 3.0)

    # Lines spaced uniformly in velocity, with the comb's own offset
    step = SPACING / SPEED_OF_LIGHT
    lnmin, lnmax = np.log(LIBRARY_RANGE[0]) + 3 * step, np.log(LIBRARY_RANGE[1]) - 3 * step
    centers = np.exp(np.arange(lnmin + offset * step, lnmax, step))
    sigma = np.interp(centers, wave, idsp)
    if case == 'resolution':
        sigma = np.where(centers < JOIN, 3.0, 12.0)

    flux = np.ones(wave.size) if continuum else integrated_comb(borders, centers, sigma)
    mask = np.zeros(wave.size, dtype=bool)
    if case == 'masked':
        mask = (wave > GAP[0]) & (wave < GAP[1])
        flux[mask] = 0.0

    grid = sampling.SpectralGrid('irregular', wave.size, wave=wave, borders=borders)
    library = templates.TemplateLibrary(
        flux, grid, key=f'{case}-{offset:.3f}', mask=mask, idsp=idsp
    )
    return library, centers, sigma


def galaxy():
    """
    Return the galaxy, at 20 km/s but for a dip to 15 km/s at its blue end.

    Returns
    -------
    :class:`~dc3.spectra.GalaxySpectra`
        The galaxy.
    """
    return GalaxySpectra(
        np.ones(GALAXY_NPIX),
        sampling.SpectralGrid.from_log_spacing(GALAXY_LOG10LAM0, GALAXY_DLOGLAM, GALAXY_NPIX),
        idsp=np.where(np.arange(GALAXY_NPIX) < 20, 15.0, 20.0)
    )


def prepare_case(library, growth):
    """
    Prepare one library, suppressing the warnings the setup causes.

    Parameters
    ----------
    library : :class:`~dc3.templates.TemplateLibrary`
        The library.
    growth : float
        Passed to :func:`~dc3.templates.prepare` as ``convolution_mask_growth``.

    Returns
    -------
    :class:`~dc3.templates.PreparedTemplates`
        The prepared library.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return templates.prepare(library, galaxy(), convolution_mask_growth=growth)


def join_kernel(prepared, library):
    """
    Return the matching kernel at the join, in km/s: the larger of its two sides.

    Parameters
    ----------
    prepared : :class:`~dc3.templates.PreparedTemplates`
        The prepared library.
    library : :class:`~dc3.templates.TemplateLibrary`
        The library it was prepared from.

    Returns
    -------
    float
        The kernel dispersion, which sets the distance scale.
    """
    kernel = prepared.match.kernel_sigma
    join = np.searchsorted(library.wave, JOIN)
    return max(kernel[join - 1], kernel[join])


def measure(case, offset, growth=3.0):
    """
    Prepare one library and measure every line in the galaxy's range.

    Parameters
    ----------
    case : str
        ``'sampling'``, ``'resolution'`` or ``'masked'``.
    offset : float
        The comb offset, as a fraction of the line spacing.
    growth : float, optional
        Passed to :func:`~dc3.templates.prepare` as ``convolution_mask_growth``.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per line.
    """
    library, centers, _ = make_case(case, offset)
    prepared = prepare_case(library, growth)
    sigma_kernel = join_kernel(prepared, library)

    true = (np.log10(centers) - prepared.log10lam0) / prepared.dloglam
    reported = np.interp(true, np.arange(prepared.npix), prepared.idsp[0])
    guess = reported / prepared.velscale
    halfwindow = np.maximum(8 * guess, 5.5) + 2
    keep = (true > halfwindow) & (true < prepared.npix - 1 - halfwindow)
    if case == 'masked':
        # A line inside the gap has been replaced by zeros, and is not a line
        keep &= (centers < GAP[0]) | (centers > GAP[1])
    fitted, sigma = lsf.measure_comb(prepared.flux[0], true[keep], guess[keep])

    masked = getattr(prepared.mask, CASES[case][0])[0]
    pixel = np.clip(np.round(true[keep]).astype(int), 0, prepared.npix - 1)
    distance = SPEED_OF_LIGHT * np.log(centers[keep] / JOIN)
    n = int(np.sum(keep))
    return Table({
        'case': np.full(n, case),
        'distance': distance,
        'distance_sigma': distance / sigma_kernel,
        'sigma_kms': sigma * prepared.velscale,
        'reported_kms': reported[keep],
        'shift_kms': (fitted - true[keep]) * prepared.velscale,
        'masked': masked[pixel],
        'kernel_kms': np.full(n, sigma_kernel),
    })


def band_edges(case, growth=3.0):
    """
    Return the region ``prepare`` masks, in kernel dispersions from the join.

    Parameters
    ----------
    case : str
        ``'sampling'``, ``'resolution'`` or ``'masked'``.
    growth : float, optional
        Passed to :func:`~dc3.templates.prepare` as ``convolution_mask_growth``.

    Returns
    -------
    tuple
        The lower and upper edges of the masked region.
    """
    library, _, _ = make_case(case, 0.0)
    prepared = prepare_case(library, growth)
    masked = getattr(prepared.mask, CASES[case][0])[0]
    sigma_kernel = join_kernel(prepared, library)
    borders = sampling.SpectralGrid.from_log_spacing(
        prepared.log10lam0, prepared.dloglam, prepared.npix
    ).borders
    index = np.flatnonzero(masked)
    lo, hi = borders[index[0]], borders[index[-1] + 1]
    return (
        SPEED_OF_LIGHT * np.log(lo / JOIN) / sigma_kernel,
        SPEED_OF_LIGHT * np.log(hi / JOIN) / sigma_kernel,
    )


def continuum_error(case, growth=3.0):
    """
    Prepare a flat continuum through one experiment and return its flux error.

    Parameters
    ----------
    case : str
        ``'sampling'``, ``'resolution'`` or ``'masked'``.
    growth : float, optional
        Passed to :func:`~dc3.templates.prepare` as ``convolution_mask_growth``.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per prepared pixel within the baseline distance of the join,
        with the fractional error of its flux and whether it is masked.
    """
    library, _, _ = make_case(case, 0.0, continuum=True)
    prepared = prepare_case(library, growth)
    distance = SPEED_OF_LIGHT * np.log(prepared.wave / JOIN) / join_kernel(prepared, library)
    near = np.absolute(distance) < BASELINE[1]
    n = int(np.sum(near))
    return Table({
        'case': np.full(n, case),
        'distance_sigma': distance[near],
        'error': prepared.flux[0, near] - 1,
        'masked': getattr(prepared.mask, CASES[case][0])[0, near],
    })


def run():
    """
    Measure every experiment over every comb offset.

    Returns
    -------
    tuple
        Two :class:`astropy.table.Table` objects: one row per line, with its
        departure from its section's baseline; and one row per prepared pixel
        of the flat continuum, with its flux error.
    """
    tables = []
    for case in CASES:
        lines = vstack([measure(case, offset) for offset in np.arange(NCOMB) / NCOMB])
        excess = lines['sigma_kms'] / lines['reported_kms'] - 1
        # Each section's baseline: the median over its lines clear of the join
        # but near enough to share its kernel
        lines['departure'] = 0.0
        lines['shift'] = 0.0
        for side in [lines['distance'] < 0, lines['distance'] > 0]:
            d = np.absolute(lines['distance_sigma'])
            near = side & (d > BASELINE[0]) & (d < BASELINE[1])
            lines['departure'][side] = excess[side] - np.median(excess[near])
            lines['shift'][side] = lines['shift_kms'][side] - np.median(lines['shift_kms'][near])
        # Beyond the baseline region the ordinary excess drifts with the
        # changing kernel, which a local baseline does not remove
        tables.append(lines[np.absolute(lines['distance_sigma']) < BASELINE[1]])
    return vstack(tables), vstack([continuum_error(case) for case in CASES])


def affected_extent(lines, threshold):
    """
    Return how far from the join a departure exceeds a threshold.

    Parameters
    ----------
    lines : :class:`astropy.table.Table`
        The lines of one experiment.
    threshold : float
        The fractional width departure regarded as significant.

    Returns
    -------
    tuple
        The most negative and most positive distance, in kernel dispersions,
        at which the width departs by more than ``threshold``.
    """
    affected = np.absolute(lines['departure']) > threshold
    if not np.any(affected):
        return 0.0, 0.0
    d = lines['distance_sigma'][affected]
    return float(np.amin(d)), float(np.amax(d))


def plot(table, continuum, bands):
    """
    Plot the departures and the continuum error against the distance from the join.

    Parameters
    ----------
    table, continuum : :class:`astropy.table.Table`
        The measurements from :func:`run`.
    bands : dict
        The masked region of each experiment, from :func:`band_edges`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    fig, axes = pyplot.subplots(3, 3, figsize=(9.0, 7.5), sharex=True, layout='constrained')
    for col, case in enumerate(CASES):
        lines = table[table['case'] == case]
        pixels = continuum[continuum['case'] == case]
        for row, (data, column, ylabel) in enumerate([
            (lines, 'departure', 'Line width departure'),
            (lines, 'shift', 'Line centre shift (km/s)'),
            (pixels, 'error', '|Continuum flux error|'),
        ]):
            ax = axes[row, col]
            lo, hi = bands[case]
            ax.axvspan(lo, hi, color='0.5', alpha=0.15, lw=0, zorder=0,
                       label='Masked by prepare' if row == 0 else None)
            y = np.absolute(data[column]) if column == 'error' else data[column]
            ax.scatter(data['distance_sigma'], y, color='C0', s=4, lw=0, zorder=2)
            if column == 'error':
                # A logarithmic axis masks the pixels with no error at all
                ax.set_yscale('log', nonpositive='mask')
                ax.set_ylim(1e-7, 2.0)
            else:
                ax.axhline(0.0, color='0.5', lw=1.0, zorder=1)
            if row == 0:
                ax.set_title(CASES[case][1], fontsize=9)
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel('Distance from the join (kernel dispersions)')
            ax.set_xlim(-10, 10)
    axes[0, 0].legend(loc='upper left', fontsize=7)
    fig.suptitle(
        'Lines prepared across a jump or a masked gap, against lines of the same section clear '
        'of it\n(shaded: the region prepare masks at convolution_mask_growth = 3)', fontsize=9
    )
    return fig


def main():
    """Run the characterization and write its products."""
    args = characterize_common.get_parser(__doc__.split('\n')[1]).parse_args()
    characterize_common.apply_style()
    table, continuum = run()
    bands = {case: band_edges(case) for case in CASES}
    for case in CASES:
        pixels = continuum[continuum['case'] == case]
        outside = ~pixels['masked']
        print(f'{case}: largest continuum flux error outside the masked region '
              f'{np.amax(np.absolute(pixels["error"][outside])):.2e}, inside '
              f'{np.amax(np.absolute(pixels["error"][~outside])):.2e}')
        lines = table[table['case'] == case]
        outside = ~lines['masked']
        print(
            f'{case}: masked {bands[case][0]:+.2f} to {bands[case][1]:+.2f} kernel sigma; '
            f'largest departure outside it {np.amax(np.absolute(lines["departure"][outside])):.3%}'
            f', largest shift {np.amax(np.absolute(lines["shift"][outside])):.3f} km/s; '
            f'inside {np.amax(np.absolute(lines["departure"][~outside])):.3%}, '
            f'{np.amax(np.absolute(lines["shift"][~outside])):.3f} km/s'
        )
        # The scatter between lines far from the jump is about 0.1 per cent, so
        # no finer threshold is meaningful
        for threshold in [0.01, 0.003]:
            lo, hi = affected_extent(lines, threshold)
            print(f'    width departs by more than {threshold:.1%} from {lo:+.2f} to {hi:+.2f}')
    characterize_common.save(plot(table, continuum, bands), table, 'splices', args)
    continuum.write(args.outdir / 'splices_continuum.ecsv', format='ascii.ecsv',
                    overwrite=True)


if __name__ == '__main__':
    main()
