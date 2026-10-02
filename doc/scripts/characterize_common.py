"""
Shared machinery for the template-preparation characterization scripts.

Each ``characterize_*.py`` script in this directory runs one experiment and
writes a figure, a PNG by default, and a table to
``doc/figures/characterization/``.  Everything they produce is committed; the
documentation build never runs them.  Rerun them with ``make figures`` from
``doc/``.

Figure conventions, set here by :func:`apply_style` where matplotlib allows and
otherwise followed by each script:

- points are drawn with ``scatter``, without borders (``lw=0``), and lines with
  ``plot``;
- colours come from matplotlib's default cycle.  Since ``scatter`` and ``plot``
  advance separate cycles, a panel mixing the two names them explicitly --
  ``'C0'``, ``'C1'``, ... -- so each series keeps a distinct colour;
- minor ticks are on, major and minor ticks point inwards, and every axis is
  ticked, top and right included.

The colour constants below style the axes, text and background through
:func:`apply_style`; they are not for plotting data.
"""

import argparse
from pathlib import Path

from astropy.table import Table
from matplotlib import pyplot
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'figures' / 'characterization'
"""Where every characterization product is written."""

INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'


def get_parser(description):
    """
    Return the command-line parser every characterization script shares.

    Parameters
    ----------
    description : str
        What the script measures.

    Returns
    -------
    :class:`argparse.ArgumentParser`
        The parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--outdir', type=Path, default=OUTPUT_DIR,
        help='Directory for the figure and the table.'
    )
    parser.add_argument(
        '--format', default='png',
        help='Figure format.  Any format matplotlib can write; PNG is the default, and is '
             'what the documentation includes.'
    )
    return parser


def apply_style():
    """Set the figure style shared by every characterization figure."""
    pyplot.rcParams.update({
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS,
        'axes.labelcolor': INK_SECONDARY,
        'axes.titlecolor': INK,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'grid.linestyle': '-',
        'axes.axisbelow': True,
        'xtick.color': INK_MUTED,
        'ytick.color': INK_MUTED,
        # Inward major and minor ticks on all four axes
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.labelcolor': INK_SECONDARY,
        'ytick.labelcolor': INK_SECONDARY,
        'lines.linewidth': 1.5,
        'lines.solid_capstyle': 'round',
        'lines.solid_joinstyle': 'round',
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.labelcolor': INK_SECONDARY,
        'font.size': 9,
        # High enough resolution for the figures to stay sharp in the
        # documentation, which includes them as PNGs.
        'savefig.dpi': 200,
        # For the SVG option: keep text as text, and fix the salt the element
        # ids are derived from, so regenerating an unchanged figure leaves the
        # file unchanged.
        'svg.fonttype': 'none',
        'svg.hashsalt': 'dc3-characterization',
    })


def save(fig, table, name, args):
    """
    Write a characterization figure and its table.

    Parameters
    ----------
    fig : :class:`matplotlib.figure.Figure`
        The figure.
    table : :class:`astropy.table.Table`
        The measurements the figure plots, written as ECSV so the numbers can
        be inspected without rerunning the experiment.
    name : str
        Root name of both files.
    args : :class:`argparse.Namespace`
        The parsed command line.
    """
    args.outdir.mkdir(parents=True, exist_ok=True)
    figfile = args.outdir / f'{name}.{args.format}'
    # The figures are committed, so regenerating one should not change it unless
    # its content changed: omit the creation date an SVG or PDF would record,
    # and the matplotlib version a PNG would, which would otherwise rewrite
    # every figure on an upgrade.
    if args.format == 'png':
        metadata = {'Software': None}
    elif args.format in ['svg', 'pdf']:
        metadata = {'Date': None}
    else:
        metadata = None
    fig.savefig(figfile, bbox_inches='tight', metadata=metadata)
    pyplot.close(fig)
    tabfile = args.outdir / f'{name}.ecsv'
    table.write(tabfile, format='ascii.ecsv', overwrite=True)
    print(f'Wrote {figfile} and {tabfile}')


def binned_stats(x, y, edges):
    """
    Return the median and 16th/84th percentiles of ``y`` in bins of ``x``.

    Parameters
    ----------
    x, y : :class:`numpy.ndarray`
        The data.
    edges : :class:`numpy.ndarray`
        Bin edges in ``x``.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per non-empty bin: the bin centre, the number of points, and the
        median and percentiles of ``y``.
    """
    centers, counts, med, lo, hi = [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        indx = (x >= a) & (x < b)
        if not np.any(indx):
            continue
        centers.append((a + b) / 2)
        counts.append(np.sum(indx))
        p16, p50, p84 = np.percentile(y[indx], [16, 50, 84])
        med.append(p50)
        lo.append(p16)
        hi.append(p84)
    return Table([centers, counts, med, lo, hi], names=['x', 'n', 'median', 'p16', 'p84'])


def comb_centers(npix, spacing, margin):
    """
    Return line centres spaced incommensurately with the pixels.

    A spacing that is not a whole number of pixels puts successive lines at
    different phases within their pixel, so one comb samples the full range of
    pixel phase without a separate loop over it.

    Parameters
    ----------
    npix : int
        Number of pixels in the spectrum.
    spacing : float
        Line spacing, in pixels; choose a value far from any simple fraction.
    margin : float
        Distance to keep from each end, in pixels.

    Returns
    -------
    :class:`numpy.ndarray`
        The centres, in pixels.
    """
    return np.arange(margin, npix - margin, spacing)
