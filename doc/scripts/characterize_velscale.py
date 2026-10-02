r"""
Characterization 3, context: the pixel-size ratio Step 2 meets in production.

Characterization 3 measures what resampling does as a function of the ratio of
output to input pixel size, :math:`s = \Delta_{\rm out}/\Delta_{\rm in}`.  This
computes which :math:`s` production meets when a library sampled linearly in
wavelength is resampled at the smallest ratio that keeps it Nyquist sampled.

A template library sampled linearly in wavelength, from :math:`\lambda_0` to
:math:`\lambda_1`, is resampled in Step 2 onto a logarithmic grid of
:math:`v_g/R` km/s per pixel, where :math:`v_g` is the galaxy's velocity scale
and :math:`R` the integer ``velscale_ratio``.  The problem is described by
three unitless quantities:

- :math:`F`, the FWHM of the line-spread function being resampled, in input
  pixels, assumed constant along the spectrum;
- :math:`r = \lambda_1/\lambda_0`, the wavelength span of the input;
- :math:`g = v_g/v_c`, the galaxy velocity scale over the velocity scale of the
  input pixel at the centre of the grid, :math:`v_c = c\,\Delta\lambda_{\rm
  in}/\lambda_c`, with :math:`\lambda_c = (\lambda_0 + \lambda_1)/2`.

To first order in :math:`v/c`, the ratio of output to input pixel size is
:math:`s(\lambda) = (g/R)\,\lambda/\lambda_c`, so at the two ends of the
spectrum

.. math::

    s_0 = \frac{g}{R}\,\frac{2}{1 + r},
    \qquad
    s_1 = \frac{g}{R}\,\frac{2r}{1 + r} = r\,s_0.

The line-spread function spans :math:`F/s` output pixels, so Nyquist sampling
requires :math:`s \leq F/2`.  Since :math:`s_1 > s_0`, only the red end
constrains :math:`R`:

.. math::

    R = \max\left(1, \left\lceil \frac{4gr}{F(1 + r)} \right\rceil\right),

the ratio :func:`~dc3.core.resolution.minimum_velscale_ratio` gives for that
line-spread function.  This maps that :math:`R`, and the :math:`s_0` and
:math:`s_1` it gives, over :math:`1 \leq r \leq 3` and :math:`0.5 \leq g \leq 2`
for the library's *own* resolution, :math:`F` = 2.8 for MILES.

That is an upper bound on what ``velscale_ratio = 'auto'`` selects in
:func:`~dc3.templates.prepare`.  Step 1 matches the templates to the galaxy's
resolution before Step 2 resamples them, so ``'auto'`` tests the matched
resolution, which is broader wherever the galaxy's is.  For MILES prepared for
MaNGA, the map gives 2 and ``'auto'`` gives 1.  The contour
:math:`s = 1` marks the covariance bound: below it, neighbouring output pixels
share input pixels.  A star marks MILES resampled to the MaNGA sampling.

The figure is drawn from a fine grid; the table written beside it samples the
same maps every 0.05 in :math:`r` and :math:`g`, which keeps the committed file
small.
"""

from astropy.table import Table
from matplotlib import colors, pyplot
import numpy as np

from dc3.core import sampling
from dc3.core.velocity import SPEED_OF_LIGHT

import characterize_common


FWHM = 2.8
"""The instrumental FWHM, in input pixels: MILES-like, 2.51 A FWHM in 0.9 A pixels."""


def miles_to_manga():
    r"""
    Return :math:`r` and :math:`g` for resampling MILES spectra to the MaNGA sampling.

    MILES spectra have 4300 pixels of 0.9 angstroms from 3540.5 angstroms;
    MaNGA spectra are sampled at :math:`\Delta\log_{10}\lambda = 10^{-4}`, or
    about 69 km/s per pixel.

    Returns
    -------
    tuple
        :math:`r = \lambda_1/\lambda_0` and :math:`g = v_g/v_c`.
    """
    lam0 = 3540.5
    dlam = 0.9
    lam1 = lam0 + 4299 * dlam
    vc = SPEED_OF_LIGHT * dlam / ((lam0 + lam1) / 2)
    return lam1 / lam0, sampling.velscale(1e-4) / vc


def run(nr=200, ng=200):
    r"""
    Compute :math:`R`, :math:`s_0` and :math:`s_1` over a grid in :math:`r` and :math:`g`.

    Parameters
    ----------
    nr, ng : int, optional
        The number of grid points in :math:`r` and in :math:`g`.

    Returns
    -------
    :class:`astropy.table.Table`
        One row per grid point, ordered with :math:`r` varying fastest.  The
        grid shape is recorded in the table metadata.
    """
    r, g = np.meshgrid(np.linspace(1.0, 3.0, nr), np.linspace(0.5, 2.0, ng))
    ratio = np.maximum(1, np.ceil(4 * g * r / (FWHM * (1 + r)))).astype(int)
    s0 = 2 * g / (ratio * (1 + r))
    table = Table({'r': r.ravel(), 'g': g.ravel(), 'velscale_ratio': ratio.ravel(),
                   's0': s0.ravel(), 's1': (r * s0).ravel()})
    table.meta['shape'] = [ng, nr]
    table.meta['fwhm'] = FWHM
    return table


def plot(table):
    """
    Map :math:`R`, :math:`s_0` and :math:`s_1` over :math:`r` and :math:`g`.

    Parameters
    ----------
    table : :class:`astropy.table.Table`
        The result of :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    shape = table.meta['shape']
    r = np.reshape(table['r'], shape)
    g = np.reshape(table['g'], shape)
    # The image extent runs to the pixel edges, half a step beyond the samples
    dr = r[0, 1] - r[0, 0]
    dg = g[1, 0] - g[0, 0]
    extent = [r[0, 0] - dr / 2, r[0, -1] + dr / 2, g[0, 0] - dg / 2, g[-1, 0] + dg / 2]

    fig, axes = pyplot.subplots(3, 1, figsize=(6.5, 10.0), layout='constrained',
                                sharex=True)

    # R is a small integer, so it takes one colour from the default cycle each
    ratio = np.reshape(table['velscale_ratio'], shape)
    values = np.arange(1, np.amax(ratio) + 1)
    cmap = colors.ListedColormap([f'C{i}' for i in range(values.size)])
    norm = colors.BoundaryNorm(np.append(values, values[-1] + 1) - 0.5, cmap.N)
    image = axes[0].imshow(ratio, origin='lower', extent=extent, aspect='auto', cmap=cmap,
                           norm=norm, interpolation='nearest')
    fig.colorbar(image, ax=axes[0], ticks=values, label=r'$R$')
    axes[0].set_title(r'$R = \max(1, \lceil 4gr/[F(1 + r)] \rceil)$, '
                      rf'requiring $s_1 \leq F/2$, $F$ = {FWHM:g}')

    for ax, column, title in [
        (axes[1], 's0', r'$s_0 = (g/R)\,2/(1 + r)$, at the first wavelength'),
        (axes[2], 's1', r'$s_1 = r\,s_0$, at the last wavelength'),
    ]:
        s = np.reshape(table[column], shape)
        image = ax.imshow(s, origin='lower', extent=extent, aspect='auto', cmap='viridis',
                          interpolation='nearest')
        fig.colorbar(image, ax=ax, label=f'$s_{column[1]}$')
        # s jumps where R does, so contour each value of R separately; otherwise
        # the jumps across s = 1 would be drawn as contours too
        for value in values:
            region = np.ma.masked_where(ratio != value, s)
            if np.amin(region) < 1 < np.amax(region):
                contour = ax.contour(r, g, region, levels=[1.0], colors='w', linestyles='--',
                                     linewidths=1.0)
                ax.clabel(contour, fmt='$s$ = 1', fontsize=7)
        ax.set_title(title)

    miles_r, miles_g = miles_to_manga()
    for ax in axes:
        ax.scatter(miles_r, miles_g, marker='*', s=200, color='C3', lw=0,
                   label='MILES to MaNGA')
        ax.grid(False)
        ax.set_ylabel(r'$g = v_g/v_c$')
    axes[0].legend(loc='lower right', labelcolor='w', fontsize=8)
    axes[-1].set_xlabel(r'$r = \lambda_1/\lambda_0$')

    fig.suptitle('The velscale_ratio that keeps a linearly sampled template Nyquist sampled',
                 fontsize=10)
    return fig


def main():
    """Run the characterization and write its products."""
    args = characterize_common.get_parser(__doc__.split('\n')[1]).parse_args()
    characterize_common.apply_style()
    table = run()
    miles_r, miles_g = miles_to_manga()
    ratio = max(1, int(np.ceil(4 * miles_g * miles_r / (FWHM * (1 + miles_r)))))
    s0 = 2 * miles_g / (ratio * (1 + miles_r))
    print(f'MILES to MaNGA: r = {miles_r:.3f}, g = {miles_g:.3f}, R = {ratio}, '
          f's_0 = {s0:.3f}, s_1 = {miles_r * s0:.3f}')
    for ratio in np.unique(table['velscale_ratio']):
        rows = table[table['velscale_ratio'] == ratio]
        print(f'R = {ratio}: s_0 from {np.amin(rows["s0"]):.3f} to {np.amax(rows["s0"]):.3f}, '
              f's_1 from {np.amin(rows["s1"]):.3f} to {np.amax(rows["s1"]):.3f}')
    print(f'Over the whole map: s from {np.amin(table["s0"]):.3f} to '
          f'{np.amax(table["s1"]):.3f}')
    characterize_common.save(plot(table), run(nr=41, ng=31), 'velscale', args)


if __name__ == '__main__':
    main()
