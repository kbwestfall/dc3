r"""
Exploration: choosing the logarithmic sampling for a linearly sampled template library.

Step 2 of template preparation resamples each template onto a logarithmic grid
of :math:`v_t = v_g/R` km/s per pixel, where :math:`v_g` is the galaxy's
velocity scale and :math:`R` the integer ``velscale_ratio``.  For an input
sampled linearly, with pixels of :math:`\Delta\lambda_{\rm in}` angstroms, the
ratio of output to input pixel size is, to first order in :math:`v_t/c`,

.. math::

    s(\lambda) = \frac{\Delta_{\rm out}}{\Delta_{\rm in}}
               = u\,\frac{\lambda}{\lambda_c},
    \qquad u \equiv \frac{v_t}{v_c},
    \qquad v_c \equiv \frac{c\,\Delta\lambda_{\rm in}}{\lambda_c},

where :math:`v_c` is the velocity scale of the input pixel at the centre of the
grid, :math:`\lambda_c = (\lambda_0 + \lambda_1)/2`.  Scaled this way the
problem depends on the input only through the ratio of its end wavelengths,
:math:`r = \lambda_1/\lambda_0`, and the width of its line-spread function.

Two bounds frame the choice of :math:`u`:

- **Lower: covariance.** Where :math:`s < 1` each input pixel is spread over
  more than one output pixel, and neighbouring output pixels are correlated.
  We want :math:`s \gtrsim 1`.
- **Upper: Nyquist sampling.** The line-spread function must span at least two
  output pixels at its FWHM, the criterion of
  :func:`~dc3.core.resolution.minimum_velscale_ratio`.  Beyond it the spectral
  resolution is degraded.

Assume, as for MILES, that the instrumental FWHM is constant in angstroms,
:math:`F` input pixels.  Its width in output pixels is then :math:`F/s`, so the
two bounds are both bounds on :math:`s`, and do not depend on wavelength:

.. math::

    1 \leq s(\lambda) \leq F/2.

The spectrum maps onto a band of :math:`s` of fixed logarithmic width,
:math:`\ln r`, which :math:`u` only slides.  It fits within the window only if
:math:`r \leq F/2`.  When it does not, the best choice is to put the red end on
the Nyquist limit,

.. math::

    u_* = \frac{F}{2}\,\frac{\lambda_c}{\lambda_1},
    \qquad v_{t,*} = \frac{c\,F\,\Delta\lambda_{\rm in}}{2\lambda_1},

the largest :math:`u` that undersamples nowhere.  It maximizes the fraction of
pixels inside the window, and the fraction with :math:`s < 1` is then
:math:`\max[0, 1 - \ln(F/2)/\ln r]`.  The smallest ``velscale_ratio`` that
keeps the template Nyquist sampled, :math:`R = \lceil v_g/v_{t,*}\rceil`, is the
one ``velscale_ratio = 'auto'`` selects.

In practice both :math:`v_g` and :math:`v_c` are fixed by the data, so
:math:`u = v_g/(R\,v_c)` takes only discrete values, and the choice of :math:`u`
is a choice of :math:`R`.  :math:`u_*` corresponds to the non-integer ratio
:math:`R_* = v_g/v_{t,*}`.

The width used here is the library's own.  Step 1 broadens the templates to
the galaxy's resolution before they are resampled, which can only widen the
line-spread function, so the upper bound shown is conservative.

Fractions are of output pixels, which are uniform in :math:`\log\lambda` and
so are what a fit sees.  The script writes three figures:

1. ``log_sampling_window``: for a MILES-like spectrum and one galaxy velocity
   scale (``--velscale``, 69 km/s by default), :math:`s(\lambda)` against the
   window for several :math:`R`, and the fraction of pixels in each regime
   against :math:`R`;
2. ``log_sampling_span``: the window in :math:`u` against :math:`r` for
   several :math:`F`, and the fraction with :math:`s < 1` at :math:`u_*`;
3. ``log_sampling_ratio``: for a MILES-like spectrum, the regimes against the
   galaxy velocity scale for each ``velscale_ratio``, and the ratio ``'auto'``
   selects.

This is an exploratory script, not one of the characterizations: it is not run
by ``make figures``, and writes to ``doc/figures/exploration/`` by default.
"""

from pathlib import Path

from astropy.table import Table, vstack
from matplotlib import pyplot, ticker
import numpy as np

from dc3.core import resolution
from dc3.core.velocity import SPEED_OF_LIGHT

import characterize_common


LAMBDA0 = 3540.5
"""The first input pixel centre, in angstroms, as in the MILES library."""

DLAMBDA = 0.9
"""The MILES-like input pixel size, in angstroms."""

MILES_NPIX = 4300
"""The length of a MILES spectrum, in pixels."""

MILES_FWHM = 2.51
"""The MILES instrumental FWHM, in angstroms (Falcón-Barroso et al. 2011)."""

GALAXY_VELSCALES = [7.5, 69.0]
"""Galaxy velocity scales marked on the figures: the DiskMass data and MaNGA, in km/s."""


def regime_fractions(u, lam0, lam1, fwhm):
    r"""
    Return the fraction of output pixels in each sampling regime.

    The output grid is logarithmic, so :math:`\ln s` is uniform across it, and
    the spectrum covers :math:`\ln s` from :math:`\ln(u\lambda_0/\lambda_c)` to
    :math:`\ln(u\lambda_1/\lambda_c)`.  Each fraction is the length of its part
    of that interval over the whole.

    Parameters
    ----------
    u : float or :class:`numpy.ndarray`
        The output velocity scale in units of :math:`v_c`.
    lam0, lam1 : float
        The wavelengths of the first and last input pixel, in angstroms.
    fwhm : float
        The instrumental FWHM, in input pixels.

    Returns
    -------
    tuple
        The fractions with :math:`s < 1` (covariant), with :math:`s > F/2`
        (undersampled), and within the window.
    """
    lamc = (lam0 + lam1) / 2
    length = np.log(lam1 / lam0)
    lo = np.log(np.asarray(u) * lam0 / lamc)
    hi = np.log(np.asarray(u) * lam1 / lamc)
    nyquist = np.log(fwhm / 2)
    covariant = np.clip(-lo / length, 0.0, 1.0)
    undersampled = np.clip((hi - nyquist) / length, 0.0, 1.0)
    inside = np.clip((np.minimum(hi, nyquist) - np.maximum(lo, 0.0)) / length, 0.0, 1.0)
    return covariant, undersampled, inside


def miles_like():
    r"""
    Return the quantities describing the MILES-like input.

    Returns
    -------
    tuple
        :math:`\lambda_0`, :math:`\lambda_1` and :math:`\lambda_c` in
        angstroms, the FWHM :math:`F` in input pixels, :math:`v_c` in km/s per
        pixel, and :math:`u_*`.
    """
    lam1 = LAMBDA0 + (MILES_NPIX - 1) * DLAMBDA
    lamc = (LAMBDA0 + lam1) / 2
    fwhm = MILES_FWHM / DLAMBDA
    vc = SPEED_OF_LIGHT * DLAMBDA / lamc
    return LAMBDA0, lam1, lamc, fwhm, vc, fwhm * lamc / (2 * lam1)


def auto_ratio(velscale):
    """
    Return the ``velscale_ratio`` that ``'auto'`` selects for the MILES-like input.

    The selection is made by
    :func:`~dc3.core.resolution.minimum_velscale_ratio`, from the narrowest
    instrumental dispersion in km/s, which for a FWHM constant in angstroms is
    at the red end.

    Parameters
    ----------
    velscale : float
        The galaxy velocity scale, in km/s per pixel.

    Returns
    -------
    int
        The selected ratio.
    """
    _, lam1, _, _, _, _ = miles_like()
    narrowest = SPEED_OF_LIGHT * MILES_FWHM / resolution.SIGMA_TO_FWHM / lam1
    return resolution.minimum_velscale_ratio(np.array([narrowest]), velscale)


def run(velscale):
    r"""
    Compute the regimes over the three views.

    Parameters
    ----------
    velscale : float
        The galaxy velocity scale for the first view, in km/s per pixel.

    Returns
    -------
    tuple
        Three :class:`astropy.table.Table` objects: the regimes against
        ``velscale_ratio`` for the MILES-like input at the given galaxy
        velocity scale; the window and the covariant fraction at :math:`u_*`
        against :math:`r` and :math:`F`; and the regimes against galaxy velocity
        scale and ``velscale_ratio`` for the MILES-like input.
    """
    lam0, lam1, _, fwhm, vc, _ = miles_like()

    ratio = np.arange(1, 9)
    u = velscale / (ratio * vc)
    covariant, undersampled, inside = regime_fractions(u, lam0, lam1, fwhm)
    window = Table({'velscale_ratio': ratio, 'u': u, 'covariant': covariant,
                    'undersampled': undersampled, 'inside': inside})
    window.meta['velscale'] = velscale
    window.meta['auto'] = auto_ratio(velscale)

    # The window in u, from s_min = 1 to s_max = F/2, with lambda_c / lambda_0
    # = (1 + r)/2 and lambda_c / lambda_1 = (1 + r)/(2r)
    r = np.geomspace(1.05, 4.0, 200)
    tables = []
    for f in [2.5, fwhm, 3.5, 5.0]:
        tables.append(Table({
            'fwhm': np.full(r.size, f),
            'r': r,
            'u_lower': (1 + r) / 2,
            'u_upper': f * (1 + r) / (4 * r),
            'covariant_at_optimum': np.clip(1 - np.log(f / 2) / np.log(r), 0.0, 1.0),
        }))
    by_span = vstack(tables)

    ug = np.geomspace(0.1, 5.0, 600)
    auto = np.array([auto_ratio(x * vc) for x in ug])
    tables = []
    for ratio in range(1, 7):
        covariant, undersampled, inside = regime_fractions(ug / ratio, lam0, lam1, fwhm)
        tables.append(Table({
            'velscale_ratio': np.full(ug.size, ratio),
            'ug': ug,
            'auto': auto,
            'covariant': covariant,
            'undersampled': undersampled,
            'inside': inside,
        }))
    by_ratio = vstack(tables)
    return window, by_span, by_ratio


def shade_bounds(ax, fwhm):
    """
    Shade the covariant and undersampled regions of :math:`s`.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        The axes, with :math:`s` on the y-axis.
    fwhm : float
        The instrumental FWHM, in input pixels.
    """
    ax.axhspan(1e-3, 1.0, color='C0', alpha=0.1, lw=0)
    ax.axhspan(fwhm / 2, 1e3, color='C3', alpha=0.1, lw=0)
    ax.axhline(1.0, color='C0', ls='--', lw=1.0)
    ax.axhline(fwhm / 2, color='C3', ls='--', lw=1.0)


def plain_log_ticks(axis, ticks):
    """
    Label a logarithmic axis with plain numbers at the given ticks.

    Parameters
    ----------
    axis : :class:`matplotlib.axis.Axis`
        The axis, already set to a logarithmic scale.
    ticks : list
        The major tick locations.
    """
    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    # Keep the minor ticks but not their labels, which would crowd the major ones
    axis.set_minor_formatter(ticker.NullFormatter())


def plot_window(window):
    r"""
    Plot :math:`s(\lambda)` against the window, and the regimes against ``velscale_ratio``.

    Parameters
    ----------
    window : :class:`astropy.table.Table`
        The first table from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    lam0, lam1, lamc, fwhm, vc, ustar = miles_like()
    velscale = window.meta['velscale']
    auto = window.meta['auto']
    rstar = velscale / (ustar * vc)
    fig, (top, bottom) = pyplot.subplots(2, 1, figsize=(6.5, 7.0), layout='constrained')

    # s(lambda) for the first few ratios, and for the non-integer optimum
    wave = np.linspace(lam0, lam1, 200)
    shown = window[:4]
    for i, row in enumerate(shown):
        label = rf"$R$ = {row['velscale_ratio']}, $u$ = {row['u']:.2f}"
        if row['velscale_ratio'] == auto:
            label += " ('auto')"
        top.plot(wave, row['u'] * wave / lamc, color=f'C{[4, 5, 6, 8][i]}', label=label)
    top.plot(wave, ustar * wave / lamc, color='0.3', ls=':',
             label=rf'$R_*$ = {rstar:.2f}, $u_*$ = {ustar:.2f}: red end at Nyquist')
    shade_bounds(top, fwhm)
    # Label each region just beside its bound
    top.text(0.02, 0.97, r'covariant, $s < 1$', transform=top.get_yaxis_transform(),
             color='C0', va='top')
    top.text(0.02, 1.03 * fwhm / 2, r'undersampled, $s > F/2$',
             transform=top.get_yaxis_transform(), color='C3', va='bottom')
    top.set_yscale('log')
    top.set_ylim(0.8 * np.amin(shown['u']) * lam0 / lamc,
                 1.4 * max(np.amax(shown['u']) * lam1 / lamc, fwhm / 2))
    plain_log_ticks(top.yaxis, [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    top.set_xlabel('Wavelength (A)')
    top.set_ylabel(r'$s = \Delta_{\rm out}/\Delta_{\rm in}$')
    top.set_title(rf'MILES-like: {MILES_NPIX} pixels of {DLAMBDA} A, FWHM {MILES_FWHM} A '
                  rf'($F$ = {fwhm:.2f}), $v_c$ = {vc:.1f} km/s')
    top.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncols=2, fontsize=7)

    for column, color, label in [
        ('covariant', 'C0', r'$s < 1$'),
        ('undersampled', 'C3', r'$s > F/2$'),
        ('inside', 'C2', r'$1 \leq s \leq F/2$'),
    ]:
        bottom.plot(window['velscale_ratio'], window[column], color=color)
        bottom.scatter(window['velscale_ratio'], window[column], color=color, lw=0,
                       label=label)
    bottom.axvline(auto, color='0.5', ls='--', lw=1.0, zorder=0)
    bottom.text(auto, 0.6, " 'auto'", transform=bottom.get_xaxis_transform(), fontsize=8)
    bottom.axvline(rstar, color='0.3', ls=':', lw=1.0, zorder=0)
    bottom.text(rstar, 0.4, r' $R_*$', transform=bottom.get_xaxis_transform(), fontsize=8)
    bottom.xaxis.set_major_locator(ticker.MultipleLocator(1))
    bottom.xaxis.set_minor_locator(ticker.NullLocator())
    bottom.set_xlabel(r'velscale_ratio, $R$')
    bottom.set_ylabel('Fraction of output pixels')
    bottom.set_title(rf'Pixels in each regime for a galaxy at $v_g$ = {velscale:g} km/s, '
                     r'$u = v_g/(R\,v_c)$')
    bottom.legend(loc='center right', fontsize=7)

    fig.suptitle('Resampling a linear spectrum onto a logarithmic grid: the bounds on $s$',
                 fontsize=10)
    return fig


def plot_span(by_span):
    """
    Plot the window in :math:`u` and the covariant fraction at :math:`u_*` against :math:`r`.

    Parameters
    ----------
    by_span : :class:`astropy.table.Table`
        The second table from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    lam0, lam1, _, fwhm_miles, _, _ = miles_like()
    fig, (top, bottom) = pyplot.subplots(2, 1, figsize=(6.5, 7.0), layout='constrained',
                                         sharex=True)

    for i, fwhm in enumerate(np.unique(by_span['fwhm'])):
        rows = by_span[by_span['fwhm'] == fwhm]
        label = rf'$F$ = {fwhm:.2f}' + (' (MILES)' if fwhm == fwhm_miles else '')
        top.plot(rows['r'], rows['u_upper'], color=f'C{i}', label=label)
        top.fill_between(rows['r'], rows['u_lower'], rows['u_upper'],
                         where=rows['u_upper'] >= rows['u_lower'], color=f'C{i}', alpha=0.15,
                         lw=0)
        bottom.plot(rows['r'], rows['covariant_at_optimum'], color=f'C{i}', label=label)
    rows = by_span[by_span['fwhm'] == fwhm_miles]
    top.plot(rows['r'], rows['u_lower'], color='0.3', ls='--',
             label=r'lower bound, $\lambda_c/\lambda_0$')
    for ax in [top, bottom]:
        ax.axvline(lam1 / lam0, color='0.5', ls=':', lw=1.0, zorder=0)
    top.text(lam1 / lam0, 0.03, ' MILES', transform=top.get_xaxis_transform(), fontsize=8)
    top.set_xscale('log')
    plain_log_ticks(top.xaxis, [1, 1.5, 2, 3, 4])
    top.set_ylabel(r'$u = v_t/v_c$')
    top.set_title(r'The window: $s_{\rm min} \geq 1$ above the dashed line, '
                  r'$s_{\rm max} \leq F/2$ below each solid line')
    top.legend(fontsize=7)

    bottom.set_xlabel(r'Wavelength span, $r = \lambda_1/\lambda_0$')
    bottom.set_ylabel(r'Fraction with $s < 1$ at $u_*$')
    bottom.set_title(r'The cost of a span wider than the window: $1 - \ln(F/2)/\ln r$')
    bottom.set_ylim(-0.03, 1.03)

    fig.suptitle('The sampling window against the wavelength span of the input', fontsize=10)
    return fig


def plot_ratio(by_ratio):
    """
    Plot the regimes against the galaxy velocity scale, for each ``velscale_ratio``.

    Parameters
    ----------
    by_ratio : :class:`astropy.table.Table`
        The third table from :func:`run`.

    Returns
    -------
    :class:`matplotlib.figure.Figure`
        The figure.
    """
    lam0, lam1, _, fwhm, vc, ustar = miles_like()
    fig, (top, middle, bottom) = pyplot.subplots(3, 1, figsize=(6.5, 9.0), layout='constrained',
                                                 sharex=True)

    rows = by_ratio[by_ratio['velscale_ratio'] == 1]
    top.plot(rows['ug'], rows['auto'], color='C0')
    top.set_ylabel(r'$R$')
    top.set_title(r"The ratio 'auto' selects: $R = \lceil v_g/v_{t,*}\rceil$, "
                  rf'$v_{{t,*}}$ = {ustar * vc:.1f} km/s')

    for ratio in np.unique(by_ratio['velscale_ratio']):
        rows = by_ratio[by_ratio['velscale_ratio'] == ratio]
        valid = np.asarray(rows['undersampled'] == 0)
        covariant = np.asarray(rows['covariant'])
        color = f'C{ratio - 1}'
        # Masked points break the line, so each style is drawn only where it applies
        middle.plot(rows['ug'], np.ma.masked_where(~valid, covariant), color=color,
                    label=f'$R$ = {ratio}')
        middle.plot(rows['ug'], np.ma.masked_where(valid, covariant), color=color,
                    ls=':', lw=1.0)
        bottom.plot(rows['ug'], rows['undersampled'], color=color, label=f'$R$ = {ratio}')

    # The covariant fraction along the ratio 'auto' selects
    rows = by_ratio[by_ratio['velscale_ratio'] == 1]
    covariant, _, _ = regime_fractions(rows['ug'] / rows['auto'], lam0, lam1, fwhm)
    middle.plot(rows['ug'], covariant, color='k', lw=0.8, label="'auto'")
    middle.set_ylabel(r'Fraction with $s < 1$')
    middle.set_title('Covariant pixels: solid where Nyquist sampled everywhere, dotted '
                     'where not')
    middle.legend(fontsize=7, ncols=2)

    bottom.set_xscale('log')
    plain_log_ticks(bottom.xaxis, [0.1, 0.2, 0.5, 1, 2, 5])
    bottom.set_xlabel(r'Galaxy velocity scale, $v_g/v_c$')
    bottom.set_ylabel(r'Fraction with $s > F/2$')
    bottom.set_title('Undersampled pixels')
    top.yaxis.set_major_locator(ticker.MultipleLocator(1))
    top.yaxis.set_minor_locator(ticker.NullLocator())

    for ax in [top, middle, bottom]:
        for velscale in GALAXY_VELSCALES:
            ax.axvline(velscale / vc, color='0.5', ls='--', lw=1.0, zorder=0)
    for velscale in GALAXY_VELSCALES:
        top.text(velscale / vc, 0.85, f' {velscale:g} km/s', transform=top.get_xaxis_transform(),
                 fontsize=8)

    fig.suptitle(rf'Choosing velscale_ratio for a MILES-like library ($v_c$ = {vc:.1f} km/s)',
                 fontsize=10)
    return fig


def main():
    """Run the exploration and write its products."""
    parser = characterize_common.get_parser(__doc__.split('\n')[1])
    parser.set_defaults(outdir=Path(__file__).resolve().parents[1] / 'figures' / 'exploration')
    parser.add_argument(
        '--velscale', type=float, default=69.0,
        help='Galaxy velocity scale, in km/s per pixel, for the log_sampling_window figure.'
    )
    args = parser.parse_args()
    characterize_common.apply_style()
    window, by_span, by_ratio = run(args.velscale)

    lam0, lam1, _, fwhm, vc, ustar = miles_like()
    covariant, _, _ = regime_fractions(ustar, lam0, lam1, fwhm)
    print(f'MILES-like: r = {lam1 / lam0:.3f}, F = {fwhm:.3f} px, window F/2 = {fwhm / 2:.3f}, '
          f'v_c = {vc:.2f} km/s')
    print(f'  u_* = {ustar:.3f} (v_t = {ustar * vc:.2f} km/s): fraction with s < 1 = '
          f'{covariant:.3f}')
    for velscale in GALAXY_VELSCALES:
        auto = auto_ratio(velscale)
        covariant, undersampled, _ = regime_fractions(velscale / (auto * vc), lam0, lam1, fwhm)
        print(f'  galaxy {velscale:g} km/s: auto R = {auto}, v_t = {velscale / auto:.2f} km/s, '
              f'fraction with s < 1 = {covariant:.3f}, undersampled = {undersampled:.3f}')
    print(f'  galaxy {args.velscale:g} km/s, by velscale_ratio:')
    for row in window:
        print(f"    R = {row['velscale_ratio']}: u = {row['u']:.3f}, s < 1: "
              f"{row['covariant']:.3f}, s > F/2: {row['undersampled']:.3f}, inside: "
              f"{row['inside']:.3f}")

    characterize_common.save(plot_window(window), window, 'log_sampling_window', args)
    characterize_common.save(plot_span(by_span), by_span, 'log_sampling_span', args)
    characterize_common.save(plot_ratio(by_ratio), by_ratio, 'log_sampling_ratio', args)


if __name__ == '__main__':
    main()
