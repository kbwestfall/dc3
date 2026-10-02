"""
Regression tests pinning the template-preparation characterization.

The full characterization is the ``characterize_*.py`` scripts in
``doc/scripts/``, which produce the committed figures.  These are small, fast
versions of their central findings, so that a change to ``prepare``, to
``Resample``, or to ``ppxf_util.varsmooth`` upstream that alters them fails
here rather than silently changing every instrumental correction.  See
verification item 9 of the port plan.

They pin the behaviour as measured, including where it departs from what
``prepare`` reports: the prepared templates are broader than
``PreparedTemplates.idsp`` says.  When that is corrected,
:func:`test_prepared_templates_follow_the_predicted_excess` should change with
it.
"""

import warnings

import numpy as np
import pytest
from scipy.special import ndtr

from dc3 import templates
from dc3.core import lsf, resample, resolution, sampling
from dc3.core.velocity import SPEED_OF_LIGHT
from dc3.spectra import GalaxySpectra


LOG10LAM0 = np.log10(4000.0)
DLOGLAM = 2e-5


def resampled_dvar(sigma_in, ratio, offset):
    """
    Resample a comb and return the median change in pre-pixelized variance.

    In units of the input pixel squared; see ``characterize_resampling.py``.
    """
    npix = 2000
    centers = np.arange(100, npix - 100, 61.37)
    flux = lsf.gaussian_comb(npix, centers, sigma_in)
    step = ratio * DLOGLAM
    first = LOG10LAM0 - DLOGLAM / 2 + (5 + 0.5 + offset) * step
    nout = int(npix / ratio) - 12
    resampled = resample.Resample(
        flux, x=10 ** (LOG10LAM0 + DLOGLAM * np.arange(npix)), inLog=True,
        newRange=[10 ** first, 10 ** (first + (nout - 1) * step)], newdx=step, newLog=True
    )
    out_centers = (LOG10LAM0 + DLOGLAM * centers - first) / step
    halfwindow = max(8 * sigma_in / ratio, 5.5) + 2
    keep = (out_centers > halfwindow) & (out_centers < nout - 1 - halfwindow)
    _, sigma = lsf.measure_comb(resampled.outy, out_centers[keep], sigma_in / ratio)
    return np.median(np.square(sigma * ratio) - sigma_in ** 2)


def test_aligned_integer_resampling_preserves_the_pre_pixelized_width():
    """
    Binning aligned pixels by an integer leaves the pre-pixelized width alone.

    Averaging ``k`` aligned pixels adds exactly the variance the wider output
    pixel accounts for, so this is the one case in which the ``mangadap``
    assumption -- that resampling does not change the line-spread function --
    holds.
    """
    dvar = resampled_dvar(1.0, 2.0, 0.0)
    assert abs(dvar) < 1e-3, \
        f'Aligned integer binning changed the pre-pixelized variance by {dvar:.4f} pixels^2'


def test_upsampling_adds_a_sixth_of_the_input_pixel_squared():
    """
    Resampling onto a much finer grid broadens the line by about Delta_in^2 / 6.

    Both the input's own pixel integration and ``Resample``'s treatment of each
    input pixel as flat survive as width on the finer grid.  A finer output grid
    does not remove it; this is why oversampling cannot fix Step 2.
    """
    dvar = resampled_dvar(1.0, 0.25, 0.37)
    assert abs(dvar - 1 / 6) < 0.015, \
        f'Upsampling broadened the pre-pixelized variance by {dvar:.4f}, not about 1/6 pixel^2'


@pytest.mark.parametrize(
    'ratio, offset, expected', [(2.0, 0.25, 0.25), (1.5, 0.2, 0.185), (0.5, 0.0, 0.125)]
)
def test_resampling_excess_follows_the_flat_pixel_prediction(ratio, offset, expected):
    r"""
    The excess follows from ``Resample`` treating each input pixel as flat.

    See ``characterize_resampling.predicted_dvar``: for :math:`s = p/q`, the
    excess is :math:`1/6 - [1 - 6\phi'(1 - \phi')]/(6q^2)` with
    :math:`\phi' = {\rm frac}(p x)`.  The cases are an integer ratio midway
    between alignments, where it peaks at 1/4, and the two half-integer
    ratios at their maximum and minimum.  Lines of 4 input pixels are wide
    enough that the Gaussian fit follows the second moment.
    """
    dvar = resampled_dvar(4.0, ratio, offset)
    assert abs(dvar - expected) < 0.005, \
        f'Resampling at s = {ratio}, offset {offset} added {dvar:.4f} input pixels^2, ' \
        f'not the predicted {expected}'


@pytest.mark.parametrize('velscale', [20.0, sampling.velscale(1e-4), 120.0])
def test_nyquist_velscale_ratio_has_a_closed_form_for_a_constant_fwhm(velscale):
    r"""
    For a line-spread function of constant FWHM, the Nyquist ratio follows from F, r and g.

    See ``characterize_velscale.py``: with :math:`g = v_g/v_c`,
    :func:`~dc3.core.resolution.minimum_velscale_ratio` gives
    :math:`\max(1, \lceil 4gr/[F(1 + r)] \rceil)` for a line-spread function of
    :math:`F` pixels on a linear grid, and the pixel-size ratio at each output
    pixel is :math:`(g/R)\,\lambda/\lambda_c`.  The cases select ratios 1, 2
    and 3 for a MILES-like library at its own resolution.

    This checks the ratio for a given resolution, not what ``prepare`` selects
    with ``velscale_ratio = 'auto'``.  That tests the resolution after
    matching, which is broader wherever the galaxy's is, so it can be smaller.
    """
    lam0, dlam, npix, fwhm = 3540.5, 0.9, 4300, 2.51
    linear = sampling.SpectralGrid.from_linear_spacing(lam0, dlam, npix)
    lam1 = linear.wave[-1]
    lamc = (lam0 + lam1) / 2
    r = lam1 / lam0
    f = fwhm / dlam
    g = velscale * lamc / (SPEED_OF_LIGHT * dlam)

    idsp = SPEED_OF_LIGHT * fwhm / resolution.SIGMA_TO_FWHM / linear.wave
    ratio = resolution.minimum_velscale_ratio(idsp, velscale)
    assert ratio == max(1, int(np.ceil(4 * g * r / (f * (1 + r))))), \
        f'minimum_velscale_ratio gave {ratio}, not the closed-form ratio for r = {r:.3f}, ' \
        f'g = {g:.3f}'

    # The exact pixel-size ratio across the output grid, from its borders
    dloglam = sampling.dloglam_from_velscale(velscale / ratio)
    nout = int(np.floor(np.log10(lam1 / lam0) / dloglam))
    log = sampling.SpectralGrid.from_log_spacing(np.log10(lam0), dloglam, nout)
    exact = np.diff(log.borders) / dlam
    predicted = g / ratio * log.wave / lamc
    assert np.allclose(exact, predicted, rtol=1e-3, atol=0), \
        'The pixel-size ratio departs from (g/R) lambda / lambda_c by more than 0.1%'


def test_varsmooth_adds_the_variance_of_linear_interpolation():
    """
    Step 1 adds about Delta^2 / 6 once the kernel is resolved.

    ``varsmooth`` moves the spectrum onto its stretched coordinate by linear
    interpolation, a convolution with a triangle of variance Delta^2 / 6; its
    ``oversample`` refines the stretched grid but not the input samples, so it
    cannot remove this.
    """
    npix = 3000
    velscale = sampling.velscale(DLOGLAM)
    kernel = np.linspace(1.5, 2.5, npix)
    match = resolution.ResolutionMatch(
        kernel * velscale, 0.0, np.zeros(npix, dtype=bool), velscale, 0.1
    )
    centers = np.arange(150, npix - 150, 61.37)
    convolved = resolution.apply_kernel(
        lsf.gaussian_comb(npix, centers, 1.0), match, oversample=4
    )
    expected = np.sqrt(1.0 + np.square(np.interp(centers, np.arange(npix), kernel)))
    _, sigma = lsf.measure_comb(convolved, centers, expected)
    dvar = np.median(np.square(sigma) - np.square(expected))
    assert abs(dvar - 1 / 6) < 0.015, \
        f'varsmooth added {dvar:.4f} pixels^2 of variance, not about 1/6'


def miles_comb_and_manga_galaxy():
    """
    A MILES-like comb and a MaNGA-like galaxy, as in ``characterize_preparation.py``.

    Returns the library, the line centres in native pixels, and the galaxy.
    """
    lam0, dlam, npix, fwhm = 3540.5, 0.9, 4300, 2.51
    sigma_pix = fwhm / resolution.SIGMA_TO_FWHM / dlam
    centers = np.arange(30, npix - 30, 23.37)
    grid = sampling.SpectralGrid.from_linear_spacing(lam0, dlam, npix)
    library = templates.TemplateLibrary(
        lsf.gaussian_comb(npix, centers, sigma_pix), grid,
        idsp=SPEED_OF_LIGHT * sigma_pix * dlam / grid.wave
    )
    galaxy_grid = sampling.SpectralGrid.from_log_spacing(np.log10(3622.0), 1e-4, 4563)
    galaxy = GalaxySpectra(
        np.ones(galaxy_grid.npix), galaxy_grid,
        idsp=80.0 - 25.0 * (galaxy_grid.wave - 4000.0) / 5000.0
    )
    return library, centers, galaxy


def test_corrected_templates_carry_the_reported_resolution():
    r"""
    With the excess corrected for, prepared lines are as broad as reported.

    See ``characterize_correction.py``.  MILES-like templates are prepared for
    MaNGA-like data at ``velscale_ratio = 2`` with ``correct_lsf_excess`` on,
    and each line is compared with ``PreparedTemplates.idsp``.  That is the
    target where the matching succeeds, and what the templates actually carry
    where it cannot.  Uncorrected, the lines are about 0.34
    :math:`\Delta_{\rm tpl}^2` broader; corrected, the mean departure is a few
    thousandths, either way.
    """
    library, centers, galaxy = miles_comb_and_manga_galaxy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = templates.prepare(library, galaxy, velscale_ratio=2)
    line_wave = library.wave[0] + 0.9 * centers
    out_centers = (np.log10(line_wave) - prepared.log10lam0) / prepared.dloglam
    keep = (out_centers > 30) & (out_centers < prepared.npix - 30) \
        & (line_wave > galaxy.wave[0])
    reported = np.interp(out_centers[keep], np.arange(prepared.npix), prepared.idsp[0])
    _, sigma = lsf.measure_comb(prepared.flux[0], out_centers[keep], reported / prepared.velscale)
    dv_tpl = SPEED_OF_LIGHT * 0.9 / line_wave[keep]
    dvar = (np.square(sigma * prepared.velscale) - np.square(reported)) / np.square(dv_tpl)
    unmatched = prepared.match.unmatched[np.round(centers[keep]).astype(int)]
    assert np.any(unmatched) and np.any(~unmatched), \
        'This test needs lines both where the matching succeeds and where it cannot'
    for label, select in [('matched', ~unmatched), ('unmatched', unmatched)]:
        residual = np.mean(dvar[select])
        assert abs(residual) < 0.015, \
            f'Corrected {label} lines depart from the reported dispersion by {residual:+.4f} ' \
            'Delta_tpl^2 on average, against a few thousandths characterized'


def test_prepared_templates_follow_the_predicted_excess():
    r"""
    MILES-like templates prepared for MaNGA-like data are as broad as predicted.

    See ``characterize_preparation.py``.  ``prepare`` is run at
    ``velscale_ratio = 2`` and ``varsmooth_oversample = 2``, without the
    correction for the excess.  The prepared lines are broader than the
    dispersion Step 1 applied by the sum of the two steps' predicted excesses:

    - Characterization 2's :math:`E_1` for ``varsmooth``;
    - Characterization 3's expectation of 1/6 for the resampling.

    The comparison uses one comb position, over the lines where the matching
    succeeds, above about 4000 angstroms.
    """
    library, centers, galaxy = miles_comb_and_manga_galaxy()
    lam0, dlam, npix = library.wave[0], 0.9, library.npix
    sigma_pix = 2.51 / resolution.SIGMA_TO_FWHM / dlam
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prepared = templates.prepare(
            library, galaxy, velscale_ratio=2, varsmooth_oversample=2, correct_lsf_excess=False
        )

    # The lines where matching succeeds, with a full fitting window
    kernel = np.maximum(prepared.match.kernel_sigma_pixels, resolution.VARSMOOTH_MIN_SIG)
    matched = ~prepared.match.unmatched[np.round(centers).astype(int)]
    line_wave = lam0 + dlam * centers
    out_centers = (np.log10(line_wave) - prepared.log10lam0) / prepared.dloglam
    keep = matched & (out_centers > 30) & (out_centers < prepared.npix - 30)
    k = np.interp(centers[keep], np.arange(npix), kernel)
    dv_tpl = SPEED_OF_LIGHT * dlam / line_wave[keep]
    applied = np.sqrt(np.square(sigma_pix * dv_tpl) + np.square(k * dv_tpl))
    _, sigma = lsf.measure_comb(prepared.flux[0], out_centers[keep], applied / prepared.velscale)
    dvar = (np.square(sigma * prepared.velscale) - np.square(applied)) / np.square(dv_tpl)

    # The prediction, Characterization 2's E_1 plus Characterization 3's 1/6
    predicted = resolution.varsmooth_excess(k, 2, np.amax(kernel) / k) + 1 / 6
    residual = np.mean(dvar - predicted)

    assert np.all(dvar > 0), \
        'Every prepared line should be broader than the dispersion Step 1 applied'
    assert abs(residual) < 0.01, \
        f'The prepared lines depart from the predicted excess by {residual:+.4f} ' \
        'Delta_tpl^2 on average, against about +0.002 characterized'


# ----------------------------------------------------------------------
# Jumps in the library; see characterize_splices.py
# ----------------------------------------------------------------------
SPLICE = 4025.0


def splice_borders(case):
    """The pixel borders of the library, 3985-4065 A: spliced in sampling, or regular."""
    steps = [0.02, 0.03] if case == 'sampling' else [0.02, 0.02]
    first = 3985.0 + steps[0] * np.arange(int(np.round((SPLICE - 3985.0) / steps[0])) + 1)
    second = first[-1] + steps[1] * np.arange(1, int(np.round((4065.0 - SPLICE) / steps[1])) + 1)
    return np.concatenate([first, second])


def integrated_lines(borders, centers, sigma_kms):
    """Lines Gaussian in velocity, integrated exactly over each pixel's own borders."""
    lnb = np.log(borders)
    flux = np.zeros(borders.size - 1, dtype=float)
    for center, sigma in zip(centers, sigma_kms):
        flux += np.diff(ndtr((lnb - np.log(center)) * SPEED_OF_LIGHT / sigma))
    return flux / np.diff(borders)


def splice_departures(case, ncomb=40, spacing=400.0):
    """
    Measure prepared lines near a jump against those of the same section clear of it.

    A cut-down characterize_splices.py: the same experiment over a narrower
    range, so that enough combs to sample the distance from the jump at under
    a kernel dispersion stay cheap.

    Returns each line's distance from the jump in kernel dispersions, the
    departure of its width from its section's baseline, the shift of its
    centre in km/s, and whether prepare masked it.
    """
    borders = splice_borders(case)
    wave = (borders[1:] + borders[:-1]) / 2
    idsp = np.full(wave.size, 3.0) if case == 'sampling' else np.where(wave < SPLICE, 3.0, 12.0)
    grid = sampling.SpectralGrid('irregular', wave.size, wave=wave, borders=borders)
    # A resolution constant but for a dip at the blue end, which sets dvar_inst
    # and leaves a matching kernel of about 13 km/s everywhere near the jump
    galaxy = GalaxySpectra(
        np.ones(590), sampling.SpectralGrid.from_log_spacing(np.log10(3995.0), 1.09e-5, 590),
        idsp=np.where(np.arange(590) < 20, 15.0, 20.0)
    )
    flag = 'SAMP_JUMP' if case == 'sampling' else 'RES_JUMP'
    step = spacing / SPEED_OF_LIGHT
    rows = []
    for offset in np.arange(ncomb) / ncomb:
        centers = np.exp(np.arange(np.log(3998.0) + offset * step, np.log(4052.0), step))
        sigma = np.full(centers.size, 3.0) if case == 'sampling' \
            else np.where(centers < SPLICE, 3.0, 12.0)
        library = templates.TemplateLibrary(
            integrated_lines(borders, centers, sigma), grid, idsp=idsp
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prepared = templates.prepare(library, galaxy)
        kernel = prepared.match.kernel_sigma
        join = np.searchsorted(wave, SPLICE)
        sigma_kernel = max(kernel[join - 1], kernel[join])
        true = (np.log10(centers) - prepared.log10lam0) / prepared.dloglam
        reported = np.interp(true, np.arange(prepared.npix), prepared.idsp[0])
        fitted, width = lsf.measure_comb(prepared.flux[0], true, reported / prepared.velscale)
        masked = getattr(prepared.mask, flag)[0][np.round(true).astype(int)]
        rows.append(np.column_stack([
            SPEED_OF_LIGHT * np.log(centers / SPLICE) / sigma_kernel,
            width * prepared.velscale / reported - 1,
            (fitted - true) * prepared.velscale,
            masked,
        ]))
    distance, excess, shift, masked = np.vstack(rows).T
    departure, offset = np.zeros_like(excess), np.zeros_like(shift)
    for side in [distance < 0, distance > 0]:
        near = side & (np.absolute(distance) > 12) & (np.absolute(distance) < 40)
        departure[side] = excess[side] - np.median(excess[near])
        offset[side] = shift[side] - np.median(shift[near])
    keep = np.absolute(distance) < 40
    return distance[keep], departure[keep], offset[keep], masked[keep].astype(bool)


@pytest.mark.parametrize('case', ['sampling', 'resolution'])
def test_guard_band_covers_where_prepared_lines_depart(case):
    """
    Outside the guard band, a jump leaves the prepared lines as they would be without it.

    Measured by characterize_splices.py: at either kind of jump the width
    departs by more than 0.1 per cent only within about 2.5 kernel dispersions,
    inside the default band of three.  Pinned here at a coarser sampling of the
    distance from the jump, and at looser tolerances, so that a change that
    widens the affected region beyond the band fails.  The resolution step is
    also required to depart strongly inside the band, so the test cannot pass
    by the jump having no effect at all.
    """
    distance, departure, shift, masked = splice_departures(case)
    assert np.any(masked) and np.any(~masked), 'This test needs lines both inside and outside'
    assert np.amax(np.absolute(departure[~masked])) < 0.003, \
        f'Outside the {case} guard band a prepared line departs in width by ' \
        f'{np.amax(np.absolute(departure[~masked])):.2%}; the band no longer covers the jump'
    assert np.amax(np.absolute(shift[~masked])) < 0.05, \
        f'Outside the {case} guard band a prepared line is shifted by ' \
        f'{np.amax(np.absolute(shift[~masked])):.3f} km/s; the band no longer covers the jump'
    if case == 'resolution':
        assert np.amax(np.absolute(departure[masked])) > 0.03, \
            'Inside the band the resolution step should broaden or narrow lines markedly'
