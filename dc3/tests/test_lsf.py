"""
Tests for :mod:`~dc3.core.lsf`.

The width-measurement tools are what the template-preparation characterization
measures *with*, so these check that they measure what they claim to: the
pixel-integrated fit must recover the pre-pixelized width exactly, and the fit
at pixel centres must reproduce the published post-pixelized bias.
"""

import numpy as np
import pytest

from dc3.core import lsf
from dc3.pkg.exceptions import DC3Error


def law2021(sigma_post):
    """
    Law et al. (2021, AJ 161, 52), Figure 4: sigma_post / sigma_pre.

    Their fourth-order polynomial in the post-pixelized width, fit over
    0.8 <= sigma_post <= 2.0 pixels.
    """
    return np.polynomial.polynomial.polyval(
        sigma_post, [1.5612, -1.2153, 1.0567, -0.4199, 0.0633]
    )


# ----------------------------------------------------------------------
# Profiles
# ----------------------------------------------------------------------
def test_pixelated_gaussian_conserves_flux():
    """Integrated over pixels, the line sums to its flux."""
    x = np.arange(201, dtype=float)
    for sigma in [0.3, 1.0, 5.0]:
        profile = lsf.pixelated_gaussian(x, center=100.3, sigma=sigma, flux=2.5)
        assert np.isclose(np.sum(profile), 2.5), \
            f'A pixel-integrated line of width {sigma} does not sum to its flux'


def test_pixelated_gaussian_approaches_the_sampled_one_when_resolved():
    """
    For a well-resolved line, integrating over a pixel hardly matters.

    The two differ at second order in the pixel size, so the difference falls
    as the line broadens; this checks the pixel-integrated profile is not
    offset or mis-normalized relative to the sampled one.
    """
    x = np.arange(401, dtype=float)
    narrow = np.amax(np.absolute(
        lsf.pixelated_gaussian(x, 200.0, 2.0) - lsf.sampled_gaussian(x, 200.0, 2.0)
    ))
    broad = np.amax(np.absolute(
        lsf.pixelated_gaussian(x, 200.0, 20.0) - lsf.sampled_gaussian(x, 200.0, 20.0)
    ))
    assert broad < narrow / 100, \
        'The pixel-integrated profile does not converge to the sampled one as the line broadens'


def test_gaussian_comb_is_the_sum_of_its_lines():
    """The comb is exactly the sum of its lines, despite evaluating each locally."""
    centers = [20.4, 55.9, 81.1]
    comb = lsf.gaussian_comb(100, centers, [1.0, 1.5, 2.0])
    x = np.arange(100, dtype=float)
    direct = sum(lsf.pixelated_gaussian(x, c, s) for c, s in zip(centers, [1.0, 1.5, 2.0]))
    assert np.allclose(comb, direct, rtol=0, atol=1e-15), \
        'Evaluating each line only near its centre changed the comb'


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------
@pytest.mark.parametrize('sigma', [0.6, 1.0, 2.0])
@pytest.mark.parametrize('phase', [0.0, 0.3, 0.5])
def test_pixelated_fit_recovers_the_pre_pixelized_width(sigma, phase):
    """
    The default fit returns the pre-pixelized width, at any pixel phase.

    This is the property everything measured with the fitter depends on.
    """
    y = lsf.pixelated_gaussian(np.arange(101, dtype=float), center=50 + phase, sigma=sigma)
    flux, center, fitted = lsf.fit_line(y, 50 + phase + 0.2, 1.2 * sigma)
    assert np.isclose(fitted, sigma, rtol=1e-10, atol=0), \
        'The pixel-integrated fit did not recover the pre-pixelized width'
    assert np.isclose(center, 50 + phase, rtol=0, atol=1e-10), \
        'The pixel-integrated fit did not recover the line centre'
    assert np.isclose(flux, 1.0, rtol=1e-10), 'The pixel-integrated fit did not recover the flux'


@pytest.mark.parametrize('sigma_pre', [0.8, 1.0, 1.4, 1.9])
def test_sampled_fit_reproduces_the_published_post_pixelized_bias(sigma_pre):
    """
    Fitting at pixel centres gives the post-pixelized width Law et al. (2021) report.

    An independent, published check on the setup: their Figure 4 relation,
    measured the same way on Monte Carlo profiles.  Agreement to half a
    percent is well within the scatter they show over pixel phase.
    """
    y = lsf.pixelated_gaussian(np.arange(101, dtype=float), center=50.3, sigma=sigma_pre)
    _, _, post = lsf.fit_line(y, 50.3, sigma_pre, pixelated=False)
    assert post > sigma_pre, 'The post-pixelized width should exceed the pre-pixelized one'
    assert np.isclose(post / sigma_pre, law2021(post), rtol=5e-3), \
        'The post-pixelized bias disagrees with Law et al. (2021), Figure 4'


def test_fixed_center_is_held_and_the_width_still_recovered():
    """With the centre fixed, it is returned unchanged and the width still fits."""
    y = lsf.pixelated_gaussian(np.arange(101, dtype=float), center=50.37, sigma=1.3)
    _, center, sigma = lsf.fit_line(y, 50.37, 1.0, fix_center=True)
    assert center == 50.37, 'A fixed centre was not returned unchanged'
    assert np.isclose(sigma, 1.3, rtol=1e-10), 'The width was not recovered with the centre fixed'


def test_window_spans_an_odd_number_of_pixels_about_the_nearest_pixel():
    """
    The window reaches ``window // 2`` pixels either side of the nearest pixel.

    Checked through the error raised when it runs off the spectrum, which
    reports the window it tried.
    """
    y = np.zeros(30)
    with pytest.raises(DC3Error, match=r'\[-2, 9\)'):
        lsf.fit_line(y, 3.4, 1.0, window=11)
    with pytest.raises(DC3Error, match=r'\[-2, 9\)'):
        lsf.fit_line(y, 3.4, 1.0, window=10)


def test_measure_comb_fits_every_line():
    """Each line in a comb is measured independently."""
    centers = np.array([30.2, 71.7, 113.5])
    sigmas = np.array([1.0, 1.5, 2.0])
    comb = lsf.gaussian_comb(150, centers, sigmas)
    fitted_centers, fitted = lsf.measure_comb(comb, centers + 0.1, sigmas * 1.1)
    assert np.allclose(fitted, sigmas, rtol=1e-8), 'The comb widths were not all recovered'
    assert np.allclose(fitted_centers, centers, rtol=0, atol=1e-8), \
        'The comb centres were not all recovered'
