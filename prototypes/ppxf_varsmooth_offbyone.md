# `ppxf_util.varsmooth` applies the wrong kernel width when `sig_x/gradient(x)` is exactly uniform

## Summary

In the vector-`sig_x` branch of `varsmooth`, the internal stretched grid is built
with **one sample fewer than the input** whenever the kernel width in pixels is
*exactly* uniform. Interpolating onto that shortened grid and back broadens the
result, so the convolution applies a wider kernel than the one requested.

A requested uniform kernel of 0.5 pixels is applied as **0.87 pixels**.

The effect is not confined to small kernels, and it is not a sampling effect. It
is triggered purely by an integer count, and a perturbation of one part in
10<sup>12</sup> to a single element of `sig_x` — which changes nothing physical —
restores the correct result exactly.

## Environment

| Package | Version |
|---|---|
| `ppxf` | 9.5.0 |
| `numpy` | 2.5.3 |
| `scipy` | 1.18.1 |
| Python | 3.13.14 |

## Reproduction

`ppxf_varsmooth_offbyone.py` is attached; it depends only on `numpy`, `scipy`
and `ppxf`. It convolves a well-resolved Gaussian (σ = 4 px) and recovers the
applied kernel as the second moment of the output differenced in quadrature
against the input.

Results from various scenarios are given in the Table below.  The input has `N`
= 800 samples. `unif` is whether `sig_x/np.gradient(x)` is exactly constant, and
`span − (N−1)k` is the amount by which the stretched coordinate exceeds
`(N−1)*oversample` — the quantity that decides whether `ceil` rounds up to the
correct sample count.

| | case | unif | span − (N−1)k | `n` | requested | applied |
|---|---|---|---|---|---|---|
| | **Well above the 0.1-pixel clip, so undersampling is not in play** | | | | | |
| ❌ | uniform 0.5 px, exactly uniform abscissa | True | `0.000e+00` | 799 | 0.500 | **0.867** |
| ✅ | … one interior element changed by 1e-12 | False | `1.023e-12` | 800 | 0.500 | 0.500 |
| ✅ | … or the same request on a log abscissa | False | `5.497e-09` | 800 | 0.500 | 0.500 |
| | **Oversampling does not help; it trades one error for another** | | | | | |
| ❌ | uniform 0.5 px, `oversample=2` | True | `0.000e+00` | 1598 | 0.500 | **0.707** |
| ❌ | uniform 0.5 px, `oversample=4` | True | `0.000e+00` | 3196 | 0.500 | **0.661** |
| ❌ | uniform 0.5 px, `oversample=8` | True | `0.000e+00` | 6392 | 0.500 | **0.649** |
| | **Below the clip, which forces exact uniformity and so always triggers it** | | | | | |
| ❌ | uniform 0.001 px on a log abscissa | True | `0.000e+00` | 799 | 0.001 | **0.714** |
| ❌ | uniform 0.05 px on a log abscissa | True | `0.000e+00` | 799 | 0.050 | **0.714** |
| ❌ | uniform 0.09 px on a log abscissa | True | `0.000e+00` | 799 | 0.090 | **0.714** |
| ✅ | uniform 0.1 px on a log abscissa (at the clip) | False | `1.864e-09` | 800 | 0.100 | 0.100 |
| | **Control: the scalar `sig_x` branch builds no stretched grid at all** | | | | | |
| ✅ | scalar `sig_x = 0.5` | — | — | — | 0.500 | 0.500 |

Every failing row has `span − (N−1)k` of **exactly zero**; every passing row has a
tiny positive excess that happens to push `ceil` up to the correct count.

## Root cause

```python
sig = sig_x/np.gradient(x)
sig = sig.clip(0.1)
sig_max = np.max(sig)*oversample
xs = np.cumsum(sig_max/sig)
n = int(np.ceil(xs[-1] - xs[0]))        # <-- one short
x_new = np.linspace(xs[0], xs[-1], n)
y_new = interp(x_new, xs, y.T)
```

`xs` holds `len(x)` entries, so `xs[-1] - xs[0]` spans `len(x) - 1` unit
intervals. Since `sig_max/sig >= 1` by construction, that span is always at least
`len(x) - 1`, **with equality if and only if `sig` is exactly uniform**. In that
case `ceil` returns `len(x) - 1`, so `x_new` holds one sample fewer than the
input and the round trip through `interp` loses resolution.

When `sig` is not exactly uniform the span exceeds the integer, `ceil` rounds up,
and the count is correct — which is why the defect is normally invisible. Whether
it fires is decided by floating-point noise in `np.gradient(x)`.

## Proposed fix

A range spanning *S* unit intervals needs *S* + 1 samples:

```python
n = int(np.ceil(xs[-1] - xs[0])) + 1
```

In the generic non-uniform case this adds one sample to a grid that is already
denser than the input, which costs nothing. In the uniform case it restores the
identity mapping, which is what the passing rows above demonstrate: they differ
from the failing rows *only* in the value of `n`.

## When it happens

Two ways an exactly uniform `sig` arises in ordinary use:

1. **Any request below the 0.1-pixel clip.** `clip` replaces every smaller value
   with the same literal, which forces exact uniformity. This is why a request of
   0.001, 0.05 or 0.09 px is applied as ≈ 0.71 px rather than as the requested
   value. Note that this is also not the 0.1 px that the clip on its own would
   suggest.

2. **A genuinely constant kernel on a regular abscissa.** This is not exotic: it
   is what one gets when two spectral resolutions differ by a constant, which is
   a common case in resolution matching. It escapes only when `np.gradient` of
   the abscissa happens to carry floating-point noise.

In both cases the routine returns a plausible spectrum, so a downstream
instrumental-resolution budget computed from the *requested* kernel is wrong by
the difference.

## Notes

- **The result does not depend on the probe.** The measurement above uses a

  Gaussian evaluated at the pixel centres. Repeating it with a Gaussian
  *integrated over the pixel width* gives the same answer to four decimal
  places, for probe widths from 1 to 8 pixels:

  | Probe σ (px) | Profile | Applied, as-is | Applied, `n` restored |
  |---|---|---|---|
  | 1.0 | sampled | 0.8667 | 0.5000 |
  | 1.0 | pixelated | 0.8667 | 0.5000 |
  | 4.0 | sampled | 0.8666 | 0.5000 |
  | 4.0 | pixelated | 0.8666 | 0.5000 |
  | 8.0 | sampled | 0.8662 | 0.5000 |
  | 8.0 | pixelated | 0.8662 | 0.5000 |

  This is expected: pixelization adds dx²/12 to the variance of the input and
  the output alike, so it cancels in the quadrature difference. It is included
  to show that the effect is not an artefact of an unphysical line profile.

- **`oversample` does not help.** It reduces the error but converges to ≈ 0.65 px
  rather than to the requested 0.5 px, because it replaces the missing-sample
  error with interpolation onto a grid finer than the input.
- The scalar-`sig_x` branch is unaffected; it builds no stretched grid.
- The 0.1-pixel clip itself is sound as a bound on the coordinate stretch. The
  issue is only that reaching it makes `sig` exactly uniform and so exposes the
  count.
