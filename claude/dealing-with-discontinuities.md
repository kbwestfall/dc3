# Dealing with discontinuities in template sampling and resolution

**Status:** agreed on review, and implemented. Several details changed during implementation —
most importantly the size and placement of the `varsmooth` workaround in §4 — and are recorded in
the implementation record rather than here.

**Companion documents:** [`dc3-python-port-plan.md`](dc3-python-port-plan.md) (Phase 2, "Spectral
resolution" and "Resampling") and [`dc3-python-implementation.md`](dc3-python-implementation.md)
(the `SpectralGrid` work this builds on). The port plan stays fixed; the execution of this plan,
and its departures from the port plan, are recorded in the implementation record.

## Motivation

Some template libraries are spliced from sections with different pixel sampling and different
spectral resolution. `SpectralGrid` already represents such an irregular grid. What does not yet
exist is any guard against the places where the preparation pipeline assumes the sampling and the
resolution vary smoothly. The aim is to guard against jump discontinuities without modelling them,
under a piecewise-smooth assumption with a small number of well-articulated exceptions.

## 1. Where smoothness is currently assumed

| Site | Assumption | Failure at a jump |
|---|---|---|
| `varsmooth` (via `apply_kernel`) | pixel size from a centred `np.gradient` of the coordinate | at a sampling jump the pixel either side of the join gets the average of the two pixel sizes, so its kernel width in pixels is wrong |
| `varsmooth` stretched coordinate | stretch by σ_max/σ(x) makes the kernel constant | at a resolution jump the stretch changes abruptly; within a kernel footprint of the jump the effective kernel is asymmetric and of intermediate width |
| `match_resolution` | a scalar `velscale` converts `epsilon_sigma` from pixels to km/s | on an irregular grid the minimum kernel is ε pixels in only one part of the spectrum, and too narrow or too wide in pixels elsewhere |
| `prepare` Step 2: `Resample` | input borders from a single regular grid | on a spliced grid an output pixel straddling a join mixes the two segments |
| `prepare` Step 2: `np.interp` of `matched_idsp` and of `unmatched` | both vary linearly between pixels | at a straddling pixel the dispersion is interpolated to a value neither segment has |
| `check_pixelization`, `minimum_velscale_ratio` | a scalar `velscale` | the pixel velocity is wrong for part of an irregular grid |

`_fiducial_on_template_grid` also interpolates, but it acts on the galaxy's resolution, which is
on a regular log grid; it is covered by an exception in §5.

Three rows are removed outright: the first by passing `varsmooth` pixel coordinates (§4), and the
third and sixth by the per-pixel pixel velocity (§6). The other three are guarded rather than
removed: the stretch at a resolution jump, and the output pixels of Step 2 that straddle a join,
whose flux and interpolated dispersion both mix two segments (§4).

## 2. The model: a piecewise-smooth library

A template library is a sequence of **segments** separated by **breaks**. A break is a pixel
boundary at which either

- the pixel size jumps, which can happen only on an irregular grid, or
- the instrumental dispersion `idsp` jumps, which can happen on any grid.

Within a segment, everything the current code assumes still holds, subject to exception 3 of §5.
A regular library with a smooth `idsp` is the one-segment case, and its behaviour changes only by
the numerically irrelevant perturbation of §4.

## 3. Detection

Two pure functions, each returning the indices of the breaks. Each index is the first pixel of a
new segment.

- **`sampling.SpectralGrid.breaks(tol)`** detects sampling jumps. It flags adjacent pixels where
  |Δ_{i+1}/Δ_i − 1| > tol. Log and linear grids always return no breaks. For a smoothly
  irregular grid (for example, a linear grid treated in log) the pixel size changes by only about
  10⁻⁴ per pixel, while a real splice changes it by tens of percent. Default: **0.01**, matching
  the resolution tolerance.

  **Float32 floor.** Wavelengths stored in float32 put noise of about
  `np.finfo(np.float32).eps` · λ/Δλ ≈ 1.2×10⁻⁷ · λ/Δλ into the ratio of adjacent pixel sizes:
  6×10⁻⁴ at λ/Δλ = 5,000, which is harmless, but about 10⁻² at λ/Δλ ≈ 10⁵, which would produce
  false breaks everywhere. The tolerance applied is therefore
  `max(tol, 4 · eps32 · max(λ/Δλ))` — the same kind of floor `sampling_type` already applies
  through `_departure_limit`, which uses a factor of 1. The factor of 4 gives margin because a
  ratio of two differences carries roughly twice the error of one. At λ/Δλ ≈ 10⁵ the floor is
  about 5%, still well below a real splice.

  **Too many splices.** A spliced library has only a handful of sections. If the breaks divide
  the grid into **more than 5 segments**, `prepare` warns that the sampling is probably noisy
  rather than spliced, and that `sampling_jump_tol` should be increased. The limit of 5 is
  hard-coded in the function that issues the warning, not a parameter or a module-level
  constant, and is explained in its docstring. It applies to sampling breaks only; resolution
  breaks are not counted.
- **`resolution.idsp_breaks(idsp, tol)`** detects resolution jumps. It flags
  |σ_{i+1} − σ_i| / min(σ_i, σ_{i+1}) > tol. A smooth resolution vector changes by about 10⁻⁴
  per pixel, and a measured, noisy one by more, so the default has to sit above that noise and
  below the jumps that matter. Default: **0.01** (1% per pixel).

The segments are the union of the two sets of breaks, and `prepare` logs how many it finds.

**Configuration.** All three tolerances describe the library, so they belong in
`TemplateLibraryPar`:

| Key | Meaning | Default |
|---|---|---|
| `sampling_tol` | the existing `tol` of `sampling_type` and `SpectralGrid.from_vector`, now reachable from the config | 1e-3 px |
| `sampling_jump_tol` | tolerance on the adjacent-pixel size ratio, subject to the float32 floor | 0.01 |
| `idsp_jump_tol` | tolerance on the relative adjacent-pixel change in `idsp` | 0.01 |

## 4. Handling: pixel coordinates in Step 1, joins masked

### Step 1 in pixel coordinates

`varsmooth` uses its coordinate argument `x` only through a centred `np.gradient`, which it uses
to convert `sig_x` to pixels. `apply_kernel` therefore passes

```
x     = np.arange(npix, dtype=float)
sig_x = match.kernel_sigma_pix      # = match.kernel_sigma / grid.pixel_velocity; see §6
```

so that each pixel's kernel width is computed from its own size — taken from the grid's borders
via `SpectralGrid.pixel_velocity` — rather than from centred differences of the pixel centres. On
a regular log grid this is the same computation as passing `loglam`: the two agree to 3×10⁻¹² on a
varying kernel. What it buys:

- the averaged gradient at a sampling join (§1, first row) disappears;
- a smooth but irregular segment no longer relies on centred differences of centres matching the
  pixel widths (§5, exception 3);
- the choice of coordinate disappears — log, linear and irregular grids take the same call;
- `sig_x` is in the units of `epsilon_sigma` and of `varsmooth`'s 0.1-pixel clip, and is exactly
  the per-pixel `kernel_sigma_pix` of §6.

What it does **not** buy: inside `varsmooth` the convolution runs in pixel space whatever `x` is.
A kernel whose footprint crosses a sampling join treats pixels of two sizes as equal, so on the
far side its physical width is scaled by the ratio of the pixel sizes; and the stretch at a
resolution jump behaves as before. Hence the guard band below.

### Workaround for the `varsmooth` off-by-one

Pixel coordinates make the off-by-one broadening documented in the `dc3.core.resolution` module
docstring certain rather than latent. When `sig_x` is exactly uniform, `varsmooth`'s internal
stretched grid spans a whole number of pixels, comes out one sample short, and the round trip
through it broadens the result. Today `dc3` avoids it only because `np.gradient(loglam)` carries
floating-point noise. With `x = arange` the gradient is exact, and on a log grid `pixel_velocity`
is exactly constant, so any kernel that is constant in km/s — both `idsp` vectors constant, which
is realistic for a same-instrument setup — would trigger it every time. A 0.5-pixel uniform kernel
on a σ = 1 px line measures 1.332 px instead of √1.25 = 1.118.

The bug has been reported upstream and a fix is promised for the next `ppxf` release. Until then,
`apply_kernel` perturbs `sig_x` by a numerically irrelevant amount:

- **Size.** One interior element is reduced by a factor (1 − 10⁻⁶). The perturbation lengthens
  the stretched grid by about ε, which only registers if it exceeds the rounding error of a sum
  over about npix·`oversample` terms, roughly npix·`oversample`·2×10⁻¹⁶. Measured with the uniform
  kernel above, at `oversample=1`:

  | npix | ε = 0 | 10⁻¹² | 10⁻⁹ | 10⁻⁶ |
  |---|---|---|---|---|
  | 2,000 | 1.314 | 1.118 | 1.118 | 1.118 |
  | 20,000 | 1.306 | 1.306 | 1.118 | 1.118 |
  | 200,000 | 1.305 | 1.305 | 1.118 | 1.118 |

  10⁻¹² — the figure currently quoted in the module docstring — is lost by 20,000 pixels, and
  10⁻⁹ would fail near a million pixels at `oversample=4`. 10⁻⁶ leaves many orders of magnitude of
  margin. On a varying kernel it changes the output by at most 1.2×10⁻⁷ of the peak, and on the
  uniform kernel it moves the measured width only in the sixth decimal place.
- **Always applied**, not only when `sig_x` is exactly uniform: a nearly uniform `sig_x` can
  trigger the bug when its variation is lost in the same sum. At worst the internal grid gains
  one sample.
- **Which element.** An interior one, because the first element does not change the grid length,
  and not the maximum, because reducing a unique maximum shrinks every other element's
  contribution and can shorten the grid. The smallest interior element,
  `1 + np.argmin(sig_x[1:-1])`, satisfies both.
- **Removal.** `test_resolution.py` already reproduces the bug; that test fails once the upstream
  fix lands, which is the signal to remove the workaround. Once the fixed release number is known,
  the workaround can instead be skipped for that version and later.

With `oversample=4` the corrected width is 1.184 rather than 1.118. That is the Step-1
linear-interpolation broadening already found by the characterization, not this bug: with
`oversample=1` and a uniform kernel the internal grid coincides with the input pixels, so the
interpolation happens not to broaden.

### Handling the joins

1. **Step 1 runs once over the whole spectrum**, in pixel coordinates. Processing each segment
   separately would only exchange a distorted kernel at the join for an edge effect, and both are
   masked, so it buys nothing once the gradient error is gone.
2. **Step 2 resamples the whole spectrum once**, with `Resample(xBorders=grid.borders)`. The
   borders come from the `SpectralGrid`, not from an assumption of a regular grid; `Resample`
   already accepts `xBorders`. `idsp` and `unmatched` are interpolated within each segment only.
   An output pixel that straddles a join is not interpolated across it: it takes its `idsp` from
   whichever segment covers more of the pixel, and its `unmatched` flag is the OR of the two
   sides. Either way the pixel lies within the `SAMP_JUMP` guard band.
3. **A guard band around every join is masked**, with a separate bit for each kind of break,
   both appended to `SpectrumBitMask`:

   ```
   'SAMP_JUMP': 'Pixel is within the guard band of a jump in the template sampling'
   'RES_JUMP': 'Pixel is within the guard band of a jump in the template resolution'
   ```

   The two are kept apart because their causes and their consequences differ. At a sampling join
   the kernel footprint spans pixels of two sizes, and Step 2 has output pixels that straddle the
   join; at a resolution jump the pixels are uniform and the fault is `varsmooth`'s stretch. A
   user, or the Phase 3 mask transcription, can therefore tell which fault a masked pixel carries,
   and the characterization (§7, step 6) can measure the affected width of each independently. A
   join at which both jump sets both bits.

   Each band's width is ±`jump_guard` × (local kernel σ) in native pixels on each side, and at
   least one native pixel. The local kernel σ is the larger of the kernel σ, in native pixels, on
   the two sides of the join. When no matching was performed (a resolution vector is missing),
   there is no kernel and no convolution, so the band falls to its one-pixel minimum; that
   minimum still matters for `SAMP_JUMP`, because of the straddling output pixel. The band is
   then mapped onto the prepared grid. The kernel footprint is
   exactly the region that edge effects and the stretch can reach. Default:
   **`jump_guard = 3`**, in `TemplatePar`, since it is a preparation choice. One width serves both
   bits for now; if the characterization shows the two kinds of break need different widths, it
   splits into one parameter per bit. Unlike `UNMATCHED`, neither bit can be turned off. Phase 3
   consumes both on the same path as `UNMATCHED`.

**`dvar_inst` is still the minimum over all pixels.** Each pixel's own σ_T and σ_G are correct;
only the convolution near a join is compromised. Excluding guard pixels from the minimum would
make δ depend on the guard width for no gain.

An earlier draft processed Step 1 once per segment, to keep `varsmooth`'s gradient from averaging
across a sampling join. Pixel coordinates remove that error directly, so per-segment processing
was dropped.

## 5. Exceptions: what is deliberately not guarded

These go into the docstrings and onto the Phase 8 assumptions page.

1. **The galaxy.** It must be on a regular log grid, and constructing the galaxy spectra raises
   if it is not; only the template library may be irregular.
   Jumps in its resolution enter only through the fiducial, a statistic across spectra. The
   fiducial is interpolated onto the template grid, a pointwise operation that depends on
   smoothness only within one pixel of a jump. That residual is not guarded.
2. **Changes below the tolerance are treated as smooth.** A resolution ramp spread over several
   pixels is not a break, however large its total change.
3. **Within a segment, the kernel is assumed locally uniform in pixels over its footprint.**
   With pixel coordinates (§4) each pixel's kernel width is exact, but the convolution treats
   neighbouring pixels as equal in size. On a smooth but irregular segment the error this leaves
   is second order in the drift of the pixel size across the footprint. This is accepted, not
   guarded.
4. **Joins are masked, never modelled.** There is no correction for the effective kernel at a
   join, and no stitching, consistent with the no-deconvolution rule.
5. **The characterization correction, if one is adopted, is calibrated on uniform grids.** A
   spliced characterization case is added (§7), but the correction is not made per segment.
6. **Spliced-region detection as library metadata** — naming segments, reporting their
   provenance — remains the future enhancement already logged. The breaks here are what it would
   build on.

## 6. `match_resolution` with a per-pixel pixel velocity

`velscale` becomes an array v_pix(λ), taken from `SpectralGrid.pixel_velocity`:

```
δ²        = min_λ [ res_match(λ) − (ε · v_pix(λ))² ]
kernel(λ) = sqrt(res_match(λ) − δ²)     ⇒  kernel(λ) ≥ ε·v_pix(λ) everywhere, with equality at the argmin
```

Consequences:

- A scalar still works: it is broadcast to a constant array, and results on regular grids are
  numerically unchanged.
- `ResolutionMatch.velscale` becomes an array, and `kernel_sigma_pix` divides pixel by pixel.
- `apply_kernel` no longer needs a coordinate at all: the `ResolutionMatch` already carries the
  per-pixel `kernel_sigma_pix`, which it passes to `varsmooth` against `x = np.arange(npix)`,
  with the perturbation of §4. Its `loglam` argument is removed.
- `check_pixelization` takes an array for the templates.
- `minimum_velscale_ratio` keeps the galaxy's scalar, since its output grid is regular.

## 7. Order of work

1. **Migrate `Spectra` onto `SpectralGrid`** (galaxy log only; library any kind), updating
   `deredshift`, `PreparedTemplates.velocity_offset` and `prepare`. Remove `grid_from_wave`,
   replacing its use in `test_resample.py` with `sampling_type`. Drop `TemplateLibraryPar.log10`,
   and add `sampling_tol` (§3), which the library's `SpectralGrid.from_vector` call needs.
2. **Generalize `match_resolution` to a per-pixel velocity** (§6). The existing tests should pass
   unchanged; new tests cover an irregular grid.
3. **Move `apply_kernel` to pixel coordinates and add the off-by-one workaround** (§4). Tests:
   - on a regular log grid, the pixel-coordinate call matches the `loglam` call to round-off when
     both are given the same (perturbed) kernel, and differs from the unperturbed `loglam` call
     by no more than the 10⁻⁷ level of §4;
   - a uniform kernel in km/s on a log grid gives the correct width at 2,000, 20,000 and 200,000
     pixels, and at `oversample` of 1 and 4 (the regression test for the workaround);
   - the existing test reproducing the upstream bug is kept, as the signal to remove the
     workaround.

   Correct the "one part in 10¹²" statement in the `dc3.core.resolution` module docstring, which
   holds only for short spectra.
4. **Add the detection functions** (§3) and their config keys, tested against the
   `spliced_borders()` helper and a synthetic resolution step. Also test that a float32-stored
   high-resolution grid (λ/Δλ ≈ 10⁵) yields no false sampling breaks, and that a grid broken into
   more than 5 segments raises the too-many-splices warning while one with 5 does not.
5. **Add the segment-bounded interpolation in Step 2, and the `SAMP_JUMP` and `RES_JUMP` bits
   and their guard bands** (§4). Tests:
   - on a one-segment library, `prepare` agrees with its output after step 3 to round-off (the
     switch to `Resample(xBorders=...)` may move the borders at that level), and neither bit is
     set;
   - a sampling join sets only `SAMP_JUMP`, a resolution step only `RES_JUMP`, and a join with
     both sets both;
   - the band widths follow the larger kernel σ at each join, and fall to one pixel when no
     matching was performed;
   - a straddling output pixel takes the `idsp` of its majority segment and the OR of the
     `unmatched` flags;
   - on a spliced library, the prepared widths away from the join match the existing
     uniform-grid characterization.
6. **Add a spliced case to the characterization scripts** — a comb across a sampling join and,
   separately, across a resolution step — measuring the width of the band that is really affected
   by each. This tests whether `jump_guard = 3` is right, and whether the two kinds of break need
   different widths. The figure goes to `doc/figures/characterization/`. Test that each guard
   band covers the region where the measured width departs.

## Decisions

All settled on review.

1. **Step 1 runs once over the whole spectrum in pixel coordinates**, with the off-by-one
   workaround (§4), rather than once per segment.
2. **The resolution-jump criterion** is a relative first difference, with a default tolerance of
   0.01.
3. **The guard width `jump_guard`** is in units of local kernel σ, with a default of 3, shared by
   both bits. The spliced characterization case (step 6) is to confirm or revise it, and to show
   whether it should split into one width per bit.
4. **The parameters** are as proposed: `sampling_tol`, `sampling_jump_tol` and `idsp_jump_tol` in
   `TemplateLibraryPar`, and `jump_guard` in `TemplatePar`.
5. **`sampling_jump_tol` defaults to 0.01**, subject to a float32 floor (§3).
6. **More than 5 sampling segments raises a warning** advising a larger `sampling_jump_tol`, with
   the 5 hard-coded for now (§3).
7. **The float32 floor uses a factor of 4**: `max(tol, 4 · eps32 · max(λ/Δλ))` (§3).
8. **A straddling output pixel** takes the `idsp` of the segment covering more of it and the OR
   of the `unmatched` flags (§4).
9. **The local kernel σ of a guard band** is the larger of the two sides', and the band falls to
   one pixel when no matching was performed (§4).
10. **The guard-band coverage test belongs to step 6**, where the spliced characterization
    provides the measured widths (§7).

## Change log

- **2026-09-30** — Initial plan: where smoothness is assumed, a piecewise-smooth model of the
  library, detection of breaks in sampling and resolution, per-segment Step 1 with a masked guard
  band at each join, the exceptions that are not guarded, a per-pixel pixel velocity in
  `match_resolution`, and the order of work. Awaiting review.
- **2026-09-30** — Step 1 moves to pixel coordinates: `varsmooth` is passed
  `x = np.arange(npix)` and the per-pixel `kernel_sigma_pix`, which removes the averaged-gradient
  error at a sampling join and replaces per-segment processing with a single call plus the guard
  band (decision 1 settled). Added the workaround for `varsmooth`'s off-by-one on a uniform kernel,
  which pixel coordinates would otherwise trigger for any kernel constant in km/s: one interior
  element of `sig_x` reduced by a factor (1 − 10⁻⁶), with the measurements showing why the
  10⁻¹² figure in the module docstring fails beyond a few thousand pixels, and removal tied to the
  upstream fix. Revised §1, exception 3 and §6 to match, and added the change as step 3 of the
  order of work.
- **2026-09-30** — Replaced the single `DISCONTINUITY` mask bit with two, `SAMP_JUMP` and
  `RES_JUMP`, so sampling and resolution breaks are flagged separately; a join where both jump
  sets both. `jump_guard` is shared by the two for now, with the spliced characterization to
  decide whether it splits. Updated the order of work and its tests to match, and corrected the
  step number cited in decision 3.
- **2026-09-30** — Decisions 2–4 settled on review: a 1% tolerance for `idsp_breaks`,
  `jump_guard = 3`, and the proposed parameter names and homes. "Decisions for review" becomes
  "Decisions", and the status is updated. The default for `sampling_jump_tol` is recorded as the
  one value still open.
- **2026-09-30** — Closed the last open value: `sampling_jump_tol` defaults to 0.01, subject to a
  float32 floor of the same form `sampling_type` uses, since float32 wavelengths put noise of
  order 10⁻² into the pixel-size ratio of a λ/Δλ ≈ 10⁵ grid. Added a warning when the sampling
  breaks divide the grid into more than 5 segments, advising a larger `sampling_jump_tol`, with
  the limit hard-coded for now. Added both to the tests of step 4, and marked the plan agreed.
- **2026-09-30** — Consistency pass. §1 now says which rows are removed (1, 3, 6) and which
  guarded (2, 4, 5); §2 says sampling breaks arise only on irregular grids, and that a
  one-segment library changes only by the §4 perturbation; §3's configuration covers all three
  tolerances; §4's code sets `sig_x` from `match.kernel_sigma_pix`, and its table states
  `oversample=1`; §6 drops `apply_kernel`'s `loglam` argument, since the match carries the
  per-pixel kernel; exception 1 no longer says `Spectra` raises, since the library shares it.
  In the order of work, `sampling_tol` moves to step 1 where it is first needed, and the step 3
  and step 5 tests no longer claim agreement beyond the perturbation and round-off. Removed
  stale "once agreed" and "proposed" wording.
- **2026-09-30** — Closed the four gaps found in the consistency pass (decisions 7–10): the
  float32 floor's factor is 4; a straddling output pixel takes the `idsp` of its majority segment
  and the OR of the `unmatched` flags; a guard band's local kernel σ is the larger of the two
  sides', falling to a one-pixel band when no matching was performed; and the guard-band coverage
  test moves from step 5 to step 6, which supplies the measured widths. Step 5 gains tests for the
  band widths and the straddling pixel.
- **2026-09-30** — Implemented, steps 1–6. The companion-document note no longer says the
  decisions will be folded into the port plan, which stays fixed. Departures from this plan made
  during implementation — a `GalaxySpectra` class, the `varsmooth` workaround raising the largest
  element by a hundredth of a stretched sample rather than reducing the smallest by 10⁻⁶, and
  others — are recorded in the implementation record's change log for this date.
- **2026-10-02** — `jump_guard` is renamed `convolution_mask_growth`, and also grows each run of
  pixels masked in the library itself, flagged with the new `TPL_MASKED` bit. The text above
  keeps the old name, as written; see the implementation record's change log for this date.
