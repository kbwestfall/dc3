# DC3 Port Plan — Reassessment Before Implementation

**Status: ✅ complete and applied (2026-09-16).** All questions are resolved, and every decision
recorded here has been written into [`dc3-python-port-plan.md`](dc3-python-port-plan.md) and
[`dc3-original-implementation.md`](dc3-original-implementation.md). This document is retained as
the *reasoning* behind those revisions — the two companion documents state the decisions, this
one says why.

§§1–5 hold the findings and decisions; §7 and §8 record the two rounds of follow-up questions and
their answers.

A full re-read of [`dc3-original-implementation.md`](dc3-original-implementation.md) (the C++
survey, "report") and [`dc3-python-port-plan.md`](dc3-python-port-plan.md) (the port plan,
"plan"), looking for logical inconsistencies and unanswered design questions that should be
settled before code is written. Responses from the `notes` file are folded in below, marked
**▶ Response**, with the resulting decision and its consequences for the two documents.

Tags: **[resolved]** decided, ready to apply · **[edit]** mechanical, no judgement needed ·
**[verify]** checked against source or paper, result recorded.

"Report §N" and "plan, Phase N" point to the two companion documents; "W11" is Westfall,
Bershady & Verheijen (2011, ApJS 193, 21) and "W19" is Westfall et al. (2019, AJ 158, 231).

---

## 1. Resolution matching, resampling, and the sign of the instrumental offset **[resolved]**

### 1.1 What I originally claimed, and what was wrong with it

The plan (Phase 2, "Do not match resolution by default") says DC3's `res_base` is
"structurally *precisely* the W19 scheme" needing only promotion from a numerical hack. I
showed that to be wrong and described the mechanism as "backwards", implying a defect.

**The mechanics I described are correct and are not disputed. The interpretation was wrong.**
Both signs are legitimate physics, and DMS genuinely operated in the negative regime by
necessity. The correction is recorded in §1.3.

### 1.2 The mechanics (unchanged, for the record)

From `Correlation::match_resolution` (report §8.3):

```cpp
res_match[j] = SQR(spl0.sample(wave+zw)) - SQR(spec[2].resolution(i));   // σ_G(λ)² − σ_T(λ)²
min          = min(min, res_match[j]);
res_base     = min < SQR(MIN_SIG_RESMATCH) ? SQR(MIN_SIG_RESMATCH) - min : 0.0;
for (...) res_match[i] = sqrt(res_match[i] + res_base);                 // the convolution kernel
res_base     = res_base > 0 ? sqrt(res_base) : 0.0;
```

The template's post-convolution resolution is `sqrt(σ_T² + σ_G² − σ_T² + res_base²) =
sqrt(σ_G² + res_base²)` — uniformly **broader** than the galaxy by `res_base`. `VDF::getbtsp`
then applies `ap[VSIG] = sqrt(p[VSIG]² − base_sig²)`, so the total model broadening is
`sqrt(σ_G² + σ_fit²)`: **`res_base` cancels exactly** and `σ_fit = σ_*`.

Because `sqrt(σ_fit² − base_sig²)` requires `σ_fit ≥ res_base`, `res_base` is a **floor on the
measurable σ**. The fitter says so explicitly (report §8.1, after the main loop): if
`p[VSIG]² − base_sig² < EPSDP`, fix σ to 0 and recursively re-`fit()`. Production `RES_BASE`
ran **0 to 9.50 km/s** (report §16.4), against W11 §4 simulations reaching down to
`σ_m = 5` km/s.

### 1.3 ▶ Response — this is a framing difference, not a defect

> *"The difference between W11 and W19 is, I would say, largely an issue of framing. In MaNGA,
> we had templates that were higher resolution than the galaxy spectra, so we could adopt a
> positive δσ_inst². In the DiskMass survey, the templates were observed with the same
> instrument as the galaxy data, but slightly different conditions in the system could lead to
> small resolution differences. This meant that we tended to need to lower the resolution in the
> templates to keep a pedestal offset, such that we were effectively adopting a negative
> δσ_inst². In the DMS, this floor was seen as an observational limitation; i.e., a dispersion
> measurement below the floor meant that the dispersion was not measurable with the
> observational data."*

**Decision: the port must track both signs, and must warn when the offset is negative.**
Negative is not a bug — it is the regime a same-instrument survey lands in — but it imposes a
floor on measurable σ that the user has to know about.

This also confirms W11 §5 was itself in the negative regime: the demonstration used HD 167042
observed with SparsePak, the same instrument as the galaxy data.

### 1.4 ▶ Response — nomenclature: use a signed variance

> *"Perhaps we should adopt different nomenclature than δσ_inst², adopting something like a
> variance so we don't need to keep tracking the square and avoid the ambiguity of the imaginary
> number when the quantity is negative."*

**Decision: adopt a signed variance in (km/s)².** Proposed definition, with `σ_T'` the template
resolution *after* preparation:

```
dvar_inst  ≡  σ_G² − σ_T'²            (signed, km²/s², may be negative)
σ_obs²     =  σ_*² + dvar_inst
σ_*²       =  σ_obs² − dvar_inst
```

| `dvar_inst` | Meaning | Regime | Consequence |
|---|---|---|---|
| **> 0** | template sharper than galaxy | MaNGA / W19 | `σ_obs ≥ sqrt(dvar_inst) > 0`; measurement held away from the zero boundary |
| **= 0** | matched | current C++ with `res_base` cancelled | no pedestal either way |
| **< 0** | template broader than galaxy | DMS / W11 | floor on measurable σ at `sqrt(|dvar_inst|)`; **warn** |

No square roots in the bookkeeping, no imaginary intermediates, one sign convention everywhere.
▶ Confirmed (§7.3): the name `dvar_inst` and this sign convention are both as intended.

### 1.5 ▶ Response — three hard constraints on the design

> *"The fitting code should never manipulate the flux distribution within the provided galaxy
> spectrum."*

**Constraint 1 — the galaxy's flux distribution is never redistributed.** This is absolute. It
resolves §5.5 (de-redshifting) below and constrains what "preparation" can mean. Note the
asymmetry it implies: *template* preparation freely resamples and convolves, because templates
are model components; the galaxy is data and is left alone.

> *"Any resolution matching cannot involve a deconvolution because it leads to amplified noise."*

**Constraint 2 — no deconvolution, ever.** This bounds how sharp the prepared template may be
made: the convolution kernel `sqrt(res_match(λ) − δ²)` must be real everywhere, so
`δ² ≤ min_λ res_match`.

> *"Because of the difference in computational expense between convolution by a
> wavelength-independent vs. wavelength-dependent sigma, the model evaluations internal to the
> fitting iterations prefer to use a wavelength-independent kernel. This means the resolution
> matching step is a preparatory step before the fit is started."*

**Constraint 3 — resolution matching is preparation, never part of the cost function.** Inside
the fit, the broadening kernel is a single wavelength-independent σ. This is a stronger and
better reason than the representational one I gave in the old §2, and it has a large
consequence for the ppxf dependency (§4.7).

### 1.6 ▶ Response — preparation couples resolution matching and resampling

> *"Preparation of the template spectra is a combination of both resolution matching **and**
> resampling … These two operations are not independent. If the difference in resolution is
> large enough, resampling the template spectra to match the galaxy data will dramatically
> degrade its effective resolution (Nyquist sampling of the line-spread function). This both
> limits the practical resolution offset allowed for the fit-prepared templates, and may
> motivate an oversampling of the template spectra that are then downgraded during the fitting
> process. The latter is possible in `ppxf`, and DC3 could do the same in the construction of
> the B spectrum."*

This is a constraint I had missed entirely. `δ` is bounded from **two** directions:

| Bound | Source | Effect |
|---|---|---|
| `δ² ≤ min_λ res_match` | no deconvolution (Constraint 2) | template cannot be made sharper than it is |
| prepared template LSF must stay Nyquist-sampled on its own grid | pixelization | template cannot be made sharper than the grid can carry |

The second bound is relaxed by **oversampling the template** relative to the galaxy — ppxf's
`velscale_ratio`. DC3 can do the same when constructing the broadened template, broadening on
the oversampled grid and binning down afterwards.

▶ **`velscale_ratio` defaults to 1, with optional automatic determination.**

> *"I like the option of having the code determine a reasonable value automatically. Despite the
> additional complexity, this will save users from having to determine a good value and defer to
> DC3's approach. A reasonable approach is to ensure that the FWHM of the spectral LSF is sampled
> by at least 2 pixels."*

**Criterion: FWHM of the prepared template's LSF ≥ 2 pixels**, i.e. `σ_T' ≥ 2/√(8 ln 2) ≈ 0.85`
pixels. Note this is numerically the same Nyquist threshold as the C++'s `minsig = 0.85 px`
block-replication floor (report §8.4) — the two are the same criterion applied to different
objects, which is worth pointing out in the docs rather than leaving as a coincidence.

### 1.7 ▶ Response — template preparation, in two steps, with four parameters

> *"I was referring to (a) … The template preparation steps should be as follows (this does not
> necessarily follow the algorithm currently used in the C++ DC3 code)."*

**The exposed floor is on the convolution kernel — option (a) of the old §7.1.** The pointer to
`Sasuke` was a red herring; the governing reference is W19 Appendix A, implemented as
[`SpectralResolution.GaussianKernelDifference`](../mangadap/mangadap/util/resolution.py#L416).

**Template preparation is a two-step pipeline, run once per execution:**

| Step | Operation |
|---|---|
| **1** | Resolution matching of the templates to a **fiducial** galaxy resolution |
| **2** | Resampling to the galaxy's sampling, up to an integer oversampling (`velscale_ratio`) |

Two properties of this that the plan must state explicitly, because both make
velocity-dispersion corrections unavoidable:

- **Step 1 is only ever approximate.** One matching operation is applied to all templates against
  a *fiducial* galaxy resolution. Unless every galaxy spectrum has identical resolution — MaNGA's
  did not, and neither will DMS data — the fiducial will not match any individual spectrum
  exactly.
- **Step 2 also alters the effective template resolution**, and *"this is something `mangadap`
  has never accounted for."* Resampling onto a coarser grid degrades the LSF. This is a **new
  requirement** for the port; §8.2 answers how it enters `dvar_inst`.

> *"Regardless of the resolution matching approach, calculation of velocity-dispersion
> corrections will always be an important step."*

**Matching follows W19 Appendix A with three caveats:**

1. **Pedestal offset (W19 Appendix A, option 2) is the default**, with the pedestal set by the
   user's `epsilon_sigma`.
2. **The target resolution is adjusted to permit a *negative* `dvar_inst`**, so as to avoid or
   minimise the span of template regions that cannot reach the target.
3. **Regions that cannot reach the target resolution** are masked or not, at the user's choice.

**Four parameters:**

| Parameter | Meaning | Default |
|---|---|---|
| `velscale_ratio` | integer number of prepared-template pixels per galaxy pixel | `1` |
| `epsilon_sigma` | minimum viable dispersion of the Gaussian convolution kernel, **in pixels** | `0.1` — see below |
| `sigma_floor` | maximum pedestal allowed to accommodate template regions of lower resolution than the galaxy; sets the minimum (most negative) `dvar_inst` | `0` |
| `mask_unmatched_sres` | whether template regions that cannot meet the target resolution are masked (caveat 3) | `False` |

#### `epsilon_sigma` must match `varsmooth`'s internal clip

Cappellari (2017) §4.3 argues that, mathematically, the analytic Fourier transform of the
Gaussian makes the convolution accurate even for a severely undersampled kernel — so
`epsilon_sigma = 0` is defensible in principle. **But the implementation does not honour it.**
`ppxf_util.varsmooth` contains:

```python
sig = sig_x/np.gradient(x)
sig = sig.clip(0.1)   # Clip to >=0.1 pixels
```

so any requested kernel σ below **0.1 pixels** is silently raised to 0.1. This is not a
cosmetic detail: if `epsilon_sigma` were set below the clip, the code would believe it had
applied a smaller kernel than `varsmooth` actually applied, and **`dvar_inst` would be wrong by
the difference** — the prepared template would be broader than the bookkeeping claims, biasing
`σ_*` low.

**Decision: default `epsilon_sigma` to the `varsmooth` clip (0.1 pixels), and say why in the
parameter description**, so the user sees that the floor is an implementation limit of the
convolution routine rather than a free choice. Validation should reject, or warn and raise, any
value below the clip. Units in pixels also match `GaussianKernelDifference`'s `min_sig_pix`.

▶ **Hard-code 0.1 with an asserting test** (§8.3): *"That should be sufficient to catch changes
to `ppxf` by the developers, but does not require a redetermination of the limit on every
import."* So `epsilon_sigma`'s floor is a module constant, and a unit test convolves a
sub-0.1-pixel kernel through `varsmooth` and asserts the realised width — a behavioural check
that fails loudly if upstream moves the literal.

**Two warnings to emit:**

- the resampled template is not Nyquist-sampled → advise increasing `velscale_ratio`;
- template regions cannot be matched to the target resolution.

#### `epsilon_sigma` is a two-sided target, not a one-sided floor

▶ **Confirmed (§8.1).** Read strictly, `GaussianKernelDifference`'s option 2 only ever *raises*
the kernel, so `dvar_inst ≤ 0` in every case and the positive-pedestal regime that motivated
this change would be unreachable. The intended behaviour is the two-sided form: force the
**minimum** kernel σ to *equal* `epsilon_sigma`, lowering it where it would otherwise be larger.

```
δ²         = min_λ res_match − epsilon_sigma²        (signed)
kernel(λ)  = sqrt( res_match(λ) − δ² )               ⇒  min_λ kernel = epsilon_sigma exactly
dvar_inst  = δ²                                      > 0 when the template is sharper everywhere
```

This reproduces `GaussianKernelDifference` whenever `min_λ res_match < epsilon_sigma²` and
extends it in the other direction, so it is a **superset of the W19 behaviour, not a
contradiction** — worth saying in the docs, since anyone comparing against `mangadap` will
notice the difference. `sigma_floor` caps the negative excursion; `mask_unmatched_sres` disposes
of what remains.

Because `epsilon_sigma` defaults to the smallest usable value (0.1 px), the default behaviour is
to make `δ` **as large as the no-deconvolution bound allows** — which is the stated intent
("users will generally want to make δ as large as possible") and needs no number from the user.

#### When is matching *not* performed?

Your phrasing — *"when the user requests resolution matching"* — implies a toggle, while §2
concludes that an unmatched path is not worth supporting. These reconcile if "off" means **no
resolution information is available**, rather than "available but declined":

- If either `sres` vector is absent, matching is impossible. The C++ already handles this by
  returning early from `match_resolution` with `res_base = 0` (report §8.3). The port should do
  the same, set `dvar_inst = 0`, and **warn that the reported `σ_obs` is uncorrected** — it is
  then the user's responsibility to interpret it.
- If both are present, matching runs. A boolean to decline it is not offered, because the
  resulting wavelength-dependent `δσ_inst(λ)` cannot be represented in the forward model (§2).

Recording this explicitly because it is otherwise the kind of thing that gets decided silently
during implementation.

### 1.8 ▶ Response — `δ` is per-run, not per-spectrum

> *"We should expect δ to be the same for all spectra to be fit. I.e., for each execution of the
> main fitting script, we should only manipulate the resolution and sampling of the template
> spectra once."*

**Decision: template preparation happens exactly once per execution.** This removes the
per-spectrum-`δ` complication I raised in the old §5.2, and it fits the "prepare once, fit N
times" pattern already in the plan's workflow-integration section. `min_λ res_match` is then
taken over the galaxy set as a whole — worth stating explicitly, since the safe choice is the
minimum over *all* galaxy spectra, not a representative one.

### 1.9 ▶ Response — the undetermined-σ case needs hooks, not automation

> *"In the limit where the velocity dispersion is 'undetermined', it should be up to the user
> whether or not they want to refit the spectra by fixing the measured velocity dispersion to be
> 0. In terms of hooks in the code, this means adding the ability to fix the kinematic
> parameters to an input value and only fitting a subset of the spectra provided. At least for
> now, I don't think this means … adding an optional path that automatically performs this
> refitting if the user toggles a boolean input parameter."*

**Decision: no automatic refit path.** Instead, two general capabilities:

1. **Fix kinematic parameters to input values.** Per-parameter, per-spectrum.
2. **Fit a subset of the provided spectra.** A selection mask.

Together these let the user do the σ ≡ 0 refit by hand, and they are more broadly useful. Note
this makes the C++'s automatic behaviour — "if `p[VSIG]² − base_sig² < EPSDP`, fix σ to 0 and
recursively re-`fit()`" — a **deliberate removal**, to be recorded on the Phase 8 differences
page. Both hooks are confirmed general (§7.4); how the per-spectrum constraints are *passed*
is the subject of §8.4.

### 1.10 Consequences for the two documents

Phase 2's "Do not match resolution by default" subsection needs rewriting end to end:

- Drop "DC3 already has the machinery — it is `res_base`" (wrong; it is the negative case).
- Drop "generalize `res_base` to a wavelength-dependent `δσ_inst(λ)` vector" (impossible; see §2).
- Drop "DC3 should implement W19 Appendix A's third option" (impossible, same reason).
- Replace `ResolutionPar.match ∈ {none, offset, full}` with the derived-`δ` scheme of §1.7.
- Add Constraints 1–3, the two bounds on `δ` (§1.6), the `dvar_inst` sign convention (§1.4),
  the negative-`dvar_inst` warning, template oversampling / `velscale_ratio`, once-per-run
  preparation, and the two fitting hooks of §1.9.
- Phase 7's "two estimators" restructuring is superseded: with `δ` chosen, `dvar_inst` is known
  exactly and W19 Eq. 7's biased average is not needed on the primary path. See §5.4.

---

## 2. `match='none'` is not representable by DC3's forward model **[resolved]**

`VDF::getbtsp` convolves by a **single scalar** Gaussian; `ppxf_util.losvd_rfft`'s `sigma_diff`
is also scalar. A wavelength-dependent `δσ_inst(λ)` has nowhere to enter the model. Left
unmatched, the single-Gaussian `B` is misspecified wherever `δσ_inst(λ)` varies, and the
misspecification is absorbed into the fitted σ as a feature-weighted average — exactly the
difficulty W19 §7.1.5 describes as "non-trivial because of the unknown relative influence of
each spectral feature".

> *"For item 2 specifically, this means I agree with the provided recommendation."*

**Resolved.** Constant-offset preparation is the default. §1.5's Constraint 3 (computational
cost of a wavelength-dependent kernel inside the fit) is the primary justification, with the
representational argument as a secondary one. A true "leave the template alone" mode is not
worth keeping as a supported path, since it is both slower and misspecified.

---

## 3. The analytic-derivative speedup is overestimated **[edit]**

Plan, Phase 5, attack #1 claims "5–7× reduction". Censoring is applied to the broadened template
**in real space** before correlation (`getbtxc` calls `update()`, which re-runs the full masking,
apodization and FFT correlation; report §8.1–8.2). So `∂X_T/∂σ` cannot stay in Fourier space —
per parameter it needs inverse FFT → mask/apodize → forward FFT → multiply by `conj(T̂)` →
inverse FFT, ≈3 FFTs.

| | forward evals | FFTs |
|---|---|---|
| Central differences (current) | 7 | ≈28 |
| Analytic Jacobian | 1 + 3 × 3 | ≈13 |

→ **~2×, not 5–7×.** Still worth doing and it compounds with FFTW plan reuse, but the figure is
currently setting the order of Phase 5.

Unaffected by §1: resolution matching leaving the cost function does not change this count,
because `varsmooth` was never inside the LM loop to begin with — `match_resolution` is called
from `set_mt_cw` once per tier-2 iteration, not per model evaluation.

### 3.1 The model's velocity shift **[resolved]**

`getbtsp` applies the shift with `shift_lininterp`, real-space **linear** interpolation, so
`∂X_T/∂V` is not smooth and the interpolation is itself a small systematic.

> ▶ *"Yes, replace with the Fourier phase shift. It is a valuable departure from W11."*

**Decided.** Fold the shift into the LOSVD's Fourier transform via the time-shifting property,
C17 Eq. 37: `h(t − t₀) ⟺ H(ω) e^{iωt₀}` — which is `losvd_rfft`'s `vsyst` argument. C17 §4.3
lists this as one of the two advantages of the analytic transform ("simple velocity shifting"),
the other being that the LOSVD can never be undersampled in the frequency domain. It composes
naturally with the template oversampling of §1.6: broaden and shift on the oversampled grid,
bin down afterwards.

This is distinct from galaxy de-redshifting (§5.5): that is a once-per-run integer shift of the
data, this is a continuous shift of the model inside the fit. Both change, for different reasons.

It **is** an algorithmic departure from W11 and belongs on the Phase 8 differences page.

---

## 4. Internal contradictions **[edit unless noted]**

| # | Where | Problem | Status |
|---|---|---|---|
| 4.1 | plan, Phase 4, `dc3/core/window.py` | "Fix … the `-f`/`-v` argument confusion" contradicts the backwards-compatibility section, where that parser ceases to exist. Keep the `vdf.cpp:487` `win.invs`/`win.inve` fix (a real algorithm bug); drop the CLI half. | edit |
| 4.2 | plan, Phase 2, "Templates" | "Mimic its resolution-matching approach specifically" — now *correct* but for the reason in §1, not the one stated. Rewrite around derived-`δ`. | folded into §1.10 |
| 4.3 | plan, Phase 4, Tier 2 | "re-match resolution" inside the tier-2 loop — contradicts §1.5 Constraint 3. Preparation is once per run, so tier 2 must **not** re-match. This is a genuine behaviour change from the C++ (`set_mt_cw` calls `match_resolution` every mask iteration) and belongs on the differences page. | **elevated** |
| 4.4 | plan, Phase 2, template cache | Cache key "library + galaxy + velocity scale" — with §1.8's once-per-run preparation the key is well defined, but must now include `δ` (or the σ floor that determines it) and the oversampling ratio. | edit |
| 4.5 | plan, Phase 3 vs. Test data | Phase 3 rounds `nn` up with `scipy.fft.next_fast_len`; Test data says `_xc.fits` validates `nn = 4508` and `v0 = −24823.75` km/s. Changing `nn` moves the velocity grid, killing direct array comparison. Needs a compatibility flag reproducing the exact C++ sizing. | edit |
| 4.6 | plan, Phase 6 vs. Verification 7 | "bit-identical" vs. "machine precision" for the K = 1 regression. Use "machine precision" in both. | edit |
| 4.7 | plan, Phase 5, JAX extra | See §4.7 below — answered, with a consequence for the ppxf dependency. | **see below** |
| 4.8 | plan, Phase 6 | Linearity in `w` holds only if masking, mean subtraction and apodization are identical across the library, so censoring must be computed once for the library as a whole. ▶ *"Yes, this is worth mentioning, but the same is true of the templates used in `ppxf`."* Record it, noting the parallel — it is a general property of template-mixing fits, not a DC3 quirk. | resolved |
| 4.9 | plan, Phase 6 | With K > 1 there is no separate amplitude `I`; the weights absorb it, so `solve_norm` is moot. State it. | edit |
| 4.10 | plan, Phase 4 | "the `fititer=5` randomized restarts": 5 is the `progfiles` default, production used 2, and `VDF::fit()` allows `totiter = fiter * 5` attempts for up to `fiter` successes. Describe the mechanism, not a number. | edit |
| 4.11 | plan, Phase 4, Tier 3 | `Legendre.fit(..., w=ivar)` — numpy's `w` multiplies residuals, so it must be `sqrt(ivar)`. | edit |
| 4.12 | report, §15 "Validation" | Still says the W11 §5 SparsePak data is unlocated; §16.8 supersedes it. | edit |
| 4.13 | report, between §15 and §16 | Stray doubled `---`. | edit |
| 4.15 | report, §8.3 and §14 | **New finding, not yet in the report.** §8.3 describes `res_base` as a numerical-stability floor but does not record its *consequence*: because `getbtsp` applies `sqrt(σ_fit² − base_sig²)`, `res_base` is a hard floor on the measurable σ, and the fitter sets σ ≡ 0 and re-fits below it. Production values reached **9.50 km/s** (§16.4). This is a real, previously undocumented limitation of the C++ and belongs in the report — extend §8.3 and add an entry to §14. | **add** |
| 4.14 | plan, Context | "built on numpy/scipy/astropy" predates `ppxf` becoming a dependency and `specutils` a boundary one. | edit |

### 4.7 JAX, `varsmooth`, and the ppxf non-vendoring rule **[resolved]**

> *"First, I think the discussion above means that `varsmooth` should not be part of the fit cost
> function, but correct me if I'm wrong."*

**Correct.** Under §1.5 Constraint 3, `varsmooth` is called during template preparation only —
once per execution — and never inside the objective. The JAX conflict for `varsmooth` therefore
disappears entirely: the JAX path does not need it.

> *"Regarding reimplementation of `losvd_rfft`, what is our recourse on the Gauss–Hermite
> function if JAX is used? Would a re-implementation of `losvd_rfft` that is JAX-traceable breach
> ppxf's non-vendoring rule?"*

**No — provided it is written from the published equations rather than from the source.** The
license prohibits redistributing *the code*; it does not and could not restrict implementing
published mathematics. The analytic Fourier transform of a Gauss–Hermite LOSVD is in the
literature: van der Marel & Franx (1993) for the parameterization, Cappellari (2017) §2.3 for
the Fourier-space form ppxf uses.

What would breach it is a line-by-line translation of `ppxf_util.losvd_rfft` into `jax.numpy` —
that is a derivative work of the code, regardless of the change of language.

Practical protocol, worth writing into the plan so it is defensible later:

1. Implement `dc3/core/losvd.py::losvd_ft` from the **papers**, citing the specific equations in
   the docstring.
2. Do not consult `ppxf_util.py` while writing it.
3. Verify **numerically** against `ppxf_util.losvd_rfft` in the test suite. Numerical agreement
   is evidence of correctness, not evidence of copying.

**▶ Resolved: `ppxf` stays a hard dependency.**

> *"A core difference between `ppxf_util.varsmooth` and `mangadap`'s
> `convolution_variable_sigma` is that the former uses the more robust analytic equation for the
> FFT of a Gaussian. This is the preferred approach."*

This settles it on accuracy rather than convenience, and the argument survives `varsmooth` being
demoted to a once-per-run call. `varsmooth` implements **Algorithm 1 of Cappellari (2023),
MNRAS 526, 3273** — a third reference the plan does not yet cite. It stretches the wavelength
coordinate so the variable σ becomes constant, then performs a *single* FFT convolution against
the analytic Gaussian transform:

```python
sig = sig_x/np.gradient(x)
sig = sig.clip(0.1)                      # the floor discussed in §1.7
xs  = np.cumsum(sig_max/sig)             # stretch to constant sigma
...
ft_gau = np.exp(-0.5*w**2)               # analytic FT of the Gaussian
yout   = np.fft.irfft(ft*ft_gau, npad)
```

`mangadap`'s `convolution_variable_sigma` instead builds a banded matrix of sampled Gaussians,
which degrades exactly where C17 says it should — when the kernel is undersampled. Keep it as
the independent cross-check (plan, Phase 2), but `varsmooth` is the production path.

**Consequence for the JAX extra:** the clean-room `losvd_ft` of the protocol above remains
necessary for the JAX path, but it no longer implies dropping the dependency. `dc3` depends on
ppxf for `varsmooth` regardless, so the BSD-3/packaging caveats in the plan's External
dependencies section stand as written.

---

## 5. Design questions raised in the first pass — responses **[all resolved]**

> *"None of the following are hard requirements. We should be open to revisiting them as we
> develop the code."*

### 5.1 Template library

> *"For now, we should leverage the templates in the `mangadap` repo. Notably, the templates used
> in the DiskMass survey are not distributed. Some of the templates are at
> `/Volumes/seshat/data/diskmass/mab_jul05_templates/Mg`; however, I'm unsure of their
> provenance. There is a README that may be useful."*

**Decision: ship/point at the `mangadap` libraries for now.** DMS templates are not distributed
and are not a dependency.

**[verify] — I read the READMEs; provenance is partly documented.**
`mab_jul05_templates/README` and `Mg/README` record:

- Extracted, wavelength-calibrated, averaged template spectra from the **three SparsePak
  commissioning runs of May–June 2001**, Mg region (order 11), reduced by Westfall.
- Naming: originally by spectral type (`[type]_lin.fits`, `[type]_log.fits`); re-issued
  12 Jul 2005 by catalogue ID (`[ID].Mg.lin.fits`, `[ID].Mg.log.fits`).
- The 2005 log spectra were re-processed with `dispcor` to a **common starting wavelength, size
  and dispersion** — the earlier log spectra did not share a dispersion. Use the 2005 set.
- `master.db` / `master_lin.db` / `master_log.db` hold observing and instrument information,
  which is where any resolution vector for these would have to come from.
- The top-level README is an email dated 5 Nov 2002 describing moment-analysis window tests.

The README's own caveat — *"The exact extraction and averaging needs to be detailed by
Westfall"* — is the one that matters for using these as fixtures. Its other caveat, *"Error
spectra are also needed"*, is **not** relevant to the port:

> ▶ **Template spectra are never provided with errors.** A standard assumption (and limitation)
> shared by `ppxf` and DC3 is that the model spectra built during fitting are **noise-free**:
> observational error in the template is ignored. Errors are a property of the galaxy spectra
> only. This is visible in the algorithm itself — the Statler covariance
> `(δX)²_{j,k} = Σ_n T_{n−j} T_{n−k} (δG_n)²` (W11 Eq. 4) carries `δG` but has no `δT` term at
> all. Worth stating explicitly as a documented assumption in the plan, since a reader coming
> from a full forward-modelling background would expect otherwise.

Note these templates are the same instrument and spectral region as the W11 §5 demonstration, so
they are a natural multi-template library for exercising Phase 6 against that dataset.

Still to record: **template-library licensing** (MILES et al. have their own redistribution
terms), which interacts with the BSD-3 goal the same way the ppxf license does.

### 5.2 Default `δ`

Answered in §1.7–1.8: `δ` is derived from a user-set minimum σ, made as large as that floor
allows, and computed **once per execution** so it is common to all spectra. Exposed parameter
ambiguity → resolved in §1.7 and §8.1.

### 5.3 `DC3Fit.fit()` signature

> *"The outermost function should assume multiple spectra are provided, but it should be able to
> operate on a single spectrum. The innermost function — i.e., the function that is called to
> execute the fit after all the prep work has been completed — should operate on a single
> spectrum only. The innermost function, to my mind, represents the natural place for
> multiprocessing hooks."*

**Decision: a two-level API.**

| Level | Input | Role |
|---|---|---|
| Outer | many spectra (1 is a valid case) | preparation, template library handling, iteration, results assembly |
| Inner | exactly one spectrum, fully prepared | the fit itself — **the multiprocessing boundary** |

This settles Phase 5's parallelism design: the pool maps over the inner function, and everything
expensive and shared (prepared templates, their FFTs, the correlator, FFTW plans) is prepared
once outside it and passed in. It also fixes what `ConvolveFFTW.__reduce__` has to survive.

### 5.4 Phase 7 ordering and W11 Table 1 precision

> *"As already stated in the plan doc, let's plan to pull phase 7 forward. The current doc notes
> that some aspects of phase 7 should be pulled forward to phase 4, but that others may remain in
> phase 7. Please reassess. And, yes, please check W11 again for relevant measurement precision"*

**[verify] W11 Table 1, "Example Stellar Kinematic Fits":**

| Fiber | S/N (px⁻¹) | V_f,DC3 | V_f,pPXF | ΔV/δ | σ_f,DC3 | σ_f,pPXF | Δσ/δ |
|---|---|---|---|---|---|---|---|
| 52 | 27.8 | 1125.3 ± 1.5 | 1125.2 ± 1.4 | 0.02 | 68.3 ± 1.6 | 63.9 ± 1.6 | 1.95 |
| 55 | 1.7 | 1076.4 ± 3.9 | 1073.6 ± 3.9 | 0.52 | 11.2 ± 5.7 | 11.9 ± 5.9 | −0.08 |

Three conclusions:

1. **Table 1 reports `σ_obs`, not `σ_LOS`.** W11 §5 says the fits "converge to measurements of
   `V_obs` and `σ_obs`", and the comparison is against ppxf output, which is likewise
   uncorrected. **So reproducing Table 1 does *not* require Phase 7.** The ordering worry I
   raised is resolved — this is a Phase 4 test.
2. **"Tight tolerances" was overstated.** Values are quoted to 0.1 km/s but carry errors of
   1.5–5.7 km/s. A defensible gate is ~1 km/s on fiber 52's `V` and `σ` — tight in absolute
   terms, but nothing like machine precision. Fiber 55 (S/N 1.7, `σ = 11.2 ± 5.7`) is
   essentially unconstrained and cannot gate anything.
3. **Figure 9's population statistics are the better gate**, and they are stated numerically in
   the text: *all* velocity measurements within their errors, and for σ, **82% and 94% of all
   spectra within one and two times the measurement errors** respectively. That is a
   whole-field statistic over every fiber rather than two hand-picked numbers.

Also noted in passing (W11 §4): the block-replication floor is quoted as
`σ_f ≲ (0.85 ΔV_p = 6.4) km/s`. This is a *separate* floor from §1's `dvar_inst` floor —
pixelization rather than resolution — and fiber 55's `σ = 11.2` km/s sits less than a factor of
two above it. Both floors need to be reported per spectrum.

**Phase 7 reassessment.** With `δ` chosen rather than estimated (§1.7), the split changes:

| Was | Now |
|---|---|
| Phase 7 = the whole instrumental-dispersion treatment, applied post hoc | **Phase 2/4:** `dvar_inst` is fixed at template-preparation time, known exactly, reported per spectrum, with the negative-value warning. No estimator needed. |
| Phase 7 route 1 = W19 Eq. 7 flat average, biased low | **Not needed on the primary path.** Retain only for the unmatched case, if that is kept at all (§2 says it should not be). |
| Phase 7 route 2 = W11 Appendix A fitted estimator | **Stays in Phase 7.** Still the better-posed route, still the one thing DC3 can do that the DAP cannot, and still needed to validate that the chosen `δ` behaves as assumed. |

So Phase 7 shrinks to the *fitted* estimator plus its error propagation, and the bookkeeping
moves into Phases 2 and 4. That is a larger simplification than "pull Phase 7 forward" suggests.

### 5.5 De-redshifting **[resolved]**

> *"Given the logarithmic sampling of the spectra in wavelength, we can recast this as an
> approximate de-redshifting (as you say, we'll be fitting the velocity offset anyway) using a
> simple pixel shift and truncation, not a redistribution of flux within the spectrum. This shift
> and/or truncation can then be analytically propagated to the measured velocity offset."*

**Decision: approximate de-redshifting by integer pixel shift and truncation**, with the applied
shift propagated analytically into the reported velocity. No interpolation, no resampling, no
induced covariance — consistent with §1.5 Constraint 1.

The plan's current framing ("one resampling of the galaxy … *introduces* inter-pixel
covariance") is wrong and must be replaced. Science question 4 (spectral covariance) can drop
its "revisit alongside the de-redshifting decision" caveat: this route introduces none.

### 5.6 Smaller points

- **Template resampling.** ▶ *"The template(s) should always be resampled to the same wavelength
  grid as the galaxy spectra, up to an integer ratio in the number of samples. The approach
  should mimic the `velscale_ratio` parameter in `ppxf`."* **Decided.** Ties directly to §1.6:
  the oversampling ratio is what buys headroom for a larger `δ`. Default and auto-selection
  criterion at §1.6 and §7.6.
- **No-errors path.** ▶ Keep it. This concerns the **galaxy** spectra only (templates are always
  noise-free — see §5.1). Future development: a post-processing script estimating galaxy errors
  from a moving-window standard deviation of the model residuals, with the workflow *fit without
  errors → estimate errors → re-fit*. Record as a future path, not this port.
- **Size estimate.** ▶ *"I would suggest dropping the size estimate."* **Remove** the
  "6,000–8,000 lines of Python" line from the plan's Context section.
- **Test-data location.** ▶ *"Do not include test data in the `tests/` directory."* Replacement
  design resolved in §7.7 below: follow PypeIt's `data/tests` + cache + `MANIFEST.in` pattern.

---

## 6. Order of work

No blockers remain. The two documents can now be updated in one pass.

1. **Rewrite plan Phase 2** around §1 — the two-step preparation pipeline, the four parameters,
   the two-sided `epsilon_sigma`, `dvar_inst` and its sign, the three hard constraints, template
   oversampling, once-per-run preparation, and the empirical pixelization characterization of
   §8.2(b). This is the largest single edit and touches Phases 4 and 7 as well.
2. **Rewrite plan Phase 7** to its reduced scope (§5.4): the W11 Appendix A fitted estimator plus
   error propagation, with the bookkeeping moved into Phases 2 and 4.
3. **Rewrite the Test data section** around the PypeIt `data/tests` + cache + `MANIFEST.in`
   pattern (§7.7), and the Verification section around the W11 Table 1 findings (§5.4).
4. **Apply the mechanical edits** of §3 and §4. Three of these land in the **report** rather
   than the plan: §4.12 (stale "SparsePak unlocated" text in §15), §4.13 (stray `---`), and
   §4.15 (the `res_base` σ-floor finding, which is new content for §8.3 and §14).
5. **Add the new interfaces**: the two fitting hooks (§1.9) and the constraints table (§8.4);
   the two-level fit API (§5.3); integer-shift de-redshifting (§5.5); the Fourier phase shift
   (§3.1); the no-errors path and its future residual-based companion script (§5.6).
6. **Record the new references** — Cappellari (2017) and Cappellari (2023) — and the new
   documented assumptions: pre-pixelized `sres` vectors (§8.2a) and noise-free templates (§5.1).

Items destined for the Phase 8 "Differences from Westfall et al. (2011)" page, collected in one
place since they are scattered through this document: no tier-2 resolution re-matching (§4.3);
Fourier phase shift in place of linear interpolation (§3.1); no automatic σ ≡ 0 refit (§1.9);
optional B-spline continuum; multi-template mixing; the revisited fit-window width.

---

## 7. Follow-up questions — responses **[all resolved]**

### 7.1 Which "minimum sigma" is exposed → **(a), the convolution kernel**

Answered in full at §1.7, which now carries the two-step preparation pipeline, the four
parameters, the `varsmooth` clip that sets `epsilon_sigma`'s default, and the two warnings.
The sign question it left open is resolved in §8.1.

New reference added to the project: **Cappellari (2017), MNRAS 466, 798**
(`/Users/westfallold/Work/literature/Cappellari_2017_MNRAS_466_798.pdf`), where the analytic
Fourier transform of the Gaussian and Gauss–Hermite kernels is introduced. §4.3 and Eqs. 33–38
are the relevant part; Eq. 37 is the velocity-shift property used in §3.1. A third reference,
**Cappellari (2023), MNRAS 526, 3273**, is the algorithm behind `varsmooth` (§4.7).

### 7.2 `ppxf` stays a hard dependency

▶ Yes. Resolved at §4.7 — the deciding factor is that `varsmooth` uses the analytic Gaussian
FFT while `mangadap`'s `convolution_variable_sigma` samples the kernel.

### 7.3 `dvar_inst` name and sign convention

▶ *"Yes, let's stick with `dvar_inst`, and the sign convention matches my intent."* §1.4 stands
as written: `dvar_inst ≡ σ_G² − σ_T'²`, signed, km²/s², with `σ_*² = σ_obs² − dvar_inst`.

### 7.4 Generality of the "fix kinematics" and "fit a subset" hooks

▶ *"Your read is consistent with my intent, however, we will need to be clever about how these
constraints are passed to the program given the level of granularity needed when large batches
of spectra are passed to the outer loop."*

**Both hooks are general** — fix `V` and/or `σ` to arbitrary per-spectrum values, and supply a
selection mask — with the σ ≡ 0 refit as one use among several.

The open part is *how they are expressed*, and it is a real design problem: per-spectrum,
per-parameter constraints do not fit in a scalar ParSet, and the natural home is a **table**
keyed by spectrum ID. That table then has to reach the inner fit function (§5.3) through the
same channel as everything else. → §8.4 proposes a shape.

### 7.5 Fourier phase shift

▶ Yes. Resolved at §3.1, with C17 Eq. 37 as the citation.

### 7.6 Default oversampling ratio

▶ `velscale_ratio = 1` by default, with optional automatic determination; criterion is that the
**LSF FWHM be sampled by at least 2 pixels**. Recorded at §1.6.

### 7.7 Test-data location → **PypeIt's `data/tests` + cache + `MANIFEST.in` pattern**

> *"We should follow the `PypeIt` pattern, but not via a separate development suite repo. PypeIt
> uses its cache system to access test data that is located in `pypeit/data/tests`. The test data
> are not gitignored (the cache accesses them via GitHub if they are not local), but they are not
> included in the package release (i.e., the `pypeit/data/tests` directory is excluded by the
> `MANIFEST.in` file)."*

**[verify] — confirmed against the PypeIt checkout.** `MANIFEST.in` reads:

```
# Remove large data sets
# NOTE: If you exclude more data from the pypeit/data directory, make sure to
# update the defined_paths dictionary in pypeit.pypeitdata.PypeItDataPaths!
recursive-exclude pypeit/data/tests *.gz *.fits *.npz
```

Three details worth carrying over exactly:

1. **The exclusion is by file type, not by directory.** `README` and other small files under
   `pypeit/data/tests` *do* ship; only the bulk formats are stripped. So a fixture README and
   the provenance manifest can travel with the wheel while the data does not.
2. **`MANIFEST.in` and the data-path registry must be kept in sync** — PypeIt says so in a
   comment, and registers `tests` in `PypeItDataPaths` (`pypeit/pkg/pypeitdata.py:344`). The
   DC3 equivalent is `dc3/pkg/dc3data.py`. Worth making a test rather than a comment.
3. There is a `pypeit/data/s3_url.txt`, i.e. an S3 fallback alongside GitHub.

**Decision for DC3:**

| Path | Contents | Git | Wheel |
|---|---|---|---|
| `dc3/data/tests/` | all fixtures | **committed** | **excluded** via `MANIFEST.in` by file type |
| `dc3/data/tests/README.rst`, `MANIFEST.toml` | provenance and caveats | committed | **shipped** (not a stripped type) |
| `dc3/tests/` | test code only, no data | committed | shipped |

`dc3/pkg/cache.py` fetches anything absent from the GitHub copy, so a clean checkout without the
RAID still runs the full suite. This also supersedes the plan's `requires_testdata` skip marker
for most cases — the data is fetchable rather than optional — though the marker is still wanted
for offline runs. **`MANIFEST.toml` stays committed**, answering the second half of the old
question.

---

## 8. Second round of follow-up questions — responses **[all resolved]**

### 8.1 `dvar_inst` may be positive — two-sided `epsilon_sigma` **[resolved]**

▶ *"Yes, your proposed resolution was my intent."* Folded into §1.7 above: `epsilon_sigma` is a
two-sided target on the minimum kernel σ, `δ² = min_λ res_match − epsilon_sigma²` is signed, and
the scheme is a superset of `GaussianKernelDifference`'s option 2 rather than a contradiction of
it.

### 8.2 Pixelization and the effect of resampling

#### (a) Resolution vectors are **pre-pixelized** — an input contract, not a computation

▶ *"Yes, that is the intent here. However, this is mostly a requirement for appropriate use of
the code and something we need to ensure is properly documented. That is, the code will assume
that the instrumental dispersion provided for both the template and galaxy data are determined
for the pre-pixelized data. It will be up to the users to ensure they're providing the correct
resolution measurement."*

**Decision: `dc3` assumes both `sres` vectors describe the LSF *before* integration over the
spectral channel, and does not attempt to detect or correct a violation.** This matches W19
§7.1.5's use of MaNGA's `PREDISP` rather than `DISP`, and it matches what ppxf assumes.

Consequences to write into the plan:

- **This is a documented input contract.** It belongs in the `sres` datamodel description, in
  the parameter documentation, and in the tutorial — not buried in an algorithm page. Getting it
  wrong biases every `σ_*` in the output, silently.
- For MaNGA-style inputs, name the correct extension explicitly (`PREDISP`, not `DISP`), since
  that is the most likely concrete mistake.
- Consider a **soft diagnostic** rather than a check: if the supplied `sres` implies
  `σ_inst ≲ Δ/√12`, the vector is smaller than pixelization alone would produce and is very
  likely post-pixelized or simply wrong. Warn, do not fail.

#### (b) Where `Δ²/12` comes from, and why it should *not* be applied analytically

▶ *"First, where does the Δ²/12 metric come from?"*

Integrating a spectrum over a pixel of width `Δ` is convolution with a **top-hat** of width `Δ`.
Convolution adds variances, and the variance of a uniform distribution on `[−Δ/2, Δ/2]` is

```
∫_{−Δ/2}^{+Δ/2} x² (1/Δ) dx  =  (1/Δ)·[x³/3]  =  Δ²/12
```

so `σ²_post-pixel ≈ σ²_LSF + Δ²/12`. That is the whole content of the `DISP` vs. `PREDISP`
distinction. It is approximate twice over: the true pixel response is not a perfect top-hat, and
the sum-of-variances result is exact for second moments but not for the fitted width of a
non-Gaussian profile.

▶ *"The correct path on this is currently unclear to me … I expect the resolution vectors to be
provided pre-pixelized, and it's unclear to me how the resampling propagates to this
pre-pixelized measurement. Nominally it should have no effect, but I expect there are practical
limitations to this. I expect we will need to characterize the effect empirically."*

**You are right on both counts, and the tension has a specific cause.** Nominally, resampling
cannot change a pre-pixelized LSF: the intrinsic LSF is a property of the optics, not of the
sampling, so the *vector* is unchanged. But the **array** being resampled is not the intrinsic
spectrum — it is the intrinsic spectrum already integrated over its native pixels of width
`Δ_tpl`. Rebinning that onto a new grid of width `Δ_new` gives, schematically,

```
true ⊗ LSF ⊗ tophat(Δ_tpl) ⊗ [resampling kernel ≈ tophat(Δ_new)]
```

whereas a spectrum *natively* sampled at `Δ_new` would carry only `tophat(Δ_new)`. The excess is
of order `Δ²_tpl/12` and **cannot be removed without deconvolution**, which Constraint 2
forbids. It is irreducible: oversampling (`velscale_ratio > 1`) shrinks `Δ_new` but leaves
`Δ_tpl` untouched, because that term is a property of the data as delivered.

So the pre-pixelized convention is a modelling assumption that discards *both* spectra's native
pixel integration on the grounds that they approximately cancel. They cancel well when the two
native samplings are similar and poorly when they are not — and nothing in the convention
signals which case you are in. ppxf inherits the same gap; C17 treats pixel integration only for
gas emission lines (Eq. 28), never for stellar templates.

**Decision: do not add an analytic `Δ²/12` term.** Under the pre-pixelized contract it would
double-count, and it assumes a top-hat response that the resampler does not actually apply.

**Instead, characterize the preparation pipeline empirically**, as you expect. A concrete and
cheap experiment:

1. Build a synthetic spectrum of narrow, isolated Gaussian lines of **known** width
   `σ_in` (≈1 native pixel — resolved enough to avoid interpolation artefacts, narrow enough to
   be sensitive), at a grid of wavelengths spanning the template range.
2. Push it through the **exact** preparation pipeline — Step 1 resolution matching at the chosen
   `epsilon_sigma`, then Step 2 resampling at the chosen `velscale_ratio`.
3. Fit the output line widths and subtract `σ_in` in quadrature.

The result is the *measured* effective `σ_T'(λ)` of the prepared templates, including every
resampling, interpolation and clipping effect, with no analytic model of the pixel response at
all. Compare it against the nominal vector; the difference is the correction that belongs in
`dvar_inst`.

This is an impulse-response measurement of the preparation pipeline. It runs **once per
configuration**, not per spectrum or per fit, so it is affordable; the measured `σ_T'(λ)` and
its departure from nominal should be recorded in the output header. It also doubles as a
regression test on the whole of Step 1 + Step 2.

**Plan action:** add this as an explicit Phase 2 characterization task with a deliverable —
a table or figure of measured-minus-nominal `σ_T'` as a function of `velscale_ratio` and of the
template/galaxy sampling ratio. That answers the question with data rather than with an
assumption, and it is the part of this that `mangadap` never did.

### 8.3 Hard-code the `varsmooth` floor, with an asserting test **[resolved]**

▶ *"Let's hard-code the current 0.1 pixel limit but include a test that asserts this is correct.
That should be sufficient to catch changes to `ppxf` by the developers, but does not require a
redetermination of the limit on every import."*

Recorded at §1.7. A module constant plus a behavioural unit test — convolve with a requested σ
below the floor and assert the realised width — so no import-time cost and a loud failure if
upstream moves the literal.

### 8.4 Per-spectrum constraints: a fixed-width ASCII table **[resolved]**

▶ *"I like your suggestion of a constraints table. The table should be ascii so that it is
directly human readable and writable. I prefer fixed-width columnated format to ecsv, but ecsv
is preferred over FITS."*

**Decision: fixed-width columnated ASCII**, via `astropy.io.ascii` (`format='fixed_width'` or
`fixed_width_two_line`), with ECSV accepted as a second format and FITS as a last resort.

| Column | Type | Meaning |
|---|---|---|
| `ID` | str/int | spectrum identifier, matched against the input spectra |
| `FIT` | bool | include this spectrum in the fit at all |
| `V` | float | velocity: initial guess, or the fixed value if `V_FIX` |
| `V_FIX` | bool | hold `V` at the supplied value |
| `SIG` | float | dispersion: initial guess, or the fixed value if `SIG_FIX` |
| `SIG_FIX` | bool | hold `σ` at the supplied value |

Missing rows mean "fit normally". Supplying a value without its `_FIX` flag makes it an initial
guess, which covers the common case of per-spectrum starting velocities without needing a second
mechanism. The outer function (§5.3) resolves the table into per-spectrum arguments so that only
a single row crosses the multiprocessing boundary.

> **Note — this is not a reversal of the FITS-only output decision.** That decision (plan,
> Phase 4) governs *products the code writes*, where precision, units and provenance matter and
> the C++ `.db` format failed on all three. This is an *input* a human writes by hand, where
> readability and writability dominate. Worth stating explicitly in the plan so the two do not
> look inconsistent.

The one cost of fixed-width over ECSV is that dtypes and units are inferred rather than
declared; mitigate by validating the parsed table against a declared schema on read, and by
shipping a documented example file that users can copy.

---

## Change log

- **2026-09-15** — Created, from a full re-read of the report and the plan. Principal finding:
  DC3's `res_base` has the opposite sign to a W19-style resolution pedestal — it makes the
  template broader than the galaxy, cancels exactly inside `getbtsp`, and acts as a floor on the
  measurable σ (up to 9.5 km/s in production) rather than keeping σ away from zero. Second
  finding: a wavelength-dependent `δσ_inst(λ)` cannot enter DC3's forward model at all, since
  `getbtsp` and `ppxf_util.losvd_rfft` both take a scalar σ. Proposed a constant-offset mode
  resolving both. Corrected the Phase 5 analytic-derivative estimate from 5–7× to ~2×, and
  catalogued fourteen internal contradictions and six open design questions.
- **2026-09-16** — Incorporated the responses from `notes`. **Correction to §1:** the negative
  offset is a legitimate regime, not a defect — DMS used same-instrument templates and had to
  adopt a negative `δσ_inst²`, with the resulting σ floor understood as an observational
  limitation. The port must support both signs and warn on negative; nomenclature moves to a
  signed variance (§1.4). Recorded three hard constraints — the galaxy's flux distribution is
  never redistributed, no deconvolution, and resolution matching is preparation-only because a
  wavelength-dependent kernel is too expensive inside the fit (§1.5) — and the resampling/Nyquist
  coupling that bounds `δ` from a second direction and motivates template oversampling (§1.6).
  Recorded the derived-`δ` design with the `Sasuke.etpl_sinst_min` precedent (§1.7),
  once-per-run template preparation (§1.8), and the fix-kinematics / fit-subset hooks replacing
  the C++'s automatic σ ≡ 0 refit (§1.9). Answered the JAX/Gauss–Hermite licensing question:
  clean-room implementation from van der Marel & Franx (1993) and Cappellari (2017) is fine,
  line-by-line translation is not, with a three-step protocol (§4.7) — and noted this weakens
  the case for ppxf as a hard dependency. **[verify]** Read the `mab_jul05_templates` READMEs:
  SparsePak May–June 2001 commissioning runs, Mg order 11, re-issued Jul 2005 on a common
  dispersion, with `master.db` metadata (§5.1). Recorded that **template spectra never carry
  errors** — both `ppxf` and DC3 assume noise-free model spectra, and W11 Eq. 4 has no `δT` term
  — so the READMEs' missing error spectra are irrelevant, and the "no-errors path" of §5.6
  concerns the galaxy spectra alone. **[verify]** Read W11 Table 1: it reports `σ_obs`, *not*
  the corrected `σ_LOS`, so reproducing it does **not** require Phase 7 — the ordering concern is
  resolved; "tight tolerances" was overstated (errors of 1.5–5.7 km/s on values quoted to 0.1);
  and Figure 9's population statistics (all `V` within errors; 82%/94% of σ within 1×/2×) are the
  better gate (§5.4). Reassessed the Phase 7 split: with `δ` chosen rather than estimated, the
  bookkeeping moves to Phases 2/4 and Phase 7 reduces to the W11 Appendix A fitted estimator.
  Recorded the two-level fit API with the inner single-spectrum function as the multiprocessing
  boundary (§5.3), integer-shift de-redshifting (§5.5), `velscale_ratio`-style template
  resampling, retention of the no-errors path, removal of the size estimate, and the removal of
  `dc3/tests/data/` (§5.6). Added seven follow-up questions (§7), of which §7.1 and §7.7 block
  the Phase 2 and Test data rewrites.
- **2026-09-16** — Incorporated the responses to the §7 follow-up questions; §7 now records the
  answers and a new §8 holds what remains open. **Template preparation is specified** (§1.7) as
  a two-step pipeline — resolution matching to a *fiducial* galaxy resolution, then resampling
  at an integer `velscale_ratio` — with four parameters (`velscale_ratio`, `epsilon_sigma`,
  `sigma_floor`, `mask_unmatched_sres`), W19 Appendix A's option-2 pedestal as the default, and
  two warnings. Recorded that both steps make velocity-dispersion corrections unavoidable:
  Step 1 matches only a fiducial resolution, and **Step 2 alters the effective template
  resolution, which `mangadap` has never accounted for** — a new requirement, with §8.2 asking
  how it enters `dvar_inst`. **[verify]** Found that `ppxf_util.varsmooth` clips the kernel σ to
  ≥ 0.1 pixels internally (`sig.clip(0.1)`), so `epsilon_sigma` must default to that value or
  `dvar_inst` would be biased by the difference; §8.3 asks whether to hard-code or test for it.
  `velscale_ratio` defaults to 1 with optional automatic determination from an LSF-FWHM ≥ 2 px
  criterion — numerically the same threshold as the C++'s `minsig = 0.85 px` (§1.6). Confirmed
  `dvar_inst` and its sign (§7.3), the Fourier phase shift with C17 Eq. 37 as the citation
  (§3.1), and both fitting hooks as general (§7.4). **`ppxf` stays a hard dependency** (§4.7):
  `varsmooth` implements Cappellari (2023) Algorithm 1 using the analytic Gaussian FFT, where
  `mangadap`'s `convolution_variable_sigma` samples the kernel — the latter stays as the
  cross-check. Added Cappellari (2017) and Cappellari (2023) as project references.
  **[verify]** Confirmed PypeIt's test-data pattern against the checkout (§7.7): fixtures live
  in `data/tests`, are committed, and are excluded from the wheel by `MANIFEST.in` **by file
  type** — so a README and the provenance manifest still ship — with `MANIFEST.in` and the
  data-path registry required to stay in sync. DC3 adopts the same layout, which restores
  data-driven CI on a clean checkout via the cache. Added §8: the sign question for `dvar_inst`
  (§8.1) and Step-2 resampling (§8.2) block the Phase 2 rewrite; the `varsmooth` floor (§8.3)
  and the per-spectrum constraints table (§8.4) do not.
- **2026-09-16** — Incorporated the responses to §8. **All questions are now resolved and the
  document is ready to be applied to the plan and report.** `epsilon_sigma` is confirmed as a
  **two-sided target** on the minimum kernel σ, so `dvar_inst` can be positive and the scheme is
  a superset of `GaussianKernelDifference`'s option 2 rather than a contradiction of it (§1.7,
  §8.1). Both `sres` vectors are assumed **pre-pixelized**, as W19 §7.1.5 and ppxf assume —
  recorded as a documented *input contract* rather than something the code detects, with a
  suggested soft diagnostic for the likely failure mode (§8.2a). Answered where `Δ²/12` comes
  from — the variance of a top-hat of width `Δ`, which is the whole content of the
  `DISP`/`PREDISP` distinction — and concluded it should **not** be applied analytically, since
  under the pre-pixelized contract it would double-count and it assumes a response the resampler
  does not apply. Identified why the tension exists: the array being resampled already carries
  its own native `tophat(Δ_tpl)`, an excess that cannot be removed without deconvolution and
  that oversampling does not touch, so the pre-pixelized convention silently assumes the two
  native samplings cancel. Proposed the empirical characterization instead — push known-width
  Gaussians through the exact Step 1 + Step 2 pipeline and fit the output widths, giving a
  *measured* `σ_T'(λ)` including every interpolation and clipping effect — as a once-per-
  configuration Phase 2 deliverable that doubles as a regression test (§8.2b). The `varsmooth`
  floor is hard-coded with a behavioural asserting test rather than introspected (§8.3). The
  per-spectrum constraints table is **fixed-width columnated ASCII** via `astropy.io.ascii`,
  ECSV second, FITS last, with a note that this does not contradict the FITS-only decision for
  *outputs* (§8.4). Rewrote §6 as an ordered work plan and collected the Phase 8 "differences
  from W11" items scattered through the document into one list.
- **2026-09-16** — Consistency pass before applying the revisions. Cleared five cross-references
  that still pointed at questions since resolved. Closed two gaps that would otherwise have been
  decided silently during implementation: **when resolution matching is *not* performed** —
  "off" means no `sres` vector is available, not "available but declined", since an unmatched
  path cannot be represented in the forward model (§1.7); and a **third report edit, §4.15** —
  the report documents `res_base` as a numerical-stability device but never records its
  consequence, that it is a hard floor on the measurable σ reaching 9.50 km/s in production.
  That is a genuine, previously undocumented limitation of the C++ and belongs in report §8.3
  and §14.
- **2026-09-16** — ✅ **Applied.** Every decision in this document is now written into the port
  plan and the report; see their change logs for the per-section detail. This document is
  retained as the record of *why*, not *what*. Three edits landed in the report (§8.3's
  `res_base` σ-floor finding and its §14 entry, the stale SparsePak text in §15, and a stray
  horizontal rule); the rest reshaped plan Phases 2–8, the science questions, Test data,
  Verification and Critical files.
