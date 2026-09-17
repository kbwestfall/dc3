# DC3 — Phased Python Port and Refactor Plan

**Companion document:** [`dc3-original-implementation.md`](dc3-original-implementation.md) — a
full survey of the existing C++ code base. Section references below of the form "report §N"
point there.

**Paper references:**

| Tag | Paper | Role here |
|---|---|---|
| **W11** | Westfall, Bershady & Verheijen 2011, ApJS 193, 21 | describes the DC3 algorithm itself; the specification the port reproduces |
| **W19** | Westfall et al. 2019, AJ 158, 231 (`mangadap` v2.2.1) | lessons from driving `ppxf` over ~4 million spectra; source of the spectral-resolution treatment in Phase 2 (§7.1.5, §7.4.3, Appendices A–B) |
| **C17** | Cappellari 2017, MNRAS 466, 798 | the analytic Fourier transform of the Gaussian and Gauss–Hermite LOSVD (§4.3, Eqs. 33–38), and the velocity-shift property (Eq. 37) used in Phase 3 |
| **C23** | Cappellari 2023, MNRAS 526, 3273 | §3.1 and Algorithm 1, the variable-σ convolution behind `ppxf_util.varsmooth`, including its acknowledged interpolation limitation; §2 for the velocity/redshift relation |

Two further companion documents: [`plan-reassessment.md`](plan-reassessment.md) records the
review that produced the current form of Phases 2, 4 and 7, with the reasoning behind each
decision; [`dc3-python-implementation.md`](dc3-python-implementation.md) records the execution of
this plan and any deviations from it.

## Context

`DC3` ("Detector-Censored Cross-Correlation") measures stellar kinematics (V, σ) from
galaxy-continuum spectra via cross-correlation, as described in **Westfall, Bershady &
Verheijen (2011, ApJS 193, 21)** — Paper III of the DiskMass Survey. The method
cross-correlates a galaxy spectrum `G` with a stellar template `T` and fits the resulting
CC function `X = G∘T` with the model `X_T = (T⊗B)'∘T`, where `B` is a parameterized
broadening function. Its novelty (paper §2.2) is that the broadened template is
**detector-censored in exactly the same way as the galaxy spectrum** before correlation —
i.e. it deliberately does *not* use the commutation `X_T = A_B`, because that introduces a
~10% systematic error in σ on real, truncated, masked data.

The C++ implementation has not been touched since 2016 and is effectively dead:

- **Only 11 of 19 executables compile.** The main interactive `DC3` binary and 7 others are
  stale against a 2012–2014 rewrite of the `Spectrum`/`Correlation`/`VDF` classes; the
  makefile says so explicitly, and the ten `obj/DC3_*.o` files their `lnk` files require do
  not exist. `DC3_express` is the only working end-to-end driver.
- Prebuilt binaries in `scripts/bin` are x86_64 Mach-O linked against `/usr/local/lib`
  dylibs that no longer exist on this machine — they do not run.
- It depends on **Numerical Recipes** (2nd and 3rd ed.), which blocked open-sourcing, and on
  custom `Vector`/`Matrix` templates (`headers/vec.h`) with ~10,000 occurrences across the
  code that would touch.
- It links six C libraries (cfitsio, fftw3, wcslib, gsl, pgplot, hdf5 — the last unused).

**Goal:** a fully open-source, BSD-3-Clause Python package built on numpy/scipy/astropy/`ppxf`,
with `specutils` at the I/O boundary, structured after PypeIt, that (a) reproduces the published
algorithm, (b) is materially faster, and (c) adds optimal mixing of a *template library* rather
than a single template.

Roughly 20,000 of the ~72,000 lines of C++ are pure infrastructure that numpy/scipy/astropy
provide for free (§10.3, the `vec.h` type system, `myfuncs`, `nr_2.11`, the PGPLOT UI). The
remainder is what has to be understood before it can be rewritten.

### Decisions already made

| Question | Decision |
|---|---|
| Port scope | CC kinematics + instrumental-dispersion corrections. Report covers all 19 executables. Stacking is the first planned extension (not this port). Asymmetry metrics are computed and reported; deeper integration into the fit is deferred pending deliberation. |
| **Backwards compatibility** | **Not a design driver** — neither for file formats nor for the CLI. The code was never widely used, so nothing external depends on its interfaces. Match the old behaviour only where it is convenient or where it encodes a scientific decision. |
| **Spectral resolution** | Templates are prepared *once per run* to a fiducial galaxy resolution, offset by a constant `dvar_inst` that may be of **either sign**. Matching is never part of the cost function. See Phase 2. |
| **Galaxy data** | Never resampled, and its flux distribution never redistributed. De-redshifting is an integer pixel shift only. |
| Multi-template | Fixed-correlator + NNLS default, with symmetric composite as an optional outer tier. |
| Optimizer | SciPy core (least-squares and differential evolution); JAX optional extra; MCMC as a user-selectable alternative. |
| Configuration | TOML via `tomllib`, with a strict 1:1 mapping between file keys and in-code parameters. |
| Data products | FITS throughout. No ASCII `.db` tables. |
| `ppxf` | A dependency, not a source to copy from — see "External dependencies" below. |
| Interface | CLI + static QA plots, in user-selected tiers. No interactive GUI. |

### Backwards compatibility is explicitly out

This is worth stating up front because it removes constraints that would otherwise shape
several phases. The C++ code's inputs (40-line positional parameter files, `.db` ASCII
tables, keystroke scripts piped into an interactive binary), its outputs (six per-spectrum
FITS files plus two ASCII tables), and its CLI (single-letter tags parsed by a hand-rolled
`readcommandline`) are all free to be redesigned. Concretely:

- The output datamodel is redesigned from scratch (Phase 4), not reproduced.
- The CLI is designed as an `argparse` interface, not as a transcription of the old tags —
  which also means the `-SG`/`-ST` off-by-one and the `-f`/`-v` confusion (report §14.1) are
  *not bugs to fix* so much as a parser that ceases to exist.
- The one thing that *is* preserved deliberately is the **algorithm**, including the
  hardwired constants whose values are empirically justified in `doc/develop.txt`.

The archived C++ inputs still need readers, but only as *ingest* paths for the test fixtures
(Test data section) — not as the package's native formats.

### External dependencies

| Package | Role | Notes |
|---|---|---|
| numpy, scipy, matplotlib, astropy | core | hard dependencies |
| `ppxf` | `ppxf_util.varsmooth` (variable-σ resolution matching, template preparation only), `ppxf_util.losvd_rfft` (analytic FFT of the Gaussian/Gauss–Hermite LOSVD), `log_rebin`, `air_to_vac`/`vac_to_air` | **hard dependency, import-only** |
| `specutils` | file I/O / format sniffing at the boundary only | see Phase 2 |
| `pyfftw` | optional `[fftw]` extra | Phase 5 |
| `jax` | optional `[jax]` extra | Phase 5 |
| MCMC sampler (`emcee`, `numpyro`, or `dynesty`) | optional `[mcmc]` extra | Phase 4; choice deferred |

> ⚠️ **`ppxf` is not BSD.** Its license is "Other/Proprietary": use and personal modification
> are permitted, but **"You may not redistribute the code."** This does not block using it —
> `dc3` would declare it in `pyproject.toml` and `pip` installs it from PyPI — but it does
> impose two rules that must hold for the life of the project:
>
> 1. **Never vendor, copy, or adapt ppxf source into `dc3`.** Import it. A copied twenty-line
>    helper would make the `dc3` wheel non-redistributable and silently void the BSD-3 license.
> 2. **Expect packaging friction downstream.** conda-forge and distribution packagers cannot
>    redistribute ppxf, so anything that wants to ship `dc3` as a system package will need the
>    ppxf-dependent paths to be optional. Keep the `ppxf` imports localized to
>    `dc3/core/resolution.py` and `dc3/core/losvd.py` so an internal fallback can be swapped in
>    later if that ever becomes necessary — but do not write the fallback pre-emptively.
>
> **Test against ppxf ≥ 9.5.0** (the current release; 9.4.2 is what is installed here). Pin a
> floor in `pyproject.toml` and add ppxf to the CI matrix so API drift in `varsmooth` /
> `losvd_rfft` surfaces immediately.

Depending on ppxf removes the need to port most of `libraries/fitfunc.cpp` and the
`gsl_integration_qagi` machinery in `cnvlv.cpp` — see Phases 2 and 3.

**Why `ppxf` rather than an internal implementation**, given that `varsmooth` is now called only
once per run (Phase 2) and a clean-room `losvd_ft` is needed for the JAX path regardless
(Phase 5): **accuracy, not convenience.** `varsmooth` implements C23 Algorithm 1 — it stretches
the wavelength coordinate so the variable σ becomes constant, then performs a *single* FFT
convolution against the **analytic** Gaussian transform. `mangadap`'s
`convolution_variable_sigma` instead builds a banded matrix of *sampled* Gaussians, which
degrades exactly where C17 predicts: when the kernel is undersampled. `varsmooth` is the
production path; the mangadap routine is retained as an independent cross-check (Phase 2,
verification item 8).

> **One upstream constant the port depends on.** `varsmooth` clips the kernel σ to **≥ 0.1
> pixels** internally (`sig = sig.clip(0.1)`). It is not exported and not documented as API. The
> port hard-codes 0.1 as `dc3.core.resolution.VARSMOOTH_MIN_SIG` and asserts it with a
> behavioural test — request a sub-0.1-pixel kernel, measure the realised width — so an upstream
> change fails loudly rather than silently biasing `dvar_inst`. See Phase 2.

---

## Phase 0 — Report and repository scaffold

**Deliverable 1: `claude/dc3-original-implementation.md`.** ✅ **Complete.** The report covers:

1. Scientific background from Westfall et al. (2011) — Eqs. 1–5, the three iteration tiers
   (§3), detector censoring (§2.2), masking (§2.3), FFT considerations (§2.4), the Statler
   (1995) covariance, and Appendix A (instrumental broadening).
2. **Build system**: `makefile` → `make.directory_defs` → `make.binary_defs` →
   `make.linkage_defs`; `lnk/*.lnk` per-executable object lists; env vars; which of the six
   external libraries are genuinely used (hdf5 and GPC are not).
3. **All 19 executables**, each with purpose, CLI, inputs/outputs, control flow, link set,
   and current build state — explicitly flagging the 8 that no longer compile and *why*
   (removed/changed `Correlation::setac`, `VDF::refit`, `VDF::getbtsp(Spectrum)`, etc.).
4. **`DC3_express` traced in full** — argument parsing, `ParSet`, spectrum/mask reading,
   CC construction, CC statistics, `VDF::fit()`, asymmetry, outputs, QA plots.
5. **Where the algorithm actually lives**: not in `dc3/` but in `libraries/vdf.cpp`,
   `correlation.cpp`, `cnvlv.cpp`, `spectrum.cpp`, `contfit.cpp`. The frozen
   `DC3_fitbroad.cpp`/`DC3_err.cpp`/`DC3_chimap.cpp` are the pre-class historical versions.
6. **Dependency map**: every library module → what it provides → its Python replacement;
   every NR routine used → its Python replacement; the `vec.h` type system.
7. **File formats**: `ParSet` and legacy 40-line positional parameter files, the spectral
   region-mask table, spectrum lists, FITS keyword dictionaries, `.db` table formats.
8. **Documentation digest** — `doc/descrip.txt` (the de-facto spec), `doc/develop.txt` (the
   empirical basis for hardwired choices: mean subtraction, no apodization, σ_min = 0.85 px
   ≈ Nyquist, LM over amoeba), `doc/sckat.todo.readme`, `notes_improve_dc3`.
9. **Defects found**, including three argument-indexing bugs in `DC3_express.cpp`
   (`-SG`/`-ST` off-by-one at lines 212–213; `fixed_window` tests `-v` instead of `-f` at
   line 215, silently switching to `FIXV` windowing with `fvs = fve = -1`) and a
   `win.invs`/`win.inve` mix-up at `vdf.cpp:487`.

**Deliverable 2: repository scaffold.** `git init`; BSD-3 `LICENSE.rst`; `pyproject.toml`,
`tox.ini`, `readthedocs.yml`, `.github/workflows/ci_tests.yml` adapted from PypeIt (all three
reference packages share one skeleton); `README.rst`, `CHANGES.rst`, `CITATION.cff`,
`CLAUDE.md`; `.gitignore`.

---

## Phase 1 — Package infrastructure

PypeIt is the structural template, but this phase is **not** a wholesale lift. Two of its
central abstractions — `ParSet` and `DataContainer` — are being actively reconsidered in
PypeIt itself, and `dc3` is a small enough code base to be the right place to try the next
design rather than inherit the current one.

### Lifted essentially as-is

| New file | Source to adapt | Notes |
|---|---|---|
| `dc3/pkg/logger.py` | `pypeit/pkg/logger.py` | Already proven portable — nirvana ships a copy. |
| `dc3/pkg/exceptions.py` | `pypeit/pkg/exceptions.py` | `DC3Error`, `DC3DataModelError`. |
| `dc3/pkg/cache.py`, `dc3/pkg/dc3data.py` | `pypeit/pkg/cache.py`, `pypeitdata.py` | On-demand download of template libraries and test data; keeps the wheel small. |
| `dc3/core/bitmask.py` | `pypeit/core/bitmask.py` + `pypeit/images/bitmaskarray.py` | See "Bitmasks" below. |
| `dc3/scripts/scriptbase.py` | `pypeit/scripts/scriptbase.py` | Heavy imports inside `main()`. |

These are BSD-3, so reuse is clean with attribution in `LICENSE.rst`/`licenses/`. The `pkg/`
sub-package imports nothing from outside itself.

### Parameters — `dc3/par/`

**Start from the current PypeIt design, not the released one.** The source is
**`funcpar_rebase`** in `~/Work/packages/pypeit` — `FuncPar` rebased onto `parset_refactor`, so
the two sit on one linear history and can be read as a single design. It supersedes
`funcpar_update`, which was branched from a pre-`parset_refactor` merge base; **do not read
`funcpar_update`** — it is retained only as the rebase's source.

- **`parset_refactor`** (the base) — `pypeit/par/parset.py` (1,054 lines):
  `set_parameter_definition(dtype, default, default_factory, options, descr)` replacing the
  old parallel-list constructor, plus `validate`, `to_rst_table`, `config_lines`/`to_config`,
  `to_dict`/`from_dict`, and `to_header`/`from_header`.
- **`funcpar_rebase`** (three commits on top: `17fc59fce`, `1095cd100`, `a018658a8`) — adds
  `pypeit/par/funcpar.py` (271 lines) and `utils.get_func_kwargs`, plus tests and API docs. The
  diff is **purely additive**: it touches no existing `parset.py` code, so adopting `ParSet`
  without `FuncPar`, or adding `FuncPar` later, are both clean.

**What `FuncPar` does.** It is an abstract `ParSet` subclass that **derives its parameters from a
function's signature**. A subclass declares the function and, optionally, which keywords to take:

```python
class VarsmoothPar(FuncPar):
    func = ppxf_util.varsmooth
    kw_subset = ['oversample']          # or omitted_keys = [...] for the complement
```

`__init_subclass__` — the PEP 487 hook, deliberately used *instead of a metaclass* — then builds
the `parameters` dictionary at class-creation time: defaults come from the signature, and
`dtype` is inferred from the function's own type annotations via `typing.get_type_hints`, which
resolves postponed (string) annotations and collapses `Optional[T]` and `T | U` unions into the
list of concrete types `set_parameter_definition` expects.

#### The governing rule: `FuncPar` wraps *third-party* functions, and nothing else

`FuncPar` carries no `options`, no real `descr` (every parameter gets the generic *"Parameter for
`f` in `m`"*), no positional arguments, and a `dtype` only as good as the wrapped function's
annotations — `None`, i.e. no type checking, for unannotated third-party code.

**That sparseness is the design, not a shortfall**, and it dictates exactly where the class is
used:

| Function | Parameter set | Why |
|---|---|---|
| **Internal to `dc3`** | hand-written `ParSet` subclass | `dc3` owns the parameter, so `dc3` owns its type, options, default and description |
| **Third-party** | `FuncPar` subclass | the upstream package owns all of that, and **`dc3` depends on its documentation rather than reproducing it** |

Reproducing a dependency's parameter documentation guarantees it will drift, and drifted
documentation is worse than a pointer to the authoritative source. So for a wrapped third-party
function, `dc3` states which keywords it exposes and sends the user upstream for what they mean.
The one thing `dc3` *does* own in that case is the **restriction** — `kw_subset`/`omitted_keys`
are `dc3`'s editorial decision about which knobs are appropriate here, and that choice should be
documented even though the parameters themselves are not.

> **Add a documentation-pointer attribute to `FuncPar`.** The class needs a way to say *where* the
> authoritative documentation lives — a Sphinx cross-reference (e.g. a `:func:` role resolved
> through `intersphinx`, which gets a live link and fails the build when the target disappears) or
> a plain URL where the dependency publishes no object inventory. `to_rst_table` and the
> reflection-based parameter-doc generator should render it, so every generated table of
> third-party parameters carries the link. **ppxf is the case that forces the URL fallback** —
> it ships no Sphinx inventory, so `varsmooth`'s and `losvd_rfft`'s parameters can only point at
> the published documentation or the docstring. This is an addition to `FuncPar` itself: offer it
> upstream to PypeIt rather than carrying a divergent copy in `dc3`.

**Why this matters more for `dc3` than for PypeIt.** A larger share of this port's parameter
surface wraps external callables — `varsmooth`'s `oversample`, `least_squares` and
`differential_evolution` behind `FitPar.method`, `Legendre.fit`, `sigma_clip`,
`pypeit.core.bspline`, and whichever MCMC sampler is chosen. Each is a place where a keyword list
would otherwise be transcribed by hand and rot silently against upstream. `FuncPar` also fails
*loudly* when it rots: `_valid_default_kwargs` raises `KeyError` at class creation — i.e. at
import — if a `kw_subset` entry is no longer in the signature, so an upstream rename surfaces
immediately instead of at run time.

This also satisfies the 1:1 config-key rule below in its strictest form: the configuration key
*is* the upstream keyword, with no translation layer to get out of step. By the same token,
anything that is part of `dc3`'s own vocabulary must be declared by hand — the four `TemplatePar`
parameters of Phase 2 are `dc3`'s, not ppxf's, and get full `dtype`, `options` and `descr` even
though `epsilon_sigma` ultimately feeds `varsmooth`.

**`FuncPar` is also evidence in the pydantic decision below**, and it cuts against pydantic. Its
whole mechanism is *deriving* a schema from a signature at class-creation time; pydantic's
`create_model` can do this, but the annotation-to-`dtype` handling, the `kw_subset`/`omitted_keys`
filtering, and the `ParSet` integration are already written and tested here. Weigh that in the
prototype.

> One incidental cleanup if `funcpar.py` is adapted: it carries `from IPython import embed`, a
> PypeIt house-style debugging import that `dc3` should not inherit.

**Then consider going further: build on `pydantic` v2.** PypeIt already depends on pydantic
(`pyproject.toml`, used in `pypeit/state/`, `pypeit/dashboard/model.py`), but *not* for
`ParSet` — the parameter system is still hand-rolled. `dc3` is small enough to test whether
pydantic can replace the hand-rolled machinery outright:

| Hand-rolled in `ParSet` | pydantic equivalent |
|---|---|
| `dtype` list + manual `isinstance` checks | field type annotations + coercion |
| `options` | `Literal[...]` or `Enum` |
| `default` / `default_factory` | `Field(default=...)` / `Field(default_factory=...)` |
| `validate()` | `@field_validator` / `@model_validator` |
| `to_dict` / `from_dict` | `model_dump()` / `model_validate()` |
| nested `ParSet` members | nested `BaseModel` |
| `descr` | `Field(description=...)` |

What pydantic does *not* give for free, and must still be written: `to_rst_table`,
`to_header`/`from_header` (FITS 8-character-keyword round-trip), and the layered
default→file→CLI merge. Those are ~200 lines, against ~1,000 replaced. **Decide this by
prototyping one ParSet both ways before committing**, since it is a structural choice that is
expensive to reverse.

**Configuration files: TOML, read with the standard-library `tomllib`.** Not ConfigObj, not
ini. The governing rule, which `mangadap` violates and should not be repeated:

> **There must be a direct 1:1 mapping between a key in the configuration file and the
> parameter used throughout the code.** No renaming between the file and the object, no
> derived parameters that exist only in one of the two, no keys consumed in one place and
> reinterpreted in another. If a user reads a config file, they can grep the code for the key
> and find every place it acts.

TOML is write-only from the standard library (`tomllib` has no dumper), so use `tomli-w` or a
small writer for `to_config`. This is a minor cost against getting typed parsing for free —
ConfigObj returns everything as strings and forces a whole layer of casting that
`set_parameter_definition`'s `dtype` machinery exists partly to undo.

`dc3/par/dc3par.py` defines the parameter sets themselves: the 15 keys of
`progfiles/dc3_express.par` plus the values currently **hardwired** in `DC3_express.cpp`
(`apwin=0`, `cosper=2`, `minsig=0.85`, `winfac=2.5`, `alambda=0.001`, `lam`, `del`),
organized as `CorrelatePar`, `ConvolvePar`, `MaskPar`, `WindowPar`, `ContinuumPar`, `FitPar`,
`TemplatePar`, `DC3Par`. Exposing the hardwired constants is a genuine improvement — several
are flagged "NEED TO REVISIT" in `doc/develop.txt` (notably the fit-window width, empirically
1.0–1.7 FWHM but defaulted to 2.0).

**Automatic documentation generation must survive whichever design wins.** PypeIt's
`doc/scripts/build_par_rst.py` and `build_datacontainer_datamodels.py` generate parameter,
datamodel, bitmask, and `--help` documentation by reflection, so docs cannot drift from code.
Keep that property; it is non-negotiable and is a constraint on the pydantic decision above
(pydantic's `model_json_schema()` makes it easier, not harder).

### Data products — `dc3/datamodel.py`

**Do not adopt `pypeit.datamodel.DataContainer` wholesale.** It works, but it fuses two
concerns that are cleanly separable, and this port is a good opportunity to separate them:

1. **I/O infrastructure** — arrays, tables, and metadata to and from multi-extension FITS
   (and back), with header provenance, versioning, and compression. Generic; knows nothing
   about what the data *means*.
2. **Datamodel enforcement** — the declaration of which components exist, their dtypes,
   shapes, units, and interrelations, and strict validation against that declaration.

Splitting them means the strictness can be tightened or relaxed independently of the file
format, subclasses can opt into validation without inheriting the FITS machinery, and the
validation layer becomes testable without touching disk. As with `ParSet`, **consider
`pydantic` for layer 2** — the datamodel declaration is exactly a schema, and the two layers
would then share one validation technology.

The same prototype-first rule applies: implement `dc3/results.py` (Phase 4) against both a
`DataContainer`-style single class and a split design before fixing the architecture.

### Bitmasks

Adopt `pypeit/core/bitmask.py::BitMask` for the bit definitions, **and also**
`pypeit/images/bitmaskarray.py::BitMaskArray` for the arrays themselves — it gives
attribute-style flag access (`mask.CR`, `mask.turn_on(...)`) over a plain integer array, which
is far more readable than repeated `bitmask.flagged(arr, flag='CR')` calls. Note that
`BitMaskArray` *is a* `DataContainer` in PypeIt, so its adoption is coupled to the decision
above; if the split design wins, `BitMaskArray` should sit on the I/O layer.

---

## Phase 2 — Spectral core

### I/O boundary vs. internal representation

Keep these strictly separate, because they have opposite requirements:

- **I/O uses `specutils`.** `Spectrum` / `SpectrumList` readers handle the format zoo, WCS
  parsing, unit attachment, and the long tail of instrument-specific FITS conventions. Use
  them at the boundary, plus custom `specutils` loaders for the DC3-era formats needed to
  ingest the test fixtures.
- **Internal representation is bespoke and performant.** Immediately after reading,
  **ingest the parsed `specutils` object into `dc3`'s own spectrum class and never touch
  `specutils` again inside the fit.** `Spectrum` carries `Quantity` arrays, lazy WCS, and
  uncertainty objects whose per-access overhead is irrelevant for scripting and fatal in an
  inner loop evaluated thousands of times per spectrum. The internal class holds plain
  contiguous `float64` arrays with the units fixed by convention and documented, not carried.

`dc3/spectra.py` — the internal class, holding a log-λ-sampled spectrum set: `wave` (or
`log10lam0`/`dloglam`), `flux`, `ivar`, `mask`, `sres` (instrumental σ in km/s), `cont`.
Replaces `libraries/spectrum.cpp` (4,487 lines). It may be built on whatever `dc3`'s
`DataContainer` equivalent becomes (Phase 1) — but *the datamodel layer must not impose
per-access cost*, which is one more argument for splitting I/O from validation: validate once
on construction, then get out of the way.

- Reuse `numpy`, `astropy.io.fits`, `astropy.units`, `astropy.constants.c`. Convert from
  `specutils.Spectrum` via explicit ingest functions; do not subclass it.
- Port carefully: the log-linear grid conventions, flux-conserving resampling, the
  `getmean()`-over-unmasked-pixels semantics (this is load-bearing — see Phase 4),
  `vshift`/`calcz` (use the **relativistic** form consistently; `DC3.h` carries an unresolved
  TODO about this), S/N, and `adjsize`.
- Delete: the FITS mechanics, arithmetic-with-auto-regrid operators, `mywcs.cpp` (→
  `astropy.wcs`), `spline.cpp` (→ `scipy.interpolate`), `str_manip`, `myfuncs`.

### Resampling — `dc3/core/sampling.py`

Adopt `mangadap.util.sampling.Resample` (the same class also exists in `pypeit`). It provides
flux-conserving resampling to and from log-linear grids in one implementation, replacing the
`Spectrum` regridding operators wholesale.

It has one capability worth noting now even though it is not used yet: with `covar=True` (and
`step=True`) it returns the **covariance between output pixels** induced by the resampling.
See science question 4 — the decision for now is to continue ignoring spectral covariance, but
this is the hook if that changes. Note that the `Covariance` class this depends on has been
upstreamed into `astropy.nddata`, so the `mangadap` version is not needed; adapt `Resample` to
call `astropy.nddata.Covariance` rather than copying `mangadap/util/covariance.py`.

### Templates — `dc3/templates.py`

**Adopt `mangadap`'s `proc/templatelibrary.py::TemplateLibrary` nearly wholesale.** This is
where the single-template limitation is lifted at the data level, and mangadap has already
solved the problem properly: libraries defined by key in a config file; a single processing
pass (air→vacuum, resolution matching, log-rebin, trim, renormalize); and **the processed
library cached as a reference FITS file**. Mimic its resolution-matching bookkeeping — which
spectrum has the broader LSF at which wavelength, and what to do where matching is impossible —
since that is the part that is easy to get subtly wrong.

Two adjustments for `dc3`, both from "Spectral resolution" below:

- The processing pass is the **two-step preparation pipeline** specified there, with
  `dvar_inst` carried out of it as a first-class result rather than discarded.
- The **cache key** must include everything that changes the prepared product: library, fiducial
  galaxy resolution, velocity scale, `velscale_ratio`, `epsilon_sigma` and `sigma_floor`. Since
  preparation is once-per-run and no longer depends on any individual galaxy spectrum, the key is
  a property of the run, not of a galaxy.

> **Possible upstream path:** `TemplateLibrary` is not MaNGA-specific and would be useful to
> the wider community. Migrating it into `specutils` (or a small standalone package) would
> serve `dc3`, `mangadap`, and others from one implementation. Worth raising once the `dc3`
> requirements have shaken out — this port is a good forcing function for defining the
> general interface, since it is the second independent consumer.

### Preparation and validation — reuse `mangadap`

`mangadap` contains a substantial body of code for **checking and preparing galaxy spectra and
templates for fitting**, and the overlap with what `dc3` needs is nearly total. Repurpose
rather than rewrite, taking from `proc/ppxffit.py` and `proc/spectralfitting.py`:

| mangadap | Provides |
|---|---|
| `PPXFFit.check_templates`, `check_objects` | shape/finiteness/positivity validation, `sres` handling |
| `PPXFFit.check_pixel_scale`, `obj_tpl_pixelmatch` | that template and object share a velocity scale to within `dvtol` |
| `PPXFFit.fitting_mask`, `initialize_pixels_to_fit` | the fittable pixel range given the velocity offset and template coverage |
| `PPXFFit.check_input_kinematics`, `check_resolution_match` | guess validation, matched-resolution bookkeeping |
| `SpectralResolution.GaussianKernelDifference` (`util/resolution.py`) | W19 Appendix A matching; the reference implementation for the kernel arithmetic below |
| `PPXFFit.ppxf_tpl_obj_voff` | the template/object velocity-offset convention |
| `EmissionLineFit.check_and_prep_input` | the general wave/flux/ivar/mask/sres ingest contract |
| `util/pixelmask.py::SpectralPixelMask` + `ArtifactDB`/`EmissionLineDB` | database-driven masking |

These are BSD-3 and can be copied with attribution. The DC3-specific parts — detector
censoring, the rest-frame common-wavelength calculation, mask transcription between frames —
are *not* in mangadap and are Phase 3.

### De-redshifting — `dc3/core/deredshift.py`

**Shift the galaxy to the approximate rest frame before fitting, by an integer number of pixels
and nothing more.**

The governing constraint — stated in full under "Spectral resolution" below — is that **the
galaxy's flux distribution is never redistributed**. That rules out resampling onto a shifted
grid. But on a **log-λ** grid a redshift is a pure *translation*, so shifting by the nearest
whole pixel is exact: a relabelling of the wavelength axis, with no interpolation, no flux
redistribution, and no induced covariance. The sub-pixel residual is absorbed by the fitted `V`,
which is being solved for anyway.

```
n_shift = round( log10(1+z_guess) / dloglam )        # integer
z_applied = 10**(n_shift * dloglam) - 1              # the exact redshift removed
V_reported = V_fit + c * ln(1 + z_applied)           # propagated analytically
```

Truncation at the ends is handled the same way, and the applied shift and truncation are stored
so the reported velocity can always be reconstructed. **Shift the `sres` vector by the same
integer number of pixels**: in velocity units the instrumental dispersion is redshift-independent
per pixel (C17 §2.2), so the vector is re-indexed along with the flux, not rescaled.

C23 §2 gives a cleaner formulation of the velocity/redshift relation than W11's and is worth
following — in particular `V_pec = V_ppxf(x,y) − V_ppxf(bary)` (C23 Eq. 3), which C23 notes "is
always valid, regardless of whether the spectrum was de-redshifted to the rest-frame or not". The
same paper confirms that de-redshifting and not de-redshifting give identical results provided
the instrumental resolution is handled consistently, which is the point of the `sres` shift above.

What this buys: the region-mask transcription by `(1+z)^±1` largely disappears, the fit window
and the initial guess become nearly configuration-independent, and the convention aligns with
ppxf, simplifying the Phase-5 cross-check.

*Do not* offer an interpolating de-redshift as an alternative. It would violate the rule above
and introduce precisely the inter-pixel covariance that science question 4 decides to ignore.

### Spectral resolution — `dc3/core/resolution.py` and `dc3/templates.py`

This is the most substantive scientific change in the port, and the part most worth reading
before implementing. The reasoning is in
[`plan-reassessment.md`](plan-reassessment.md) §1 and §8.2; what follows is the specification.

#### Two rules

1. **The galaxy is data and is never altered.** Its flux distribution is never redistributed,
   its resolution is never changed. Only the *templates* are prepared.
2. **Resolution matching is preparation, never part of the cost function.** A
   wavelength-dependent convolution is far more expensive than a wavelength-independent one, so
   inside the fit the broadening kernel is a single scalar σ. Everything wavelength-dependent
   happens once, up front.

A third follows from the second: **no deconvolution, ever** — it amplifies noise, so the
preparation kernel must be real at every wavelength.

#### `dvar_inst`: a signed instrumental variance

Write the relation between the fitted and astrophysical dispersions in terms of a **signed
variance in (km/s)²**, with `σ_T'` the template resolution *after* preparation:

```
dvar_inst  ≡  σ_G² − σ_T'²                (signed; may be negative)
σ_obs²     =  σ_*² + dvar_inst            ⇒    σ_*²  =  σ_obs² − dvar_inst
```

Using a signed variance rather than W19's `δσ_inst²` keeps one convention across both regimes,
removes the square roots from the bookkeeping, and avoids an imaginary intermediate when the
quantity is negative.

| `dvar_inst` | Template vs. galaxy | Regime | Consequence |
|---|---|---|---|
| **> 0** | sharper | MaNGA / W19 | `σ_obs ≥ sqrt(dvar_inst) > 0` — the measurement is held away from the `σ → 0` boundary, where its posterior turns inverse-gamma and the fit piles up against the limit (W19 §7.4.3, Fig. 15) |
| **= 0** | matched | — | no pedestal either way |
| **< 0** | broader | DMS / W11 | a **floor on measurable σ** at `sqrt(\|dvar_inst\|)`; **emit a warning** |

**Both signs are legitimate and both must work.** MaNGA had templates sharper than the data and
could take a positive offset. The DMS observed its templates with the *same instrument* as the
galaxies, where small differences in conditions meant the templates usually had to be *degraded*
to hold any pedestal at all — a negative offset, whose floor was understood as an observational
limitation rather than a defect. The C++ sits in this regime: `res_base` makes the template
uniformly broader, cancels inside `getbtsp`, and floors the measurable σ at values that reached
9.50 km/s in production (report §8.3, §16.4). The port must track the sign, report it, and warn
when it is negative — never silently return σ = 0 as the C++ does.

#### Template preparation, in two steps

Run **once per execution**, never per spectrum:

| Step | Operation |
|---|---|
| **1** | Resolution matching to a **fiducial** galaxy resolution, offset by `dvar_inst` |
| **2** | Resampling to the galaxy's sampling, at an integer `velscale_ratio` |

Both steps make velocity-dispersion corrections unavoidable, and the plan should say so rather
than implying matching makes them go away:

- **Step 1 is approximate by construction, for two reasons.** One matching operation serves all
  templates against a *fiducial* galaxy resolution, and unless every galaxy spectrum has
  identical resolution — MaNGA's did not, and DMS data will not — the fiducial matches no
  individual spectrum exactly. Less obviously, matching is also **redshift-dependent**: C17 §2.2
  notes that in velocity units (or `R`, or `ln λ`) the instrumental resolution is independent of
  redshift *for every spectral pixel*, "but the pixel wavelength changes", so "the matching of
  the templates resolution … must still be performed for every different redshift". The fiducial
  is therefore a fiducial resolution **and** a fiducial redshift.

  > **This is a deliberate trade.** The C++ re-matches inside the tier-2 loop, partly for exactly
  > this reason — `match_resolution` shifts the galaxy resolution spline onto the template
  > wavelengths by `zw = log10(1+z)` using the *current* fitted velocity (report §8.3). Moving
  > preparation out of the loop buys a large speed-up and a `dvar_inst` that is known rather than
  > re-derived, at the cost of per-spectrum redshift registration. The residual is second-order
  > for a survey spanning a narrow redshift range and grows with the span; it is one of the things
  > the Phase 7 fitted estimator exists to detect.

- **Step 2 changes the effective template resolution** — and so does Step 1. See "Characterizing
  the pipeline" below.

Because preparation is once-per-run, `dvar_inst` is common to every spectrum in the run, and the
minimum over wavelength that sets it is taken over the **whole galaxy set**, not a representative
member.

#### The four parameters — `TemplatePar`

| Parameter | Meaning | Default |
|---|---|---|
| `velscale_ratio` | integer prepared-template pixels per galaxy pixel | `1` |
| `epsilon_sigma` | target for the **minimum** dispersion of the preparation kernel, in pixels | `0.1` |
| `sigma_floor` | largest pedestal allowed to accommodate template regions of *lower* resolution than the galaxy; sets the most negative `dvar_inst` | `0` |
| `mask_unmatched_sres` | mask template regions that cannot reach the target resolution | `False` |

Matching follows **W19 Appendix A** — the reference implementation is
`mangadap.util.resolution.SpectralResolution.GaussianKernelDifference` — with option 2 (a
constant pedestal) as the default, with the target adjusted to permit negative `dvar_inst` so as
to minimise the span of unmatchable regions, and with `mask_unmatched_sres` deciding what happens
to whatever remains.

**`epsilon_sigma` is a two-sided target, not a one-sided floor.** This is the one place the port
deliberately departs from `GaussianKernelDifference`, and the difference is worth stating because
anyone comparing the two will notice it. W19's option 2 only ever *raises* the kernel, so
`dvar_inst ≤ 0` in every case and the positive-pedestal regime is unreachable. Here the minimum
kernel σ is driven to *equal* `epsilon_sigma`, lowering it where it would otherwise be larger:

```
res_match(λ) = σ_G(λ)² − σ_T(λ)²                     the exact-matching kernel, squared
δ²           = min_λ res_match − epsilon_sigma²      signed
kernel(λ)    = sqrt( res_match(λ) − δ² )             ⇒  min_λ kernel = epsilon_sigma exactly
dvar_inst    = δ²
```

This reproduces `GaussianKernelDifference` whenever `min_λ res_match < epsilon_sigma²` and
extends it in the other direction — a **superset of the W19 behaviour, not a contradiction**.
`sigma_floor` caps the negative excursion.

Since `epsilon_sigma` defaults to the smallest usable value, the default behaviour makes `δ` **as
large as the no-deconvolution bound allows**, which is what a user wants and requires no number
from them.

**Why 0.1 pixels, and not 0.** C17 §4.3 shows that the analytic Fourier transform of the Gaussian
stays accurate for a severely undersampled kernel, so on the convolution alone
`epsilon_sigma = 0` would be defensible. The constraint comes from elsewhere. C23 Algorithm 1
works by *stretching* the coordinate by `σ_max/σ` so that the variable-σ kernel becomes constant,
and that stretch diverges wherever `σ → 0` — the number of resampled points `n = ceil(s_p − s_1)`
would grow without bound. `ppxf_util.varsmooth` therefore contains `sig = sig.clip(0.1)`, which
bounds the stretch by silently raising any smaller kernel to 0.1 px.

The clip is **not** in Algorithm 1 as published; it is an undocumented implementation detail, and
that is exactly why the port asserts it behaviourally (verification item 10). If `epsilon_sigma`
were set below it, the code would believe it applied a smaller kernel than it did, the prepared
template would be broader than the bookkeeping claims, and **`dvar_inst` would be wrong by the
difference**, biasing `σ_*` low. Default to the clip, explain the stretch-divergence reason in
the parameter description, and reject smaller values. Units in pixels match
`GaussianKernelDifference`'s `min_sig_pix`.

**Two distinct oversampling knobs — do not conflate them.**

| Knob | Acts on | Purpose |
|---|---|---|
| `varsmooth(..., oversample=m)` | the internal *stretched* grid of C23 Algorithm 1 (`σ_max = max(σ)·m`) | reduces interpolation error *within* the resolution-matching convolution |
| `velscale_ratio` | the *output* template grid relative to the galaxy | keeps the prepared template's LSF Nyquist-sampled |

Expose both. They address different error terms and the first is cheap.

**Pass `varsmooth` the log-λ coordinate.** Algorithm 1 converts σ to pixels as `σ/∇x` using a
*centred finite-difference* gradient rather than a declared pixel size, which C23 notes
"coincide[s] in uniformly-sampled regions of the spectrum". On a log-λ grid that condition holds
exactly; on a linear-λ grid it does not. Since DC3 works in log λ throughout, pass that
coordinate and the conversion is exact.

#### `velscale_ratio` and the second bound on `δ`

`δ` is bounded from two directions, not one:

| Bound | Source |
|---|---|
| `δ² ≤ min_λ res_match` | no deconvolution |
| the prepared template's LSF must stay Nyquist-sampled on its own grid | pixelization |

The second is relaxed by oversampling the template, which is what `velscale_ratio` buys: broaden
and shift on the finer grid, bin down afterwards, exactly as ppxf does. Default `1`, with
optional automatic selection on the criterion that the **LSF FWHM be sampled by at least 2
pixels** (`σ_T' ≥ 0.85` px). Worth noting in the docs that this is numerically the same Nyquist
threshold as the C++'s `minsig = 0.85 px` block-replication floor (report §8.4) — the same
criterion applied to two different objects, not a coincidence.

#### Input contract: resolution vectors are **pre-pixelized**

`dc3` assumes both `sres` vectors describe the LSF **before** integration over the spectral
channel, matching W19 §7.1.5 (which uses MaNGA's `PREDISP`, not `DISP`) and what ppxf assumes.
The code cannot verify this, so it is an input contract and must be documented where users will
actually meet it: in the `sres` datamodel description, in the parameter documentation, and in the
tutorial — not only on an algorithm page. Getting it wrong biases every `σ_*` silently.

For MaNGA-style inputs, name `PREDISP` explicitly; that is the most likely concrete mistake. Add
a **soft diagnostic**, not a check: if the supplied `sres` implies `σ_inst ≲ Δ/√12`, it is smaller
than pixelization alone would produce and is probably post-pixelized or simply wrong. Warn, do
not fail.

#### When matching does not happen

If either `sres` vector is absent, matching is impossible — the C++ handles this by returning
early with `res_base = 0` (report §8.3) and the port should do the same: set `dvar_inst = 0` and
**warn that the reported `σ_obs` is uncorrected**. There is deliberately *no* switch to decline
matching when both vectors are available, because the resulting wavelength-dependent offset
cannot be represented by the single scalar σ of the forward model.

#### Characterizing the pipeline — a Phase 2 deliverable

Both preparation steps change the effective template resolution, and the pre-pixelized convention
captures neither.

**Step 1 does, because `varsmooth` interpolates twice.** C23 Algorithm 1 stretches the coordinate
(`y_new = interpolate(s, y)(x_new)`), convolves, then interpolates back
(`y_conv = interpolate(x_new, y_conv)(s)`). C23 §3.1 is candid about what that costs, and the
passage is worth quoting because it is the clearest upstream statement of the problem:

> *"Most interpolation methods can be described as a convolution with a specific kernel … and one
> may think it would be better to remove the effect of this extra convolution as done, e.g. in the
> NUFFT methods. However, this situation is different as the spectra have noise, and one would
> have to perform the interpolation in a Bayesian framework. One should also consider that the
> spectra to fit generally already include additional interpolation and resampling, which would
> have to be modelled for rigorous results. But all this is unlikely to affect scientific results,
> and for this reason, it is beyond the scope of this paper."*

Three things follow. The effect is **real and acknowledged by the author of the algorithm**; the
right response is *not* to deconvolve it away, for exactly the reason C23 gives — noise makes that
ill-posed, which is the same "no deconvolution" rule stated above; and its magnitude is
**unquantified upstream**. `varsmooth`'s `oversample` parameter is the lever that reduces it.

**Step 2 does too**, and this part is unaffected by anything ppxf does. Nominally it should not:
the intrinsic LSF is a property of the optics, not of the sampling, so the *vector* is unchanged
by resampling. But the **array** being resampled is not the intrinsic spectrum — it already
carries integration over its own native pixels of width `Δ_tpl`. Rebinning onto a grid of width
`Δ_new` gives, schematically,

```
true ⊗ LSF ⊗ tophat(Δ_tpl) ⊗ [resampling kernel ≈ tophat(Δ_new)]
```

where a spectrum *natively* sampled at `Δ_new` would carry only `tophat(Δ_new)`. The excess is of
order `Δ²_tpl/12` — the variance of a top-hat of width `Δ`, which is the entire content of the
`DISP`/`PREDISP` distinction — and it **cannot be removed without deconvolution**. It is
irreducible: oversampling shrinks `Δ_new` but leaves `Δ_tpl` alone, because that term is a
property of the data as delivered. So the pre-pixelized convention quietly assumes the two native
samplings cancel; they do when the samplings are similar and do not when they are not, and
nothing in the convention signals which case you are in. ppxf inherits the same gap — C17 treats
pixel integration only for gas emission lines (Eq. 28), never for stellar templates.

**Do not add an analytic `Δ²/12` term.** Under the pre-pixelized contract it would double-count,
it assumes a top-hat response the resampler does not actually apply, and it would in any case
miss the Step-1 interpolation entirely. **Measure the pipeline instead** — which is precisely the
thing C23 sets aside as beyond its scope, and which is cheap here because DC3 prepares templates
only once per run:

1. Build a synthetic spectrum of narrow, isolated Gaussian lines of **known** width `σ_in`
   (≈1 native pixel — resolved enough to avoid interpolation artefacts, narrow enough to be
   sensitive), on a wavelength grid spanning the template range.
2. Push it through the **exact** preparation pipeline — Step 1 at the chosen `epsilon_sigma` and
   `varsmooth` oversampling, then Step 2 at the chosen `velscale_ratio`. Measuring the composite
   is the point: the two steps' interpolation errors are not independent and there is no reason to
   separate them.
3. Fit the output widths and subtract `σ_in` in quadrature.

The result is the *measured* effective `σ_T'(λ)`, including every resampling, interpolation and
clipping effect, with no analytic model of the pixel response at all. It is an impulse-response
measurement of the preparation pipeline, it runs **once per configuration** rather than per
spectrum, and it doubles as a regression test on Steps 1 and 2 together.

**Deliverable:** a table or figure of measured-minus-nominal `σ_T'` as a function of
`velscale_ratio`, `varsmooth` oversampling, and the template/galaxy sampling ratio, with the
measured `σ_T'(λ)` and its departure from nominal recorded in the output header. Use the measured
value in `dvar_inst` where it differs significantly. Neither `mangadap` nor `ppxf` does this —
C23 judged the effect "unlikely to affect scientific results", which is very probably right at
MaNGA's dispersions but is an assumption rather than a measurement, and DC3 operates at
dispersions several times smaller where the same absolute error is a larger fractional one.

#### Implementation notes

- `dc3/core/resolution.py` wraps `ppxf_util.varsmooth(x, y, sig_x, xout=None, oversample=1)`
  behind `match_resolution(...)`, keeping the ppxf import in one module.
- `mangadap.util.resolution.convolution_variable_sigma` is the cross-check, not the production
  path (see "External dependencies").
- The C++'s `res_base` / `MIN_SIG_RESMATCH` machinery is **replaced**, not ported. `getbtsp`'s
  `sqrt(σ_fit² − base_sig²)` disappears: the fit returns `σ_obs` directly and the correction to
  `σ_*` is applied afterwards, from a `dvar_inst` that is known exactly because it was chosen.
- Report per spectrum: `σ_obs`, `dvar_inst`, `σ_*`, and **both** floors — the `dvar_inst` floor
  when negative, and the block-replication floor `minsig·dv/maxblk` (Phase 3). They have
  different causes and a user needs to know which one bit.

---

## Phase 3 — Correlation core

`dc3/core/fft.py` — FFT plan management. Lift nirvana's `models/beam.py` pattern
(`/Users/westfall/Work/packages/nirvana/nirvana/models/beam.py`):

- A module-level `convolve_fft`-style numpy fallback and a `ConvolveFFTW` plan-cache class
  with **identical call signatures**, dispatched by `_cnv = fallback if obj is None else obj`.
- `pyfftw.empty_aligned` buffers + `pyfftw.FFTW` plans built once with `FFTW_MEASURE`.
- `__reduce__` returning `(cls, (shape,))` so plan objects survive pickling into a
  `multiprocessing.Pool` — essential for Phase 5.
- `pyfftw` as an optional `[fftw]` extra with a `requires_pyfftw` pytest marker.
- **Improve on nirvana** where it matters for 1-D real spectra: use `rfft`/`irfft` rather than
  full complex transforms (~2× saving), accept a `threads=` argument, and add
  `pyfftw.export_wisdom`/`import_wisdom` persisted via `dc3/pkg/cache.py` so plan-measurement
  cost is paid once per machine rather than once per process.
- There is **no power-of-two constraint to drop** — `Correlation::setsize()` already uses
  `nn = int(2.2·(int((ln−l0p)/dlp)+1))`, made even, and `fft.cpp` already plans real-to-complex
  transforms of arbitrary length (report §8.2, verified against production output §16.2). Keep
  the 2.2× rule, since the factor ≥ 2 is required to treat one spectrum as the response function
  for the other and the extra 10% is deliberate padding; round the result up with
  `scipy.fft.next_fast_len` for speed. **What can be improved is the plan quality:** the C++
  uses `FFTW_ESTIMATE`, so switching to `FFTW_MEASURE` with persisted wisdom is a free gain.
  ⚠️ Rounding `nn` up **moves the velocity grid**, since `v0 = −(nn/2)·dv` — so a fixture
  comparison against the archived `_xc.fits` (which has `nn = 4508`, `v0 = −24823.75` km/s) is no
  longer an array-to-array comparison. Provide `CorrelatePar.exact_cpp_length` to reproduce the
  C++ sizing exactly, used by that test and nothing else.

`dc3/core/convolve.py` — LOSVD convolution. Port `cnvlv_blk` **block replication** (the
sub-Nyquist σ trick: replicate by up to `maxblk=8` when σ < 0.85 px, convolve, average back;
below `0.85·dv/8` set σ ≡ 0) and the four padding modes, which are exactly
`numpy.pad(mode='wrap'|'reflect'|'edge'|'constant')`.

`dc3/core/losvd.py` — Gaussian and Gauss–Hermite broadening functions. Where the C++ builds
`B` in real space and FFTs it, use the **analytic Fourier transform** of the Gaussian /
Gauss–Hermite: this removes one FFT per model evaluation *and* gives closed-form `∂/∂V` and
`∂/∂σ` in Fourier space, which feeds Phase 5.

**Use `ppxf_util.losvd_rfft(pars, nspec, moments, nl, ncomp, vsyst, factor, sigma_diff)`**
rather than deriving and coding the Gauss–Hermite Fourier transform. This is the second reason
ppxf is a dependency; it handles both the Gaussian and the full Gauss–Hermite parameterization
(science question 5). Wrap it behind `dc3.core.losvd.losvd_ft(...)` to keep the dependency in one
module, and unit-test it against a direct real-space build-and-FFT of the same kernel.

Note the `sigma_diff` argument is **not** used for the instrumental offset. Under Phase 2 the
offset is a constant removed *after* the fit via `dvar_inst`, not inside the model, so the C++'s
`σ_conv = sqrt(σ_fit² − σ_base²)` has no counterpart here. `sigma_diff` remains available should a
per-fit quadrature correction ever be wanted.

**Apply the velocity shift in Fourier space, not by interpolation.** The C++ shifts the broadened
template with `shift_lininterp` — real-space linear interpolation — which makes `∂X_T/∂V`
non-smooth and introduces a small sub-pixel systematic. C17 Eq. 37 gives the exact alternative,
`h(t − t₀) ⟺ H(ω) e^{iωt₀}`, which is `losvd_rfft`'s `vsyst` argument: the shift becomes a phase
ramp folded into a transform that is being computed anyway. C17 §4.3 lists this as one of the two
advantages of the analytic transform, the other being that the LOSVD can never be undersampled in
the frequency domain. It composes naturally with `velscale_ratio`: broaden and shift on the
oversampled grid, bin down afterwards.

This **is** a departure from W11 and belongs on the Phase 8 differences page. It is distinct from
galaxy de-redshifting (Phase 2), which is a once-per-run integer shift of the *data*; this is a
continuous shift of the *model* inside the fit.

> **Clean-room rule for the JAX path.** `losvd_rfft` cannot be traced by JAX, so `dc3/jaxmodel.py`
> (Phase 5) needs its own implementation. ppxf's license forbids redistributing its *code*, but
> cannot restrict implementing published mathematics — van der Marel & Franx (1993) for the
> parameterization, C17 §4.3 and Eqs. 33–38 for the Fourier-space form. What *would* breach it is
> a line-by-line translation of `ppxf_util.losvd_rfft` into `jax.numpy`, which is a derivative
> work regardless of the change of language. Protocol: (1) implement from the papers, citing the
> equations in the docstring; (2) do not consult `ppxf_util.py` while writing it; (3) verify
> **numerically** against `losvd_rfft` in the test suite. Numerical agreement is evidence of
> correctness, not of copying.

> **Empirical LSF — keep the seam, defer the implementation.** `notes_improve_dc3` asks for a
> *user-supplied base response function* convolved with the Gaussian during fitting, i.e. an
> empirical instrumental line-spread function instead of the Gaussian assumption. This is
> **low priority**: do not build it now, but do not build anything that precludes it. In
> practice that means `losvd_ft` returns a Fourier-domain kernel, and the call site multiplies
> it into the template's transform — so an empirical LSF later enters as one more factor in
> that product, not as a restructuring. Note that an empirical LSF cannot come from
> `losvd_rfft`, so this is the one place a non-ppxf code path is anticipated.

`dc3/core/correlate.py` — replaces `libraries/correlation.cpp` (2,600 lines). This is real
science, not boilerplate; port deliberately:

- `prep_fft`: apply mask → subtract the mean **of unmasked pixels only** → apodize each
  unmasked block independently (`doc/descrip.txt` §2.C.f). Windows map to
  `scipy.signal.windows.{tukey, hann, bartlett, blackman, hamming}`; the default is no
  apodization, which `doc/develop.txt` justifies empirically.
- `commonwave` / detector censoring: the common *rest*-wavelength range, shrunk by 6σ at each
  end to keep convolution wrap-around out of the correlation. **This is the heart of the
  paper** — get it exactly right.
- `adjmask` / `shift_obj_mask`: transcribe region masks and per-pixel masks between the galaxy
  and template frames by `(1+z)^±1`, and grow/shrink region widths by `2·NSIGMSK·σ_B`
  (`NSIGMSK = 2.0`), per the closed-form expressions in `doc/descrip.txt` §2.D.
- `guessvdf`: σ₀ = sqrt(FWHM²_XC − FWHM²_AC)/2.355.
- Peak metrics: `peakv`, `fwhm`, `fw`, plus the asymmetry function `A(X)` — keep these as
  first-class outputs since they are the quantitative template-mismatch diagnostic of Paper II
  and the user's flagged area for future integration.

`dc3/core/covariance.py` — the Statler (1995) CC covariance, Eq. 4:
`(δX)²_{j,k} = Σ_n T_{n−j} T_{n−k} (δG_n)²`. The C++ builds this as `SSᵀSS` with
`SS[k, j] = T[shift(n,j,k)]·δG[k]` and then **explicitly inverts** a sub-block every time the
window changes (`vdf.cpp:427`).

> **Key optimization:** `(δX)²` depends only on `T` and `δG` — never on `B`. With a fixed
> correlating template it is constant for the whole fit. Build it once per mask iteration,
> `scipy.linalg.cho_factor` it, and evaluate χ² by triangular solve instead of forming `Σ⁻¹`.
> This replaces an O(W³) inverse inside the LM loop with one factorization per tier-2
> iteration, and is numerically better conditioned.

---

## Phase 4 — Single-template fitter (reproduce `DC3_express`)

`dc3/fit.py` — replaces `libraries/vdf.cpp` (2,623 lines). Structure the three tiers of paper
§3 explicitly rather than as nested `for(;;)` loops with an `itertype()` dispatcher.

**Forward model** (`VDF::get_fitting_functions`), preserving the exact semantics:

1. `getbtsp`: slice the template to its original pixel support; convolve with `B` at the fitted
   `σ_obs` **directly** — there is no `sqrt(σ_fit² − σ_base²)` step, because the instrumental
   offset is a constant applied after the fit (Phase 2) rather than inside the model — with
   block replication if needed; re-embed on the correlation grid; apply the velocity shift **as
   a Fourier phase ramp** (Phase 3), not by interpolation; add the Legendre continuum; subtract
   the mean over unmasked pixels.
2. `getbtxc`: substitute the broadened template into the object slot of a `Correlation` and
   re-run **the full masking + apodization + FFT correlation**. This — not a shortcut
   convolution — is what makes the model detector-censored identically to the data.
3. `renormalize`: solve the amplitude `I` analytically (below).

**Tier 1 — parameter estimation.** The objective is fixed: χ² on Cholesky-whitened residuals
`r = L⁻¹(X − X_T)` over the fit window, under the Statler covariance. **The sampler/optimizer
behind it is user-selectable**, via one `FitPar.method` keyword, with three backends sharing a
single objective function:

| `method` | Backend | Use |
|---|---|---|
| `lsq` *(default)* | `scipy.optimize.least_squares` | production; fastest; analytic Jacobian (Phase 5) |
| `de` | `scipy.optimize.differential_evolution` | globally robust; replaces the randomized-restart hack |
| `mcmc` | `emcee` / `numpyro` / `dynesty` — optional `[mcmc]` extra | posterior errors; degeneracy diagnosis |

`lsq` replaces the hand-rolled LM in `VDF::dofit`. Retain: the randomized restarts against local
minima (`VDF::fit()` allows up to `fititer` improvements from `5·fititer` attempts — carry the
mechanism, not the number; the `progfiles` template says 5 and production used 2); the parameter
clamps (`I > 0`, `0 < σ ≤ 600 km/s`, `V` inside the window); and `σ → 0` when below
`minsig·dv/maxblk`. Use `result.active_mask` to flag boundary-pinned parameters into a `BitMask`
(nirvana's pattern).

> **These clamps apply to `σ_obs`, not `σ_*`.** With a positive `dvar_inst` (Phase 2),
> `σ_obs² = σ_*² + dvar_inst`, so the block-replication floor sits below the physically
> interesting range rather than cutting through it — much of the point of the change. With a
> *negative* `dvar_inst` the two floors compound, and the more restrictive one wins. **Report the
> boundary-pinned fraction as a run-level statistic**, since it is the direct observable behind
> W19 Figure 15 and the cheapest check that the pedestal is doing what it should.

`de` exists because the randomized restarts are a workaround for a genuinely multi-modal
surface (`doc/develop.txt` records the amoeba-vs-LM experiments). A proper global optimizer is
the modern answer; if it proves as reliable and not much slower, the restart machinery can be
retired.

`mcmc` should use a broadly used, well-maintained sampler rather than a hand-rolled one —
Hamiltonian (`numpyro`/NUTS, which composes naturally with the Phase-5 JAX model) or nested
sampling (`dynesty`, which also returns the evidence and handles multimodality). **Defer the
specific choice**; make the seam a thin adapter so more than one can be tried. The forward
model is identical in all three cases, so this is genuinely a backend swap.

*Amplitude.* `VDF::calc_norm` already solves `I = (X_T·Σ⁻¹·X)/(X_T·Σ⁻¹·X_T)` in closed form.
Keep this as the default (`-a`/`solve_norm`) — it removes a parameter from the nonlinear
problem, and it is precisely the K = 1 case of the weight solve in Phase 6.

**Uncertainties.** Three routes, with a clear default:

1. **Analytic, from the precision matrix** — *the default*. The least-squares solution already
   produces `JᵀΣ⁻¹J`; invert it for the parameter covariance. This is free, deterministic, and
   is what the C++ did (`VDF::getcovar`). Report the full covariance, not just the diagonal —
   the `V`–`σ` covariance is scientifically meaningful and the C++ discarded it.
2. **Posterior, from MCMC** — whenever `method='mcmc'`, errors come from the posterior instead,
   with no extra machinery. This is the natural cross-check on route 1's linearity assumption.
3. **Monte Carlo, from repeated fits** — a **separate script**, `dc3_mcerr`, that perturbs the
   galaxy spectrum by its errors, refits `N` times, and reports the scatter. Deliberately not
   part of the main fit: it is expensive, it is a validation tool rather than a production
   path, and keeping it separate makes it trivially parallel (Phase 5). This finally settles
   the comparison `doc/develop.txt` asks for and never got redone (open question 2).

**Tier 2 — mask transcription.** Re-derive the region masks from the fitted (V, σ), re-shift the
per-pixel masks, recompute the common wavelength range, re-correlate, re-derive the window,
refactor the covariance. Converged when `|V − V_mask| < mvdiff` and likewise for σ.

> ⚠️ **Tier 2 no longer re-matches resolution.** The C++'s `Correlation::set_mt_cw` calls
> `match_resolution()` on every mask iteration; under Phase 2 preparation happens once per run,
> so this step simply disappears from the loop. That is a genuine behaviour change from the C++
> and belongs on the Phase 8 differences page. It also removes the most expensive operation from
> the tier-2 iteration.

**Tier 3 — continuum.** Fit a smooth function to `G − T_B` and add it to the broadened
template (*not* subtracted from the CC — the v3.0 design note). Replace
`libraries/contfit.cpp` (1,690 lines of IRAF-derived Cholesky normal equations) entirely,
keeping only the rejection *policy* and the convergence test. Two basis options, selected by
`ContinuumPar.basis`:

- **Legendre** *(default, reproduces the C++)* —
  `numpy.polynomial.legendre.Legendre.fit(..., w=sqrt(ivar))` + `astropy.stats.sigma_clip`.
  ⚠️ numpy's `w` multiplies the *residuals*, so the correct weight is `sqrt(ivar)`, not `ivar`;
  the C++ uses `cw[i] = 1/σ²` in its own normal equations, a different convention that is easy to
  transcribe wrongly.
- **B-spline** — `pypeit.core.bspline`, which provides breakpoint placement, inverse-variance
  weighting, and built-in iterative rejection in one call. Worth exploring because the
  continuum here is an *instrumental/normalization* residual rather than a physical one, and
  a low-order global polynomial is a poor model for it over a wide wavelength range; the
  production runs used `corder = 4` over ~2000 pixels, which is a strong smoothness prior
  chosen for stability rather than fidelity. A spline with few, well-placed breakpoints is
  more flexible where flexibility is needed and stiffer elsewhere.

Preserve the v3.1 finding regardless of basis: **the constant term must be held fixed**,
because the spectrum means are subtracted before correlation, so a free zeroth order
destabilizes the fit. For the spline this means constraining the mean of the fitted continuum
rather than a single coefficient — verify this explicitly, since it is the failure mode the
v3.1 note was written about.

Ordering is v3.2: exhaust tier 3 before each tier-2 update.

`dc3/core/window.py` — the fit-window types (`XZERO`, `PMIN`, `NFWHM`, `FIXV`, `NVSIG`,
`FULL`). Fix the `win.invs`/`win.inve` mix-up at `vdf.cpp:487`, which is a real algorithm bug.
The `-f`/`-v` argument confusion needs no fix: that parser does not survive the CLI redesign.

### The fit API — two levels

> *Settled in* [`plan-reassessment.md`](plan-reassessment.md) *§5.3.*

| Level | Input | Role |
|---|---|---|
| **Outer** — `DC3Fit.fit(...)` | many spectra (one is a valid case) | template preparation, library handling, constraint resolution, iteration, results assembly |
| **Inner** — `fit_one(...)` | exactly one fully prepared spectrum | the fit itself — **the multiprocessing boundary** |

Everything expensive and shared — prepared templates, their FFTs, the correlator, FFTW plans —
is built once in the outer function and passed in. The pool maps over the inner function, which
fixes what `ConvolveFFTW.__reduce__` has to survive (Phase 5). The outer function is also the
plug-in point for pipeline integration (see "Integration into larger workflows").

### Constraining and selecting spectra

Two general capabilities, replacing a special case in the C++:

1. **Fix kinematic parameters to input values** — per parameter, per spectrum.
2. **Fit only a subset of the provided spectra** — a selection mask.

Together these let a user re-fit with σ held at 0 where it was undetermined, which is what the
C++ did *automatically* (`vdf.cpp`: when `p[VSIG]² − base_sig² < EPSDP`, fix σ to 0 and
recursively re-`fit()`). **That automatic path is deliberately removed** — the decision belongs
to the user, not the code, and the C++ behaviour made affected spectra indistinguishable from
genuine σ = 0 measurements (report §8.3, §14.9). Record on the differences page.

Per-spectrum, per-parameter constraints do not fit in a scalar ParSet, so they are supplied as a
**constraints table** — `dc3/constraints.py`, referenced by path from the ParSet:

| Column | Type | Meaning |
|---|---|---|
| `ID` | str/int | spectrum identifier, matched against the input spectra |
| `FIT` | bool | include this spectrum in the fit at all |
| `V` | float | velocity: initial guess, or the fixed value if `V_FIX` |
| `V_FIX` | bool | hold `V` at the supplied value |
| `SIG` | float | dispersion: initial guess, or the fixed value if `SIG_FIX` |
| `SIG_FIX` | bool | hold `σ` at the supplied value |

Missing rows mean "fit normally". A value supplied without its `_FIX` flag is an initial guess,
which covers per-spectrum starting velocities without a second mechanism. The outer function
resolves the table into per-spectrum arguments, so only a single row ever crosses the
multiprocessing boundary.

**Format: fixed-width columnated ASCII** via `astropy.io.ascii` (`fixed_width_two_line`), with
ECSV accepted and FITS as a last resort. This is a file a human writes by hand, so readability
and writability dominate.

> This does **not** contradict the FITS-only decision below. That governs *products the code
> writes*, where precision, units and provenance matter and the C++ `.db` format failed on all
> three. The one cost of fixed-width over ECSV is that dtypes and units are inferred rather than
> declared; mitigate by validating the parsed table against a declared schema on read, and by
> shipping a documented example users can copy.

### Output datamodel — `dc3/results.py`

**Design this from scratch. Do not reproduce the C++ output layout.** Backwards compatibility
is explicitly not a goal, and the old layout — two ASCII `.db` tables plus six per-spectrum
FITS files plus separate `.gpm` files, per galaxy × template × run — is the single clearest
place where that freedom pays off. Requirements:

- **FITS throughout, no ASCII.** The `.db` tables lose precision, carry no units or dtypes,
  have no provenance, and required six separate post-hoc executables (`extperr`, `fitrms`,
  `mkasym`, `redoasym`, `asym_renorm`, `DC3_setflags`) purely to add columns after the fact.
  All six disappear.
- **One file per run, not per spectrum.** A multi-extension FITS file containing: the results
  table (one row per spectrum — `V`, **`SIGMA_OBS`**, **`DVAR_INST`**, **`SIGMA_STAR`**, full
  covariance, the two σ floors and which one bound (Phase 2), `NX`, `NT`, `AX`, `AT`, `AC`, `RX`,
  reduced χ², iteration counts, convergence flags, boundary-pinned flags, template weights from
  Phase 6);
  the per-spectrum functions as 2-D image extensions (`xc`, `xc_asym`, `tpl_rmatch`, `btxc`,
  `btxc_asym`, `btsp`); and their bitmasks as `BitMaskArray` extensions rather than separate
  `.gpm` files.
- **Self-describing and self-contained.** Header provenance: `dc3` version, the full parameter
  set (via `ParSet.to_header`), input file checksums, the template library used, the run
  timestamp, the applied de-redshift pixel shift (Phase 2), and the measured `σ_T'(λ)` from the
  preparation-pipeline characterization together with its departure from nominal. *A results file must contain everything needed to regenerate every QA plot* — see
  below; this is a hard requirement on the datamodel, not a nice-to-have.
- **Units and dtypes declared**, and validated on read.

This is the primary test case for the Phase-1 `DataContainer` design decision: implement it
both ways and let the more workable one settle the architecture.

### QA plots — `dc3/qa.py`

Reimplement `DC3_express::diagnostic_plots` in matplotlib: CC + model + residual, asymmetry
panels, and galaxy/template spectra **as prepared for correlation** with mask regions shaded.
Two structural changes, both driven by the measured cost (the 2013 run wrote a 93 MB
PostScript file, and ~41% of wall time was outside the fit — Phase 5):

**1. Tiered, user-selected.** One `QAPar.level` keyword:

| Level | Produces |
|---|---|
| `none` | no plots at all — the production default for large runs |
| `summary` | one figure per run: distributions of `V`, `σ`, χ², convergence; the fiber map |
| `standard` | the above plus one CC+model+residual figure per spectrum |
| `full` | the above plus per-stage diagnostics (`tpl_rmatch`, `btsp`, `btxc`), continuum fit, asymmetry panels, mask transcription |

**2. Every plot is reproducible after the fact.** Provide `dc3_qa`, a post-processing script
that regenerates **any** QA plot from the input data and the output results file alone, with
no refitting. This means a user can run production with `level=none` and recover any figure
later for the spectra that turn out to be interesting — which is the normal workflow, since
one does not know which spectra are interesting until the run finishes.

The rule this imposes on the rest of the design: **anything a plot needs must be either stored
in the results file or recomputable deterministically and cheaply from stored quantities.**
Some recomputation is fine — re-evaluating the forward model at the *stored* best-fit
parameters, re-deriving the fit window from the stored `V` and `σ`, re-transcribing masks — as
long as it is deterministic (no randomized restarts, no re-optimization) and costs a small
multiple of one model evaluation. If a figure cannot meet that bar, the quantity it needs goes
into the datamodel. Enforce this with a test that generates each figure both inline during a
fit and afterwards from the results file, and compares them.

### Scripts

| Script | Purpose |
|---|---|
| `dc3_fit` | the main fitter |
| `dc3_xcorr` | **build and inspect cross-correlations without fitting** — see below |
| `dc3_qa` | regenerate any QA plot from inputs + results |
| `dc3_mcerr` | Monte Carlo errors from repeated fits |
| `dc3_stage_testdata` | one-time fixture staging (Test data section) |
| `dc3_benchmark` | timing harness (Phase 5) |

**`dc3_xcorr` is worth building early**, not as an afterthought. It runs everything up to but
not including the fit — template preparation, masking, censoring, apodization, correlation — and
writes the CC functions, their statistics (`peakv`, `fwhm`, `fw`), and the asymmetry metrics
`AX`/`AT`/`AC`. Its value:

- **It is the natural first milestone.** Phases 2–3 are complete and testable when `dc3_xcorr`
  reproduces `_ccstat.db` and `_xc.fits` from the archived run, with no fitter in existence.
- **Template screening.** The asymmetry metrics are the quantitative template-mismatch
  diagnostic of Paper II; being able to compute them over a template library *before* fitting
  is how one chooses what to put in the library for Phase 6.
- **Debugging.** When a fit misbehaves, the first question is always whether the CC going into
  it is sane. The C++ answered that only by reading a 93 MB PostScript file.

The equivalent `--no-fit` mode of `dc3_fit` should be the same code path, not a parallel one.

**Milestone: `dc3_fit` reproduces the published algorithm end-to-end on one galaxy/template pair.**

---

## Phase 5 — Performance

The current cost structure: `VDF::mrqabc` computes all derivatives by **central finite
differences** — `2·mfit` extra forward-model evaluations per LM step, each a convolution plus a
full correlation. With `mfit = 3` that is 7 model evaluations ≈ 28 FFTs per LM step, multiplied
by `nmax = 50` steps × `fititer` restarts × tier-2 × tier-3 iterations.

**Measured baseline** (report §16.5), from the archived February 2013 production run — 352
spectra × 1 template, 2048 pixels, `nn = 4508`, single-threaded, 2013 hardware and an
unidentified build (report §16.0). Treat it as an order-of-magnitude anchor, not a calibrated
target; the *ratios* below are more robust than the absolute times:

| | `-a` (analytic `I`) | fitted `I` |
|---|---|---|
| Σ per-fit time | 2826 s | 3747 s |
| Median per fit | 6.5 s | 8.8 s |
| Whole-run wall clock | 47 m 39 s (≈ 8.1 s/spectrum; fits are ≈ 59% of it) | — |

Two things this settles before any optimization work starts: solving the normalization
analytically is already **~25% faster** than fitting it, so `-a` should be the default; and
~41% of wall time is *outside* the fit (CC construction, a 93 MB PostScript QA file, FITS I/O),
so QA-figure generation needs its own budget and an opt-out.

Four independent attacks, in order of expected payoff:

1. **Analytic derivatives.** With `B` built from its analytic Fourier transform, `∂X_T/∂V` and
   `∂X_T/∂σ` are closed-form *in the Fourier domain of the broadening step*. But the derivative
   cannot stay there: detector censoring is applied to the broadened template **in real space**
   before correlation (`getbtxc` re-runs the full masking, apodization and FFT correlation —
   report §8.1–8.2), so each derivative still costs an inverse FFT → mask/apodize → forward FFT →
   multiply by `conj(T̂)` → inverse FFT, about **3 FFTs per parameter**.

   | | forward evals | FFTs |
   |---|---|---|
   | Central differences (current) | 7 | ≈28 |
   | Analytic Jacobian | 1 + 3 × 3 | ≈13 |

   So expect **~2×**, not the 5–7× an earlier draft of this plan claimed. Still the largest single
   algorithmic win and it compounds with items 2–4, but the censoring — the paper's central
   feature — is what prevents the clean Fourier-domain result, and it is not negotiable. Pass the
   Jacobian to `least_squares(jac=...)`.
2. **Cholesky instead of inversion** (Phase 3) — removes an O(W³) inverse from the inner loop.
3. **FFTW plan reuse + wisdom** — plans built once per spectrum shape, `rfft` not full complex.
4. **Multiprocessing.** Fitting is embarrassingly parallel over spectra, over
   (galaxy × template) pairs, and over Monte Carlo realizations. Use
   `concurrent.futures.ProcessPoolExecutor` with the `ConvolveFFTW.__reduce__` trick; expose
   `--ncpu` on `dc3_fit`, `dc3_mcerr`, and `dc3_xcorr`. This is the axis mangadap notably
   lacks, and it is what makes `dc3_mcerr` (Phase 4) a routine tool rather than a one-off
   experiment.

A fifth, non-algorithmic lever, from the same measurement: **`QAPar.level = none`** (Phase 4).
~41% of the 2013 wall clock was outside the fit, much of it writing a 93 MB PostScript file.
Because `dc3_qa` can regenerate any figure afterwards, turning plots off in production costs
nothing.

**Optional `[jax]` extra** — `dc3/jaxmodel.py`: the forward model reimplemented in
`jax.numpy` with `jax.jit` + `jax.jacfwd`, selected by a ParSet keyword. Value: (a) an
independent check on the hand-derived analytic Jacobian, (b) it extends free-of-charge to
Gauss–Hermite moments and to the template weights of Phase 6, where hand-deriving becomes
painful, (c) GPU/`vmap` batching over many spectra. Keep SciPy as the always-available path so
JAX never becomes a hard dependency.

**Benchmark harness** (`dc3/tests/test_benchmark.py` + a `dc3_benchmark` script) recording
wall time per fit across configurations — the C++ `notes` file explicitly asked for timing
benchmarks and never got them.

---

## Phase 6 — Optimal template mixing

The single-template restriction is the user's long-standing TODO (`notes_improve_dc3`:
*"Allow the use to provide a template for each galaxy spectrum"*; `sckat.todo.readme`:
*"allow fitting for the best template combination"*).

**The difficulty.** In CC space a composite template `T = Σ_k w_k T_k` appears on *both* sides:

```
X   = G ∘ T            = Σ_k w_k (G ∘ T_k)                    — linear in w
X_T = (T⊗B)' ∘ T       = Σ_j Σ_k w_j w_k ((T_j⊗B)' ∘ T_k)     — quadratic in w
```

so, unlike ppxf's direct-pixel fit, mixing is not automatically a linear solve — and the
"data" moves as the weights change.

**Default: fixed correlator + non-negative GLS.** Correlate the galaxy against a single fixed
reference template `R` (a high-S/N library member, or the library mean):

```
X(v)       = (G ∘ R)(v)                             — computed once, independent of w
X_T(v; w)  = Σ_k w_k [ (T_k ⊗ B)' ∘ R ](v)          — LINEAR in w
```

Then for fixed `B`, `w` is the solution of a non-negative generalized least-squares problem
under the Statler covariance — `scipy.optimize.nnls` or `lsq_linear(bounds=(0, inf))` on the
Cholesky-whitened design matrix. This is a **strict generalization of machinery that already
exists**: `VDF::calc_norm` is exactly this solve with K = 1. It also preserves the constant
covariance (`Σ` depends on `R` and `δG`, not on `w`), so the Phase-3 factorization still holds.

Nest it exactly as ppxf does: an outer nonlinear solve over (V, σ) and an inner linear solve
for `w`. Per iteration the cost is ~3K FFTs of length N (K = library size); for K = 100,
N = 4096 that is milliseconds with FFTW plan reuse, and it parallelizes trivially.

**Optional: symmetric composite as a fourth tier.** After `w` converges, set `R ← Σ_k w_k T_k`
and repeat. This restores the full symmetry the paper argues for, at the cost of recomputing
`X` and refactoring `Σ` per outer iteration. Ship it behind a parameter so the difference
between the two formulations can be *measured* rather than assumed.

> **Two conditions the linearity depends on.** `Σ_k w_k[(T_k ⊗ B)' ∘ R] = [(Σ_k w_k T_k) ⊗ B]' ∘ R`
> holds only if masking, mean subtraction and apodization are **identical for every library
> member** — so the censoring window must be computed once for the library as a whole, never per
> template. This is a general property of template-mixing fits rather than a DC3 quirk; the same
> is true of the templates ppxf is given. Second, with `K > 1` there is no separate amplitude
> `I`: the weights absorb it, so `FitPar.solve_norm` is meaningless in the multi-template path
> and should be ignored with a note rather than silently honoured.

**Supporting pieces:**
- `TemplatePar.usetpl` modes borrowed from mangadap's iteration modes: fit a global/stacked
  spectrum first with the full library, then offer individual fits only the templates with
  non-zero weight. Stabilizes low-S/N fits and cuts cost.
- Optional Tikhonov regularization on `w` (ppxf's `regul`) for smooth weight distributions
  over an ordered library.
- Weight errors and an effective-template output, so template mismatch remains diagnosable via
  the asymmetry indices `AX`, `AT`, `AC`.
- Validation: with K = 1 the result must agree with the Phase-4 single-template path **to
  machine precision**. (Not bit-identical — an NNLS solve and a closed form do not produce
  identical floating-point results.)

---

## Phase 7 — Instrumental-dispersion corrections

> **This phase is much smaller than it was.** With `dvar_inst` *chosen* at template-preparation
> time rather than estimated after the fact (Phase 2), it is known exactly, and the routine
> bookkeeping — computing it, reporting it, converting `σ_obs` to `σ_*`, warning when it is
> negative — lives in Phases 2 and 4. W19 Eq. 7's flat-average estimator, which W19 §7.1.5
> documents as biased low by a few percent, is **not needed on the primary path at all**.
>
> What remains here is the one thing DC3 can do that the DAP cannot.

Implements Appendix A of W11 and supersedes `app_siginst` plus the 18-step keystroke-scripted
workflow in `siginst_algorithm.notes` (the archived instance is
`sircorrect_feb10/U06918_instrs_{01..18}.db`, report §16.8).

`dc3/core/siginst.py` — **the fitted estimator of `dvar_inst`.** Rather than inferring the
instrumental offset from resolution bookkeeping, measure the broadening difference the data
actually exhibit: fit the fourth broadening function `ℬ` of W11 Eq. A4 using the same CC
machinery with velocity fixed and no continuum iteration, giving
`δσ_inst² = σ_B² − σ_obs² − σ_off²` (Eq. A8) and hence `σ_LOS = (σ_obs² − δσ_inst²)^{1/2}`
(Eq. A2). Propagate errors through `σ_off` and `σ_B`.

Two uses, both worth having:

1. **Validation of the chosen `dvar_inst`.** The preparation pipeline asserts a value; this
   measures one. Systematic disagreement means the preparation is not doing what Phase 2 assumes
   — most likely because the fiducial galaxy resolution is a poor stand-in for the individual
   spectrum, or because the Step-2 resampling effect (Phase 2, "Characterizing the pipeline") is
   larger than the characterization suggested.
2. **A better correction where it matters.** At low `σ_*`, where the correction is a large
   fraction of `σ_obs`, a measured offset is worth the cost of a second fit per spectrum.

Report both the chosen and the fitted values, never silently substitute one for the other.

Ingests per-spectrum instrumental-σ vectors (what the `-SG`/`-ST` inputs supplied); with the CLI
designed rather than transcribed, the old argument off-by-one does not arise.

---

## Phase 8 — Documentation, packaging, release

Sphinx docs under `doc/`, with two generation mechanisms:

- **`numpydoc`** for the API reference. Docstrings are written in NumPy style throughout and
  rendered by the `numpydoc` Sphinx extension — the convention used by numpy, scipy, astropy,
  `pypeit`, and `mangadap`, so it is what every reader of this code already expects. Enable
  `numpydoc_validation_checks` in CI so malformed or incomplete docstrings fail the build
  rather than rendering badly.
- **PypeIt's reflection-based generator scripts** (`doc/scripts/build_par_rst.py`,
  `build_datacontainer_datamodels.py`) for parameters, datamodels, bitmasks and `--help`, so
  those cannot drift from the code (Phase 1).

Narrative content:

- **A tutorial reproducing W11 §5** — the UGC 6918 SparsePak demonstration, now that the data
  is located. This doubles as the headline verification test (item 5).
- **An algorithm page** tying each code path to its equation in the paper.
- **"Differences from Westfall et al. (2011)"** — deliberately framed against *the paper*, not
  against the C++ code. The C++ was never distributed and was used only by the W11 authors, so
  a user-facing account of how the two implementations differ has no audience. What *does* have
  an audience is a clear statement of where this package's algorithm departs from the published
  description. Anyone citing W11 for the method needs to know precisely which parts still apply:

  | Change | Phase |
  |---|---|
  | Templates prepared to a constant `dvar_inst` of either sign, reported and corrected after the fit; no in-model `sqrt(σ_fit² − σ_base²)` | 2 |
  | Galaxy de-redshifted by an integer pixel shift before fitting | 2 |
  | Resolution matching removed from the tier-2 iteration — prepared once per run | 2, 4 |
  | Velocity shift applied as an exact Fourier phase ramp, not linear interpolation | 3 |
  | No automatic σ ≡ 0 refit; the user asks for it via the constraints table | 4 |
  | Optional B-spline continuum basis alongside the Legendre default | 4 |
  | Optimizer and error-estimation routes (`de`, MCMC, `dc3_mcerr`) | 4 |
  | Analytic Jacobian and Cholesky factorization, where they change results rather than only speed | 3, 5 |
  | Optimal mixing of a template library | 6 |
  | Any hardwired constant whose value is revisited — notably the fit-window width | science question 3 |

- **A page on assumptions and limitations**, stating plainly the things a user can get wrong
  without the code noticing: `sres` vectors must be **pre-pixelized** (Phase 2); template spectra
  are treated as **noise-free**, so only the galaxy carries errors; spectral covariance between
  pixels is ignored (science question 4); and a negative `dvar_inst` imposes a floor on the
  measurable σ.
- **A licensing page** stating the BSD-3 terms and the ppxf exception explicitly, so
  downstream packagers hit it before they hit the problem.

The C++-level record — the report, this plan, and the defect catalogue — stays as **internal
markdown under `claude/`**, not in the published documentation.

Then `pip install` from a tag, a Zenodo DOI, and a JOSS-style software note if wanted.

---

## Integration into larger workflows

`mangadap` is the reference for how a fitter gets driven over thousands of spectra. The port
should be structured so the same is possible for `dc3`, without adopting mangadap's specific
patterns:

- **A callable fitter object, not just a CLI.** mangadap's `StellarContinuumModelDef` stores
  `fitpar`/`fitclass`/`fitfunc` as parameters with a documented call signature
  (`model_wave, model_flux, model_mask, model_par = fitfunc(binned_spectra, par=None)`), which
  makes the fitter a plug-in point. Expose an equivalent `DC3Fit.fit(...)` entry so `dc3` can be
  dropped into a pipeline as an alternative to `ppxf`.
- **Prepare once, fit N times.** Cache the processed template library (resolution-matched,
  log-rebinned, trimmed) as a reference FITS file, and precompute the template FFTs once —
  mangadap passes `templates_rfft=` into every `ppxf` call for exactly this reason.
- **A results table, not per-spectrum files.** One `DataContainer` keyed by spectrum ID, with the
  kinematics, errors, template weights, figures of merit, asymmetry indices, and a bitmask.
- **Comparable masking inputs.** Accept emission-line and artifact databases in the same spirit as
  mangadap's `SpectralPixelMask`/`ArtifactDB`/`EmissionLineDB`, so a `dc3` run and a `ppxf` run on
  the same data can be masked identically — a precondition for the W11 §4.1/§5.1 comparison.

## Science questions — resolved

These were raised as open; each now has a decision. Kept here with the reasoning, because they
constrain implementation choices in several phases.

1. **Asymmetry metrics in the fit.** ✅ **Keep `AX`, `AT`, `AC` as diagnostic outputs for now** —
   do not put them in the objective. *But* design for a future mode in which they are
   **recomputed as the fit proceeds**, so their improvement across tier-2/tier-3 iterations can
   be tracked. That is a cheap and genuinely informative diagnostic — if the asymmetry is not
   improving as the mask and continuum iterate, the template is wrong and no amount of
   iteration will fix it. Implementation consequence: `dc3/core/correlate.py::asymmetry` must
   be callable on an arbitrary CC function, cheap, and free of side effects, so the fitter can
   call it per iteration under a parameter; and the results datamodel needs room for a
   *per-iteration* asymmetry trace, not just a final value.

2. **Errors.** ✅ **Default: analytic, from the precision matrix** naturally produced by the
   least-squares solve — free, deterministic, and what the C++ did. **Optional: MCMC**, in
   which case errors come from the posterior instead. **Separate: `dc3_mcerr`**, a standalone
   script generating Monte Carlo errors from repeated fits. See Phase 4, "Uncertainties". The
   three routes together finally answer the comparison `doc/develop.txt` asks for; with Phase-5
   multiprocessing the MC route is cheap enough to run routinely rather than once.

3. **Fit-window width.** ✅ **Agreed — re-derive it.** The 2006 experiments found 1.0–1.7
   CC-peak FWHM optimal; W11 §3 reports ≈2; `DC3_express` defaults to 2.0. Reimplement
   `VDF::rchiset`'s reduced-χ²-vs-window analysis and settle the value once the port can run it
   cheaply. Until then keep 2.0 as the default and expose it as a parameter.

4. **Inter-pixel covariance.** ✅ **Continue to ignore spectral covariance, for now.** Two
   points, both real:
   - `mangadap.util.sampling.Resample` *can* compute the correlation matrix induced by
     resampling directly (`covar=True`). It assumes the *input* spectrum has uncorrelated
     pixels, which is itself dubious for data that has already been through a reduction
     pipeline.
   - Direct inclusion of spectral covariance in the modeling is expected to be prohibitively
     expensive. The Statler covariance `(δX)²` is already a `W × W` dense matrix built from a
     diagonal `(δG)²`; a non-diagonal `(δG)²` turns its construction into a full `N × N`
     triple product per mask iteration. That expectation may be wrong — the Phase-3
     Cholesky-once structure is more forgiving than the C++'s explicit inverse — but it is not
     worth testing until the port works.

   So: do not build it, but **keep `(δG)²` as an explicit object rather than a bare array of
   variances**, so a covariance version can be substituted later without touching the call
   sites. Note that de-redshifting (Phase 2) does *not* bear on this either way: the integer
   pixel shift introduces no covariance, which is one of the reasons it was chosen over an
   interpolating shift. Note also that the templates contribute nothing here — they are treated
   as noise-free, and W11 Eq. 4 carries `δG` with no `δT` term.

   > **If covariance I/O is ever needed, it already exists upstream.** `Resample`'s covariance
   > support was built on `mangadap.util.covariance.Covariance`, which has since been
   > abstracted and **upstreamed into `astropy.nddata.Covariance`** (present in astropy 8.0.1,
   > already a `dc3` dependency). So sparse storage, FITS serialization, correlation/covariance
   > conversion, and variance rescaling are available from the standard stack with no new
   > dependency and nothing to port. Irrelevant under the current decision — but it removes the
   > infrastructure cost from any future reconsideration, which is worth knowing when weighing
   > it.

5. **Gauss–Hermite moments.** ✅ **Agreed — keep the capability, do not make it the default.**
   W11 §1 explicitly defers `h3`/`h4` (Statler 1995: robust higher moments need S/N ≳ 30, near
   the upper limit for DMS spectra). `ppxf_util.losvd_rfft` supports them natively (Phase 3),
   so the capability costs nothing to retain. **New constraint:** the van der Marel & Franx
   parameterization is defined relative to the fitted Gaussian, so higher moments are only
   interpretable when `dvar_inst ≈ 0`. Enabling `h3`/`h4` must therefore require
   `epsilon_sigma` and `sigma_floor` to be set such that the prepared templates are matched
   (Phase 2), and must **fail validation** — not warn — if the resulting `dvar_inst` is
   materially non-zero. W19 §7.4.3 declined to fit higher moments at all for the same underlying
   reason DC3 faces: 40% of DR15 spectra have `σ_* < σ_inst`.

6. **Spectral-resolution matching.** ✅ **Prepare the templates once per run to a constant
   instrumental offset `dvar_inst` of either sign, and correct after the fit**
   (`σ_*² = σ_obs² − dvar_inst`). Full specification in Phase 2, "Spectral resolution". The short
   version: an offset is unavoidable regardless (W19 §7.4.3); a *positive* offset holds `σ_obs`
   away from the `σ → 0` boundary where its posterior turns inverse-gamma, which matters most in
   exactly the low-`σ_*` regime the DMS targets; a *negative* offset is what a same-instrument
   survey usually gets and imposes a floor on measurable σ, so it must be tracked and warned
   about rather than treated as an error. Choosing `dvar_inst` rather than estimating it also
   sidesteps the few-percent bias W19 §7.1.5 documents in its Eq. 7 estimator.

7. **Reporting `σ_*` versus `σ_obs`.** ✅ **Report both, plus `dvar_inst` and its error.**
   Following W19, which deliberately does not publish a corrected dispersion: the correction is
   model-dependent and improvable, so a user must be able to recompute it from stored quantities.
   `σ_*` is offered as a convenience column, never as the only one.

## Out of scope (documented, not ported)

- **Stacking** (`stack`, `combspec`, `dither_stack`, `prob_stack`, `stack_sim`,
  `stack_pfile`) — the user's designated first extension. Report documents all five modes of
  `stack`, `stack.par`, and `libraries/specmanip.cpp::combspec`. These binaries are already
  stale, so this will be a rewrite from the algorithm, not a port.
- **Deeper asymmetry integration** — `AX`/`AT`/`AC` are computed and reported from Phase 4;
  using them inside the objective is decided against for now (science question 1). The
  per-iteration *tracking* mode is in scope as a diagnostic.
- **Empirical instrumental LSF** — a user-supplied base response function convolved with the
  Gaussian during fitting (`notes_improve_dc3`). **Low priority, deliberately deferred.** The
  requirement is only that nothing in the design precludes it: `losvd_ft` returns a
  Fourier-domain kernel and the call site multiplies it into the template transform, so an
  empirical LSF enters later as an extra factor rather than a restructuring (Phase 3).
- **Backwards compatibility with C++ formats and CLI** — not a goal. The old `.db` and
  parameter-file readers exist only as test-fixture ingest paths.
- **An unmatched-resolution fitting mode** — i.e. using templates at their native resolution and
  collapsing a wavelength-dependent `δσ_inst(λ)` to a scalar after the fit, as the DAP does. Not
  supported: the single-Gaussian broadening of the forward model cannot represent it, so the
  model would be misspecified, and a wavelength-dependent kernel inside the cost function is far
  too slow. The only case where matching does not run is when a resolution vector is missing
  (Phase 2), which reports `dvar_inst = 0` with a warning.
- **Interpolating de-redshift** — excluded by the rule that the galaxy's flux distribution is
  never redistributed (Phase 2).
- **Simulation drivers** (`recoverv`, `stack_sim`) — superseded by the Phase-4 test suite.
- **Interactive GUI** (`DC3_menu`/`DC3_plot`/`DC3_userio`, ~7,800 lines) — dropped.
- **Post-hoc table tools** (`extperr`, `fitrms`, `mkasym`, `redoasym`, `asym_renorm`,
  `DC3_setflags`) — subsumed by the results `DataContainer`.
- **Scratch harnesses** (`test_int`, `test_convolve`, `test_spec`) — become unit tests.

---

## Test data

A February 2013 PPak production campaign on `/Volumes/seshat` provides real inputs *and* the
corresponding C++ outputs (report §16).

### Two constraints on how this data is used

**(1) The archived outputs were not produced by the code now on disk** (report §16.0). The
campaign ran 11–14 Feb 2013; `vdf.cpp` was modified in Dec 2014, `spectrum.cpp` and `cnvlv.cpp`
in Dec 2016, `DC3_express.cpp` in Jul 2016, `correlation.cpp` and `fft.cpp` within weeks of the
run. `vdf.h`'s changelog stops at Feb 2013 despite the file being edited in Dec 2014, so the
in-file version history is not a complete record. Three mutually inconsistent versions exist
(Feb 2013 source, Jun 2015 binary, 2016 source) and none can be rebuilt.

Therefore, split the fixtures by provenance:

- **Inputs are version-independent** — galaxy spectra, template, σ_inst vectors, mask table, and
  parameter file are data, not code output. These are fully trustworthy and are the main value
  here: realistic, non-synthetic, correctly formatted inputs at the right sampling.
- **Outputs are indicative only.** They pin down formats and orders of magnitude and make a good
  smoke test, but a numerical disagreement cannot be attributed to the port without first
  establishing which build wrote the table. **Do not gate CI on tight tolerances against them.**

**(2) `/Volumes/seshat` is an external RAID and must not be a dependency.** No module, test, or
script in the package may read from it at run time. Access is confined to a single explicit
staging step, run once, with the mount path as an argument.

### Staging and layout

**Follow PypeIt's pattern: fixtures live under `dc3/data/tests/`, are committed to git, are
excluded from the wheel, and are fetched through the package cache when absent locally.** There
is deliberately **no `dc3/tests/data/` directory** — `dc3/tests/` holds test *code* only.

Add `dc3/scripts/stage_testdata.py` → `dc3_stage_testdata`, invoked as

```
dc3_stage_testdata --source /Volumes/seshat/data/diskmass/PPK_rdx --nspec 10
```

It copies a curated subset into `dc3/data/tests/`, trims the multi-spectrum images to `--nspec`
rows, and writes `dc3/data/tests/MANIFEST.toml` recording for every file: source path, source
mtime, size, SHA-256, the rows retained, and the staging date.

| Path | Contents | Git | Wheel |
|---|---|---|---|
| `dc3/data/tests/` | all fixtures | **committed** | **excluded** by `MANIFEST.in`, by file type |
| `dc3/data/tests/README.rst`, `MANIFEST.toml` | provenance and caveats | committed | **shipped** — not a stripped type |
| `dc3/tests/` | test code only | committed | shipped |

Two details lifted from PypeIt's `MANIFEST.in`:

```
recursive-exclude pypeit/data/tests *.gz *.fits *.npz
```

- **The exclusion is by file type, not by directory**, so the fixture README and the provenance
  manifest travel with the wheel while the bulk data does not. Keep that property: provenance
  should always be readable, even from an installed package with no data.
- PypeIt's `MANIFEST.in` carries a comment requiring it to stay in sync with the `defined_paths`
  registry in `pypeitdata.py`. The DC3 equivalent is `dc3/pkg/dc3data.py`, and this should be a
  **test**, not a comment — assert that every registered path is covered by an exclusion rule and
  vice versa.

`dc3/pkg/cache.py` fetches anything missing from the GitHub copy, so a clean checkout with no
RAID mounted still runs the full suite, and CI needs no special provision. Retain a
`requires_testdata` marker for genuinely offline runs, but it is no longer the primary mechanism:
the data is fetchable rather than optional. Mirror to a GitHub release or Zenodo once the port is
public, so the RAID is never needed again.

### What to stage

| Source (`/Volumes/seshat/data/diskmass/PPK_rdx/U06918/vdf_feb13/`) | Size | Role |
|---|---|---|
| `U06918_merge_c.fits` (352 × 2048) | 2.9 MB | galaxy spectra |
| `U06918_merge_c.err.fits` | 2.9 MB | error spectra |
| `HR6654.c_log.fits` (2048) | 11 KB | template |
| `HR6654.log.sigmaInst.fits` | 11 KB | template σ_inst |
| `U06918_merge.instrdisp_log.fits` (352 × 2047) | 2.9 MB | galaxy σ_inst |
| `dc3_express.par` | 1.3 KB | production parameters |
| `tplmask.db` | 72 B | region mask ([O III] λ5007, [N I] λλ5198, 5200) |
| `U06918_c4_an_od_ccstat.db`, `U06918_c4_an_od_vdf_fits.db` | 172 KB | **reference outputs**, `-a` |
| `U06918_c4_fn_od_ccstat.db`, `U06918_c4_fn_od_vdf_fits.db` | 172 KB | **reference outputs**, fitted `I` |
| `U06918_c4_an_od_gf1_t1_{xc,btxc,btsp,tpl_rmatch,xc_asym,btxc_asym}.fits` + `.gpm` | 50 MB | per-spectrum reference functions |

Trim to a **10-fiber subset** (≈ 250 KB of inputs, ≈ 1.5 MB of reference functions) as the
default CI fixture, and keep the two full `.db` tables (350 KB) for statistical comparison over
all 352 spectra. Skip the 93 MB PostScript QA files entirely.

Also copy `U00448/vdf_feb13/` (331 × 2048, and its σ_inst file *does* match the data in pixel
count) as a second galaxy at a different redshift (`-v 4870 70` vs `1140 70`), and because its
`_cd` runs completed — useful for exercising the instrumental-dispersion code path even though
its outputs are not trustworthy as references (report §16.4).

**What each fixture validates:**

- `tplmask.db` + `dc3_express.par` → the ParSet and mask-table readers, and the production
  defaults (`fititer=2`, `miter=-1`, `mvdiff=-0.10`, `citer=10`, `corder=4`, `DISP=CD1_1`).
- `_ccstat.db` → CC construction, mean subtraction (`MEAN ≈ 1e-16`), zero-lag trimming
  (`NU = 3888` of `nn = 4508`), peak metrics.
- `_xc.fits` → the correlation itself, including `nn = 4508` from
  `int(2.2·(int((ln−l0p)/dlp)+1))` and `v0 = −(nn/2)·dv = −24823.75 km/s`.
- `_tpl_rmatch.fits`, `_btsp.fits`, `_btxc.fits` → the forward model, stage by stage.
- `_vdf_fits.db` → fitted `V`, `σ`, errors, continuum coefficients, reduced χ², and the
  asymmetry indices `AX`/`AT`/`AC`/`RX`.
- The `an` vs `fn` pair → the analytic-normalization path against the free-parameter path.

**Agreement expectations.** Bit-identical reproduction is neither achievable nor meaningful. The
port changes the linear algebra (Cholesky vs. explicit inverse), the continuum solver
(`numpy.polynomial` QR vs. hand-rolled Cholesky normal equations), and the derivative scheme —
*and* the reference itself was written by an unidentified earlier build. So:

- **Statistical, not per-spectrum.** Compare distributions over the 352 spectra: median and
  scatter of `V` and `σ`, the `V`–`σ` correlation, the median reduced χ², the fraction with
  `FC = MC = 1`. Agreement at the few-percent level on these is a strong signal; per-spectrum
  differences are expected and are not failures.
- **Advisory, not gating.** These run as a reported comparison, not a CI assertion. A large
  disagreement opens an investigation into *which* stage differs — which is exactly what the
  per-stage reference products (`tpl_rmatch` → `btsp` → `btxc` → `xc`) are for.
- **Tight tolerances belong to the self-consistency and published tests**, not here: internal
  identities (K = 1 vs. multi-template, analytic vs. autodiff Jacobian, FFTW vs. numpy) can be
  held to machine precision, and the W11 §4 Monte Carlo criteria are stated in the paper and are
  build-independent.

**Caveats to record in `dc3/data/tests/README.rst`:**

- **Provenance:** written by an unidentified Feb-2013 build; the surviving source is 2014–2016.
  See report §16.0. Treat the outputs as indicative.
- These are **PPak** data with template HR 6654 — *not* the SparsePak/HD 167042 configuration of
  W11 §5. Do not present agreement here as reproducing the paper's Figures 7–9.
- The `_cd` (instrumental-dispersion) products are doubly untrustworthy — provenance plus the
  `-SG`/`-ST` argument bug (report §16.4). Stage them as inputs only; never as references.
- Every command in the campaign script passes `-v`, which triggers the v3.1 `fixed_window` bug
  (report §14.1). The archived outputs predate it, but this is why the port must not simply
  mirror the current C++ argument handling.

### ✅ The W11 §5 demonstration data has been located

`/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/` (78 MB, Nov 2008 – Jan 2009) is
the SparsePak run behind W11 §5 — Figures 7–9 and Table 1. Note the path: it is **not** under
`data/diskmass/`, which is why the earlier search missed it. The identification is solid:

| Evidence | |
|---|---|
| `HR6817_K1III_Mg_log.fits` | **HR 6817 = HD 167042**, the K1 III template named in W11 §5 |
| `U06918_merge.mg.{me,ms,msn}_smc_log.fits` | UGC 6918, **Mg** region, SparsePak merged spectra |
| `ppxf_jun10/U06918_merge.ppxf.db` | the **ppxf comparison run** (June 2010) of W11 §5.1 |
| `sircorrect_feb10/U06918_instrs_{01..18}.db` | the **18-step instrumental-dispersion workflow** of `siginst_algorithm.notes` (W11 Appendix A), one `.db` per step |
| `inp`, `inp_refit` | the keystroke scripts that drove the interactive `DC3` binary |
| `U06918_vdf.{gaufit,gaucovar,vdfit,anorm,anbtxc,cntcoeff,fitrms,comb}.db` | the per-stage outputs of the pre-`DC3_express` workflow |

`inp` confirms several inferences in the report independently: `tplmask.db` as the region-mask
table, a velocity guess of `1110 60` km/s, a window factor of `2.0`, and an explicit good-fiber
list (`2,5,9,16,19,21,22,24-37,...`) — i.e. the fiber selection was manual, not automatic.

**Why this matters more than the PPak fixtures.** These outputs can be checked against
*published* numbers rather than against an unidentified build. W11 Table 1 gives `V` and `σ`
for UGC 6918 fibers 52 and 55; the paper's Figures 7–9 are reproducible from this directory.
That is a **build-independent** acceptance test and therefore *can* carry tight tolerances,
unlike everything in the PPak set (§16.0 provenance problem). It also supplies the ppxf
comparison of §5.1 as archived output rather than something to be regenerated.

**Caveats.** This run used the **interactive `DC3` binary**, driven by `inp` — a much earlier
code path than `DC3_express`, and one of the eight executables that no longer compile. The
output layout is entirely different (many small `.db` files, per-stage subdirectories
`xc/`, `btxc/`, `btsp/`, `cont/`, `asymxc/`, `asymbtxc/`) and the report's file-format section
describes the `DC3_express` layout, not this one. So the *inputs* and the *published numbers*
are the valuable parts; the intermediate `.db` files need their formats worked out before they
can be used, and the provenance caution of §16.0 applies to them at least as strongly.

**Stage as a third fixture set**, via the same `dc3_stage_testdata --source` mechanism:

| File | Size | Role |
|---|---|---|
| `HR6817_K1III_Mg_log.fits` + `.err.fits` | 23 KB | the published template (HD 167042, K1 III) |
| `U06918_merge.mg.msn_smc_log.fits` | 647 KB | the galaxy spectra actually fit |
| `U06918_merge.msn.stat` | 7.5 KB | S/N statistics per fiber |
| `tplmask.db`, `inp` | ~1 KB | mask table and the run's full parameter record |
| `U06918_vdf_final.comb.db` | 7 KB | the final combined results — compare to W11 Table 1 |
| `ppxf_jun10/U06918_merge.ppxf.db` | — | the archived ppxf comparison, W11 §5.1 |

At ~700 KB total this is small enough to commit whole, and it is the **highest-value fixture
in the project** because it is the only one tied to published numbers. Skip the two ~11 MB
PostScript files and the per-stage subdirectories on the first pass.

**Also present:** `/Volumes/seshat/SPSPK_rdx/kinematics/mg/` contains many other galaxies, and
sibling directories under `/Volumes/seshat/SPSPK_rdx/` — a much larger SparsePak archive than
assumed. Not needed for the port, but worth knowing it exists.

---

## Verification

**The C++ cannot serve as a live oracle.** The binaries do not run (x86_64, missing dylibs),
and rebuilding requires cfitsio + fftw3 + wcslib + gsl + PGPLOT + Eigen + the Numerical
Recipes sources. Validation therefore rests on the archived outputs above plus the published
results, in that order of increasing weight:

| Reference | Strength | Can it gate CI? |
|---|---|---|
| PPak `vdf_feb13` outputs | weak — unidentified Feb-2013 build (§16.0) | no, advisory only |
| SparsePak `vdf_dec08` outputs | weak for the same reason, and an older code path | no, advisory only |
| **W11 published numbers** (Table 1, Figs. 4–9) | **strong — build-independent and citable** | **yes, with tight tolerances** |
| Internal identities (K=1, Jacobians, FFT backends) | strong — self-consistency | yes, to machine precision |

1. **Unit tests** (`dc3/tests/`, pytest, module-level functions per PypeIt convention):
   padding modes vs `numpy.pad`; apodization windows vs `scipy.signal.windows`; FFT
   correlation vs `scipy.signal.correlate`; block replication round-trip; Legendre continuum
   vs `numpy.polynomial`; covariance construction vs a direct O(N·W²) double loop from Eq. 4;
   `ConvolveFFTW` vs the numpy fallback; analytic Jacobian vs finite differences; JAX model vs
   SciPy model.
2. **Archived C++ artifacts** — `test_convolve.out` (125 KB) and `test_siginst{,_log}.fits`
   are committed reference outputs of the variable-σ convolution; use them as golden files for
   `dc3/core/resolution.py`.
3. **End-to-end run on the archived production data** — the Test data section above. Two
   distinct things, which must not be conflated:
   - *Gating:* the port runs to completion on real PPak inputs with the production parameter
     file and mask table, produces every expected product, and converges on a comparable
     fraction of spectra. This is a genuine CI test.
   - *Advisory:* a reported statistical comparison against the archived `_od` tables (see
     "Agreement expectations"). Not a CI assertion, because the reference build is unidentified
     (report §16.0). `_cd` products are excluded entirely.
4. **Reproduce the paper's Monte Carlo simulations** (§4) — the primary scientific acceptance
   test, and fully self-contained (synthetic spectra, no external data): 240 simulation sets
   over ⟨S/N⟩ ∈ {1, 1.4, 2, 2.8, 4, 8, 16, 64}, V_m ∈ {1000, 2500, 3600, 4160, 4640} km/s,
   σ_m ∈ {5, 10, 20, 40, 80, 110} km/s, 50 realizations each. Must recover the paper's stated
   results: systematic error in σ_f ≲ 10% for ⟨S/N⟩ ≳ 4; ≲ 20% random error in σ for
   ⟨S/N⟩ ≳ 3 and 20 ≤ σ ≤ 40 km/s; |V_f − V_m| < 15 km/s for ⟨S/N⟩ ≳ 2. Regenerate the
   paper's Figures 4 and 5 as a CI artifact.
5. **Reproduce the paper's UGC 6918 demonstration** (§5) — the **strongest gating test
   available**, because the reference is a *paper* rather than an unidentified build. Run the
   port on `U06918_merge.mg.msn_smc_log.fits` against `HR6817_K1III_Mg_log.fits` with the
   parameters recorded in `inp`. Three things to get right:

   - **Table 1 reports `σ_obs`, not `σ_LOS`.** W11 §5 says the fits "converge to measurements of
     `V_obs` and `σ_obs`", and the comparison is against ppxf, which is likewise uncorrected. So
     this is a **Phase 4 test and does not require Phase 7** — a useful ordering result, since
     the milestone is reachable before the instrumental-dispersion machinery exists.

     | Fiber | S/N (px⁻¹) | `V_f,DC3` | `σ_f,DC3` |
     |---|---|---|---|
     | 52 | 27.8 | 1125.3 ± 1.5 | 68.3 ± 1.6 |
     | 55 | 1.7 | 1076.4 ± 3.9 | 11.2 ± 5.7 |

   - **Calibrate the tolerance to the published precision.** Values are quoted to 0.1 km/s but
     carry errors of 1.5–5.7 km/s. A defensible gate is **~1 km/s** on fiber 52's `V` and `σ` —
     tight in absolute terms, nowhere near machine precision. Fiber 55 (S/N 1.7,
     `σ = 11.2 ± 5.7`) is essentially unconstrained and cannot gate anything; note also that its
     σ sits less than a factor of two above the block-replication floor W11 §4 quotes as
     `0.85 ΔV_p = 6.4` km/s.
   - **Prefer the Figure 9 population statistics as the primary gate.** They are stated
     numerically in the text — *all* velocity measurements within their errors, and **82% and
     94% of all spectra within one and two times the measurement errors** in σ — and they are a
     whole-field statistic over every fiber rather than two hand-picked numbers.
6. **Cross-check against ppxf**, as the paper's §4.1 and §5.1 did. Reproduce Figure 6 —
   agreement in systematic and random errors — and compare against the archived
   `ppxf_jun10/U06918_merge.ppxf.db` from the original comparison. `mangadap` shows how to
   drive ppxf comparably (`proc/ppxffit.py`). ppxf is a dependency (see "External
   dependencies"), so this needs no extra setup; **run it against ppxf ≥ 9.5.0**, and note
   that current ppxf will not reproduce the 2010 version's numbers exactly — the paper's
   figures are the reference, not the archived `.db`.
7. **Regression on K = 1** — the multi-template path with a single template must agree with
   the single-template path to machine precision.
8. **Resolution-matching cross-check** — `ppxf_util.varsmooth` against
   `mangadap.util.resolution.convolution_variable_sigma` on the same inputs, to a documented
   tolerance (Phase 2). Expect them to diverge where the kernel is undersampled, which is the
   whole reason `varsmooth` is the production path; the test should bound the agreement in the
   well-sampled regime rather than assert it everywhere.
9. **Preparation-pipeline characterization** (Phase 2) — push known-width Gaussians through
   Steps 1 and 2 and confirm the measured `σ_T'(λ)` matches the nominal vector to a documented
   tolerance, as a function of `velscale_ratio` and `varsmooth`'s `oversample`. Doubles as the
   regression test on template preparation, and quantifies an effect C23 §3.1 explicitly leaves
   unmeasured.
10. **The `varsmooth` floor** — request a sub-0.1-pixel kernel and assert the realised width, so
    an upstream change to `sig.clip(0.1)` fails loudly instead of biasing `dvar_inst`.
11. **`dvar_inst` sign handling** — synthetic cases with the template sharper, matched, and
    broader than the galaxy; confirm the sign, the warning on negative, the reported floor, and
    that `σ_*² = σ_obs² − dvar_inst` round-trips.
12. **QA reproducibility** — every figure generated inline during a fit must be identical to
    the same figure regenerated by `dc3_qa` from the results file alone (Phase 4).
13. **Packaging** — assert that every path registered in `dc3/pkg/dc3data.py` is covered by a
    `MANIFEST.in` rule and vice versa (Test data section).
14. **CI** — tox matrix over Python 3.11–3.14 × `{test, test-fftw, test-jax, test-mcmc}` on
    ubuntu, plus macos/windows; `pycodestyle --select=E9`; zero-warning Sphinx build. Include
    a job pinned to the minimum supported `ppxf` and one on the latest, since two upstream
    functions are on the critical path.

---

## Critical files

**To read when implementing** (the algorithm lives here, *not* in `dc3/`):
- `/Users/westfallold/Work/scripts/libraries/vdf.cpp` — the fitter; `fit()` 1473, `dofit()` 714,
  `mrqabc()` 883, `setsig2()` 399, `calc_norm()` 1132, `getbtsp()` 1007, `fit_continuum()` 561
- `/Users/westfallold/Work/scripts/libraries/correlation.cpp` — `correl_d`, `prep_fft`,
  `commonwave`, `adjmask`, `match_resolution`, `guessvdf`, `asymmetry`
- `/Users/westfallold/Work/scripts/libraries/cnvlv.cpp` + `headers/cnvlv.h` — `cnvlv_blk`,
  padding, `VariableSigmaGaussian`
- `/Users/westfallold/Work/scripts/dc3/DC3_express.cpp` — the pipeline spec
- `/Users/westfallold/Work/scripts/dc3/doc/descrip.txt`, `develop.txt` — the de-facto spec and
  the empirical justification for the hardwired constants

**To adapt from — PypeIt** (`/Users/westfall/Work/packages/pypeit`). ⚠️ **Read the branches,
not `release`:**
- **`funcpar_rebase`** (currently checked out) — **the source for both `ParSet` and `FuncPar`.**
  `parset_refactor` plus three additive commits adding `pypeit/par/funcpar.py` and
  `utils.get_func_kwargs`. Read `pypeit/par/{parset,funcpar}.py` and
  `pypeit/tests/test_pypeitpar.py` here.
- `parset_refactor` — the base of the above; read it only to see `ParSet` without `FuncPar`.
- ~~`funcpar_update`~~ — **superseded by `funcpar_rebase`.** Pre-`parset_refactor` merge base;
  do not read or diff it.
- `pypeit/{datamodel.py, pkg/*, core/bitmask.py, images/bitmaskarray.py, core/bspline.py,
  scripts/scriptbase.py}` and `pyproject.toml`/`tox.ini`/`doc/scripts/`

**To adapt from — other:**
- `/Users/westfall/Work/packages/nirvana/nirvana/models/beam.py` — `ConvolveFFTW`
- `/Users/westfall/Work/packages/mangadap/mangadap/{proc/templatelibrary.py, proc/ppxffit.py,
  proc/spectralfitting.py, proc/sasuke.py, util/sampling.py, util/resolution.py,
  util/pixelmask.py}` — `util/resolution.py::SpectralResolution.GaussianKernelDifference` is the
  reference for W19 Appendix A matching; `proc/sasuke.py` shows the quadrature-offset idiom.
  Note `util/covariance.py` is **not** needed; it is upstream as `astropy.nddata.Covariance`
- `/Users/westfall/Work/packages/mangadap/docs/papers/Overview/ms/rev1/res.py` — the script
  behind W19 Figure 34. Line 65 is Eq. 47 in code; lines 74–75 are the Eqs. 48/49 asymptotes;
  lines 14–19 reproduce the `1.43` quoted in §7.4.3. Use it as the reference implementation of
  `Φ(ξ, σ_*/σ_g)` when testing the Phase-2 resolution decision.

**Papers:**
- `/Users/westfall/Work/papers/dmIII/Westfall_etal_2011_ApJS_193_21.pdf` — W11, the algorithm;
  §4 for the Monte Carlo acceptance tests, §5 and Table 1 for the demonstration, Appendix A for
  the instrumental-dispersion treatment
- `/Users/westfallold/Work/literature/Westfall_etal_2019_AJ_158_231.pdf` — W19; §7.1.5, §7.4.3
  and Appendices A–B for the spectral-resolution treatment
- `/Users/westfallold/Work/literature/Cappellari_2017_MNRAS_466_798.pdf` — C17; §2.2 for the
  redshift-dependence of resolution matching, §4.3 and Eqs. 33–38 for the analytic LOSVD
  transform, Eq. 37 for the velocity-shift property
- `/Users/westfall/Work/Cappellari_2023_MNRAS_526_3273pdf.pdf` — C23; §2 and Eqs. 3–5 for the
  velocity/redshift relation, §3.1 and Algorithm 1 for `varsmooth` and its acknowledged
  interpolation limitation

**To depend on, never to copy:**
- `ppxf` ≥ 9.5.0 — `ppxf_util.{varsmooth, losvd_rfft, log_rebin, air_to_vac, vac_to_air}`.
  Proprietary license, redistribution prohibited. See "External dependencies".

**Reference data — external RAID, staging only.** Nothing in the package may read these paths at
run time; they are arguments to `dc3_stage_testdata`, used once.
- `/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/` — ✅ **the W11 §5 demonstration
  data** (SparsePak UGC 6918 + HR 6817 = HD 167042), including `ppxf_jun10/` (the §5.1 ppxf
  comparison) and `sircorrect_feb10/` (the Appendix A workflow). The highest-value fixture set.
- `/Volumes/seshat/data/diskmass/PPK_rdx/feb13_dc3_inst_0*.scr` — production driver scripts
- `/Volumes/seshat/data/diskmass/PPK_rdx/{U06918,U00448}/vdf_feb13/` — PPak test-data source

**To create:** the `dc3/` package tree described above. ✅ `claude/dc3-original-implementation.md`
is complete.

---

## Change log

- **2026-09-15** — Initial plan: nine phases, decisions on port scope, multi-template
  formulation, optimizer backend, and interface; workflow-integration notes; five open science
  questions.
- **2026-09-15** — Incorporated the cross-check of the February 2013 PPak production campaign
  (report §16). Added a **Test data** section specifying which files to copy from
  `/Volumes/seshat/.../U06918/vdf_feb13/` and `U00448/vdf_feb13/`, what each fixture validates,
  numeric agreement targets, and the caveats to record in the fixture README. Added the measured
  single-threaded performance baseline to Phase 5 (8.1 s/spectrum wall clock; `-a` is ~25% faster
  than fitting the normalization; ~41% of wall time is outside the fit), which changes two
  decisions: make `-a` the default, and give QA-figure generation its own budget and an opt-out.
  Corrected Phase 3: there is **no** power-of-two constraint to drop — the C++ already uses a
  2.2× even length and real-to-complex FFTW plans; the available gain is `FFTW_MEASURE` plus
  persisted wisdom instead of `FFTW_ESTIMATE`. Promoted regression against the archived
  production outputs to verification item 3 and renumbered the rest. Narrowed the SparsePak
  open item to a specific pair of directories to check.
- **2026-09-15** — Revised the Test data section for two constraints raised on review.
  *Provenance:* the archived outputs were written by an unidentified Feb-2013 build, not the code
  now on disk (report §16.0), so fixtures are now split into version-independent **inputs**
  (trustworthy) and **outputs** (indicative). Replaced the per-spectrum tolerances (`V` within
  0.1 km/s, `σ` within 1%) with a statistical, advisory comparison; verification item 3 is now
  explicitly split into a gating end-to-end run and a non-gating numerical comparison. Tight
  tolerances are reserved for self-consistency identities and the build-independent W11 §4
  criteria. Added the same caveat to the Phase 5 performance baseline, noting the ratios are more
  robust than the absolute times. *External RAID:* added a `dc3_stage_testdata` staging script
  with a `MANIFEST.toml` recording source path, mtime, size and SHA-256 for every file; a
  committed ~600 KB subset under `dc3/tests/data/`; a gitignored full-size set under
  `dc3/data/tests/`; a `requires_testdata` pytest marker so the suite runs unmounted; and a rule
  that no package code may read `/Volumes/seshat` at run time.
- **2026-09-15** — Incorporated the `Comments` section of `notes` (first full read of both
  documents). Substantive changes throughout:
  *Scope.* Added **backwards compatibility is explicitly out** as a stated decision, with its
  consequences (output datamodel redesigned from scratch; CLI designed rather than
  transcribed; old formats survive only as test-fixture ingest paths).
  *Dependencies.* Added an **External dependencies** section making `ppxf` a hard dependency
  for `ppxf_util.varsmooth` (variable-σ resolution matching) and `ppxf_util.losvd_rfft`
  (analytic Gaussian/Gauss–Hermite LOSVD transform), which removes most of `fitfunc.cpp` and
  `cnvlv_integral` from the port. ⚠️ Flagged that **ppxf's license is proprietary and forbids
  redistribution** — import-only, never vendor, and expect downstream packaging friction;
  imports localized to two modules. Test against **ppxf ≥ 9.5.0**.
  *Phase 1.* Rewrote for the PypeIt `parset_refactor` and `funcpar_update` branches (with the
  rebase warning), a **pydantic v2** option for both `ParSet` and the datamodel with a
  feature-by-feature comparison and a prototype-first decision rule, **`tomllib`/TOML instead
  of ConfigObj** with a strict 1:1 file-key-to-parameter rule, `DataContainer` **split into
  I/O infrastructure and datamodel enforcement** rather than adopted wholesale, and
  `BitMaskArray` alongside `BitMask`.
  *Phase 2.* Added the **`specutils`-at-the-boundary / bespoke-internals** rule with its
  rationale; `mangadap.util.sampling.Resample` for resampling; **`TemplateLibrary` adopted
  nearly wholesale** from mangadap with a possible `specutils` upstream path; a table of
  mangadap spectrum/template **preparation and validation** code to repurpose; a new
  **de-redshifting** subsection to be evaluated on evidence; and `ppxf_util.varsmooth` for
  resolution matching with mangadap's implementation retained as a cross-check.
  *Phase 3.* `ppxf_util.losvd_rfft` for the LOSVD transform, plus an explicit **seam for a
  future empirical instrumental LSF** marked low priority and deferred.
  *Phase 4.* Tier 1 becomes **user-selectable `lsq` / `de` / `mcmc` backends** over one
  objective; **uncertainties** get three documented routes with the analytic precision-matrix
  route as default and a separate `dc3_mcerr` script; continuum gains a **`pypeit.core.bspline`
  option**; the **output datamodel is redesigned from scratch** (FITS only, one file per run,
  self-describing, no ASCII `.db`); **QA plots become tiered** (`none`/`summary`/`standard`/
  `full`) with a **`dc3_qa` post-processing script that can regenerate any figure from inputs
  plus results**, and the datamodel constraint that implies; added a scripts table including
  **`dc3_xcorr`** for building and inspecting cross-correlations and asymmetry metrics before
  fitting, flagged as the natural Phase 2–3 milestone.
  *Phase 5.* Extended multiprocessing to `dc3_mcerr`/`dc3_xcorr`; added `QAPar.level=none` as a
  fifth lever.
  *Phase 8.* Docs use **`numpydoc`** with validation in CI; the "differences" page is reframed
  as **"Differences from Westfall et al. (2011)"** — algorithmic departures from the *paper*,
  since the C++ was never distributed; C++-level records stay internal under `claude/`.
  *Science questions.* All five marked resolved with reasoning and implementation
  consequences: asymmetry stays diagnostic but gains a per-iteration tracking mode; errors as
  above; fit-window width to be re-derived; **spectral covariance continues to be ignored**,
  with `(δG)²` kept as an explicit object and a note that `Covariance` is now upstream as
  **`astropy.nddata.Covariance`** should I/O infrastructure ever be needed; Gauss–Hermite
  capability retained but not default.
  *Test data.* ✅ **Located the W11 §5 demonstration dataset** at
  `/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/` — not under `data/diskmass/`,
  which is why earlier searches missed it. Identification confirmed by `HR6817_K1III_Mg_log.fits`
  (HR 6817 = HD 167042, the paper's K1 III template), the `ppxf_jun10/` comparison run of
  W11 §5.1, and `sircorrect_feb10/`'s eighteen `U06918_instrs_NN.db` files matching the
  18-step Appendix A workflow. Added it as a third fixture set (~700 KB, committable whole)
  and noted that its `inp` keystroke script independently confirms `tplmask.db`, the `1110 60`
  velocity guess, the 2.0 window factor, and a manual fiber selection.
  *Verification.* Reordered by reference strength in a new table; **reproducing W11 §5 is now
  a gating test with tight tolerances** (published numbers are build-independent), promoted
  above the ppxf cross-check; added resolution-matching and QA-reproducibility items and a
  minimum/latest `ppxf` CI axis.
- **2026-09-15** — Incorporated **Westfall et al. (2019, AJ 158, 231)** — the MaNGA DAP
  overview paper, tied to `mangadap` v2.2.1 — now referenced throughout as W19, with a paper
  reference table added to the header. The substantive change is a new Phase 2 subsection,
  **"Do not match resolution by default — carry the offset and correct afterwards"**, adopting
  W19 §7.1.5/§7.4.3's approach: leave the template at its native resolution, fit `σ_obs`, and
  recover `σ_*² = σ_obs² − δσ_inst²` afterwards. `ResolutionPar.match` ∈
  `{'none','offset','full'}` defaults to `'none'`. Recorded W19's four arguments (matching
  loses no information in the Gaussian limit; a Doppler-induced offset is unavoidable anyway;
  Appendix B's `Φ` shows sharper templates monotonically reduce `ε[σ_obs]/σ_obs`; and — W19's
  stated main driver — a pedestal keeps `σ_obs` away from the `σ → 0` boundary where its
  posterior turns inverse-gamma). Noted that argument 4 binds harder for DC3 than for the DAP,
  because block replication imposes a **hard** floor at `0.85·dv/8` rather than a soft
  boundary, right in the DMS `σ_* ≲ σ_g` regime. Identified that **DC3 already has the
  machinery**: `res_base`/`MIN_SIG_RESMATCH` is structurally the same pedestal scheme, built
  as a 1 km/s numerical fudge, and needs only to be promoted to a wavelength-dependent
  `δσ_inst(λ)`. Specified that DC3 should implement W19 Appendix A's **third** option
  (wavelength-dependent resolution difference), which mangadap explicitly does not. Flagged
  that W19 Eq. 7's flat-average `δσ_inst` is documented as biased low by a few percent, and
  that DC3 can improve on it two ways — a `T`-weighted average using the weighting the Statler
  covariance already carries, and W11 Appendix A's **fitted** estimator. Consequently
  **promoted Phase 7** from a post-processing add-on to the critical path, restructuring it
  around the two estimators. Added the Gauss–Hermite constraint (van der Marel & Franx requires
  matched resolution, so `h3`/`h4` must force `match='full'` — a validation error) to science
  question 5, and added **science question 6** recording the resolution decision. Added a
  verification plan using the W11 §4 Monte Carlo suite, including an empirical test of
  Appendix B's `Φ` that appears never to have been done. Noted in Phase 4 that the σ clamps now
  apply to `σ_obs` rather than `σ_*`, and that the boundary-pinned fraction should be reported
  as a run-level statistic.
  ⚠️ Recorded a **probable algebra error in W19 Appendix B, Eq. 50**: the ratio of Eqs. 48 and
  49 is `√2(1 + σ_g²/σ_*²) = √2 + √2·σ_g²/σ_*²`, but Eq. 50 prints `√2 + σ_g²/σ_*²`, dropping
  a `√2` from the second term. Confirmed against the figure script
  `mangadap/docs/papers/Overview/ms/rev1/res.py`, whose commented-out line 7 carries the same
  expression — so it is an algebra slip, not typesetting. Eqs. 42–47 survive independent
  re-derivation, Figure 34 is correct (script line 65 implements Eq. 47 exactly), and the
  `1.43` quoted in §7.4.3 follows from Eq. 47 rather than Eq. 50 and reproduces exactly. The
  conclusion is unaffected and slightly strengthened: at `σ_* = σ_g` the corrected ratio is
  2.83 rather than 2.41. Added `res.py` to Critical files as the reference implementation of
  `Φ`, and both papers to a new Papers subsection there.
- **2026-09-16** — Applied the revisions agreed in
  [`plan-reassessment.md`](plan-reassessment.md), following a full review of both documents and
  three rounds of follow-up. **Phase 2's spectral-resolution treatment is rewritten end to end.**
  The previous "do not match resolution by default" design rested on a misreading of the C++:
  `res_base` is not a W19-style pedestal but its *negative* counterpart — it makes the template
  uniformly broader, cancels inside `getbtsp`, and floors the measurable σ at values that reached
  9.50 km/s in production. The replacement prepares templates **once per run** in two steps
  (resolution matching to a fiducial galaxy resolution, then resampling at an integer
  `velscale_ratio`), with a signed instrumental variance **`dvar_inst`** that may be of either
  sign — positive holds `σ_obs` off the zero boundary as W19 argues, negative is what a
  same-instrument survey gets and imposes a floor that must be warned about. Four parameters
  (`velscale_ratio`, `epsilon_sigma`, `sigma_floor`, `mask_unmatched_sres`) replace
  `ResolutionPar.match`; `epsilon_sigma` is a **two-sided** target on the minimum kernel σ, a
  deliberate superset of `GaussianKernelDifference`'s behaviour, and defaults to 0.1 px to match
  a clip inside `varsmooth` that would otherwise bias `dvar_inst`. Added the three governing
  rules (the galaxy is never altered; matching is preparation, never cost function; no
  deconvolution), the **pre-pixelized `sres` input contract**, and a Phase 2 deliverable
  characterizing the preparation pipeline empirically — measuring `σ_T'(λ)` by pushing
  known-width Gaussians through Steps 1 and 2 — rather than modelling pixelization analytically.
  **Phase 7 shrinks accordingly** to the W11 Appendix A *fitted* estimator, used to validate the
  chosen `dvar_inst`; W19 Eq. 7's biased flat average is not needed. De-redshifting is now an
  exact **integer pixel shift** with the offset propagated analytically, replacing the earlier
  resampling framing. Phase 3 gains the **Fourier phase shift** (C17 Eq. 37) in place of
  `shift_lininterp`, a clean-room protocol for the JAX `losvd_ft`, and an `exact_cpp_length` flag
  so the `_xc.fits` fixture comparison survives `next_fast_len`. Phase 4 gains the **two-level
  fit API** (outer over many spectra, inner over one — the multiprocessing boundary), the
  **constraints table** (fixed-width ASCII, `ID`/`FIT`/`V`/`V_FIX`/`SIG`/`SIG_FIX`) replacing the
  C++'s automatic σ ≡ 0 refit, no tier-2 re-matching, and `σ_obs`/`dvar_inst`/`σ_*` plus both σ
  floors as reported columns. Corrected the Phase 5 analytic-derivative estimate from 5–7× to
  **~2×**, since real-space censoring costs ~3 FFTs per parameter. Phase 6 records the two
  conditions its linearity depends on. **Test data** moves to `dc3/data/tests/` on PypeIt's
  pattern — committed, excluded from the wheel by file type so provenance still ships, fetched
  through the cache — and there is no `dc3/tests/data/`. **Verification** is recalibrated: W11
  Table 1 reports `σ_obs`, so reproducing §5 is a Phase 4 test needing ~1 km/s rather than "tight"
  tolerances, with Figure 9's population statistics as the primary gate; four new items cover the
  preparation pipeline, the `varsmooth` floor, `dvar_inst` sign handling, and packaging. Added
  C17 and C23 as references, a differences-from-W11 table and an assumptions page to Phase 8,
  science questions 6 and 7, and three out-of-scope entries. Dropped the line-count estimate.
- **2026-09-16** — Read Cappellari (2023) §3.1 and Algorithm 1 directly, which corrected and
  sharpened three things in the Phase 2 treatment written earlier today. **The `varsmooth` 0.1-px
  clip is about the coordinate stretch, not the convolution.** Algorithm 1 works by stretching
  the wavelength coordinate by `σ_max/σ` so the variable-σ kernel becomes constant; that stretch
  diverges as `σ → 0`, so the clip bounds the number of resampled points. The clip is **not** in
  the published algorithm — an undocumented implementation detail, which is why the port asserts
  it behaviourally. **Both preparation steps degrade the effective template resolution, not just
  Step 2**: `varsmooth` interpolates twice (in to the stretched grid, out again), and C23 §3.1
  states plainly that interpolation is itself a convolution, that removing it would require a
  Bayesian treatment because the spectra are noisy, that the input spectra already carry prior
  interpolation, and that quantifying all this is "beyond the scope of this paper". That both
  vindicates the decision not to model pixelization analytically and is the strongest argument
  for the empirical characterization, which now measures the composite of Steps 1 and 2 and
  varies `varsmooth`'s `oversample` as well as `velscale_ratio`. **Separated the two oversampling
  knobs**, which the earlier text conflated: `varsmooth(oversample=m)` acts on the internal
  stretched grid and reduces interpolation error inside the matching convolution, while
  `velscale_ratio` acts on the output grid and keeps the prepared LSF Nyquist-sampled. Added that
  `varsmooth` converts σ to pixels with a centred finite-difference gradient, exact only on a
  uniformly-sampled coordinate — so pass it log λ. Added a second reason Step 1 is approximate,
  from C17 §2.2: instrumental resolution is redshift-independent in velocity units but the pixel
  *wavelength* changes, so matching is redshift-dependent and the fiducial is a fiducial
  resolution **and** redshift. Stated the resulting trade honestly — the C++ re-matches per
  tier-2 iteration partly for this reason, and moving preparation out of the loop sacrifices
  per-spectrum redshift registration in exchange for speed and a known `dvar_inst`, with the
  Phase 7 fitted estimator as the check. Finally, de-redshifting now specifies that the `sres`
  vector is re-indexed by the same integer shift, and cites C23 §2 Eqs. 3–5 for the
  velocity/redshift relation.
- **2026-09-16** — Consistency pass before beginning Phase 1. Fixed a genuine contradiction in
  the Phase 4 forward model, which still described `getbtsp` as convolving at
  `sqrt(σ_fit² − σ_base²)` and applying the velocity shift by interpolation — both superseded by
  the Phase 2 and Phase 3 rewrites earlier today. Removed a forward reference from the
  de-redshifting section to a rule stated later, repaired a paragraph left dangling by an earlier
  edit, corrected `dc3_xcorr`'s description (it runs template *preparation*, not per-spectrum
  resolution matching), and restored two missing section rules. Added
  [`dc3-python-implementation.md`](dc3-python-implementation.md) to the companion-document list.
- **2026-09-16** — **`funcpar_rebase` is now the single source for `ParSet` and `FuncPar`.** The
  `FuncPar` work has been rebased onto `parset_refactor` as three additive commits
  (`17fc59fce`, `1095cd100`, `a018658a8`), so Phase 1 now reads one linear history instead of two
  branches with an unusable diff; the rebase warning is withdrawn and `funcpar_update` is marked
  superseded and not to be read. Documented what `FuncPar` actually does — a `ParSet` subclass
  whose parameters are derived from a function signature at class-creation time via
  `__init_subclass__` (PEP 487, not a metaclass), with `dtype` inferred from the wrapped
  function's type annotations through `typing.get_type_hints`.
  **Added the governing rule for when to use it:** `FuncPar` wraps *third-party* functions only;
  anything internal to `dc3` gets a hand-written `ParSet` subclass. Its lack of `options`,
  `descr` and reliable `dtype` is **deliberate** — for a third-party function all of that is the
  upstream package's documentation, and `dc3` depends on it rather than reproducing it, because a
  reproduced copy is guaranteed to drift. What `dc3` does own in that case is the *restriction*,
  via `kw_subset`/`omitted_keys`. Proposed a corresponding addition to `FuncPar` — **a class
  attribute pointing at the authoritative upstream documentation**, as a Sphinx cross-reference
  resolved through `intersphinx` where the dependency publishes an object inventory and a plain
  URL where it does not (ppxf being the case that forces the fallback), rendered by `to_rst_table`
  and the reflection-based doc generator; offer this upstream rather than carrying a divergent
  copy. Noted that this arrangement satisfies the 1:1 config-key rule in its strictest form (the
  key *is* the upstream keyword), that `_valid_default_kwargs` makes upstream renames fail loudly
  at import rather than silently, that Phase 2's four `TemplatePar` parameters are `dc3`'s
  vocabulary and must therefore be declared by hand, and that `FuncPar`'s existence is evidence
  against the pydantic option in the Phase 1 prototype.
