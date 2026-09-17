# DC3 — The Original C++ Implementation

A survey of the state, structure, dependencies, and control flow of the `DC3` code base as of
its last modification (2016), prepared as background for a Python port.

- **Source root:** `/Users/westfallold/Work/scripts/dc3`
- **Shared header:** `/Users/westfallold/Work/scripts/headers/DC3.h`
- **Support libraries:** `/Users/westfallold/Work/scripts/libraries`, `.../headers`
- **Numerical Recipes:** `/Users/westfallold/Work/scripts/nr3`, `.../libraries/nr_2.11.cpp`
- **Parameter templates:** `/Users/westfallold/Work/scripts/progfiles`
- **Reference paper:** Westfall, Bershady & Verheijen (2011), ApJS 193, 21 — *The DiskMass
  Survey. III. Stellar Kinematics via Cross-Correlation*. Cited below as **W11**.

Author: Kyle Westfall. Originally implemented as `SVANAL` (2002–2004), renamed `SCKAT`, and
renamed to `DC3` on 20 September 2012 (`dc3/notes` records the file-by-file rename).

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Scientific background](#2-scientific-background)
3. [Build system and compilation](#3-build-system-and-compilation)
4. [Where the algorithm actually lives](#4-where-the-algorithm-actually-lives)
5. [The 19 executables](#5-the-19-executables)
6. [`DC3_express` — control flow in detail](#6-dc3_express--control-flow-in-detail)
7. [The interactive `DC3` program](#7-the-interactive-dc3-program)
8. [The algorithmic core](#8-the-algorithmic-core)
9. [Module-by-module inventory](#9-module-by-module-inventory)
10. [Dependencies](#10-dependencies)
11. [File formats](#11-file-formats)
12. [`DC3.h`](#12-dc3h)
13. [Documentation digest](#13-documentation-digest)
14. [Defects and known issues](#14-defects-and-known-issues)
15. [Implications for the port](#15-implications-for-the-port)
16. [Cross-check against a production run](#16-cross-check-against-a-production-run)
17. [Change log](#17-change-log)

---

## 1. Executive summary

`DC3` implements the cross-correlation (CC) method for extracting stellar line-of-sight
kinematics (velocity `V`, velocity dispersion `σ`) from galaxy-continuum spectra, as described
in W11. It was the production software for the DiskMass Survey and was validated against
`pPXF` (Cappellari & Emsellem 2004) in W11 §4.1 and §5.1.

Five facts dominate any assessment of its current state.

**1. Only 11 of 19 executables compile.** The `makefile` carries the comment

```
# OUT OF DATE DUE TO NEW SPECTRUM, CORRELATION, and VDF CLASSES
```

above a commented-out block containing `DC3` (the main interactive program), `app_siginst`,
`combspec`, `dither_stack`, `prob_stack`, `recoverv`, `stack`, and `stack_sim`. This is
literally accurate: those targets require ten `obj/DC3_*.o` files that cannot be built,
because the corresponding sources call methods — `Correlation::setac(...)`, the old
`Correlation(...)` constructors, `VDF::refit(...)`, `VDF::getbtsp(Spectrum)`,
`VDF::vdfobj(...)`, `VDF::cnt()`, `VDF::adjmask()` — that were removed, made `protected`, or
had their signatures changed during the 2012–2014 library rewrite. `obj/` contains exactly the
11 buildable objects and nothing else.

**2. `DC3_express` is the only working end-to-end driver.** It is the reference implementation
for a port.

**3. The prebuilt binaries do not run.** `scripts/bin/DC3_express` (June 2015) is x86_64 Mach-O
linked against `/usr/local/lib/libcfitsio.2.dylib`, `libfftw3.3`, `libgsl.0`, `libcpgplot`,
`libpgplot`, `libhdf5.9` — none of which are present. There is no live oracle for
bit-comparison.

**4. The algorithm no longer lives in `dc3/`.** Between 2006 and 2013 the numerical core
migrated into three C++ classes in the shared library — `Spectrum`, `Correlation`, and `VDF`.
The `DC3_*.cpp` files are now a driver plus a frozen historical implementation.

**5. The Numerical Recipes dependency is the licensing blocker,** and it is shallower than it
looks. DC3's actual NR usage is a few dozen routines (LM fitting, sorting, RNGs, FFT helpers,
linear algebra) plus a pervasive custom `Vector`/`Matrix` type system in `headers/vec.h`
(~10,000 token occurrences across the code that would be touched). Every one of these has a
direct NumPy/SciPy equivalent.

### Size

| Scope | Lines |
|---|---|
| `dc3/*.cpp` | 28,319 |
| `libraries/*.cpp` linked by DC3 | ~36,700 |
| `headers/*.h` in the dependency closure | ~7,100 |
| **Total** | **~72,000** |

Of this, roughly 20,000 lines are infrastructure that NumPy/SciPy/Astropy replace outright
(custom containers, PGPLOT wrappers, cfitsio wrappers, a hand-rolled WCS, a terminal menu
system, string utilities, an FFTW wrapper, GSL interpolation wrappers), and roughly 11,000
lines contain genuine DC3 algorithm content.

---

## 2. Scientific background

The method, from W11 §2.

The observed galaxy spectrum is `G = I ⊗ F` (W11 Eq. 1), where `I` is the ideal template and
`F` the line-of-sight velocity distribution. In practice `I → T` (an observed stellar template)
and `F → B` (a parameterized broadening function, here a Gaussian with amplitude `a`, velocity
`V`, and dispersion `σ`).

Correlating both sides of Eq. 1 with `T` gives

```
G ∘ T ≈ (T ⊗ B) ∘ T          (W11 Eq. 2)
X     ≈ X_T
```

The literature CC method instead uses the commutation

```
G ∘ T ≈ (T ∘ T) ⊗ B          (W11 Eq. 3)
X     ≈ A_B
```

which is cheaper (one auto-correlation, computed once). **W11's central result is that Eqs. 2
and 3 are not equivalent for real data.** Real spectra are *detector-censored*: truncated by
the finite observed spectral range (OSR) and punched through by masks. Applying the censoring
after convolution (Eq. 2) is not the same as applying it before (Eq. 3). W11 Figure 1
demonstrates this and shows the error incurred by the Eq. 3 route is of order **10% in σ**,
with a smaller but non-zero error in `V`.

DC3 therefore takes the expensive route deliberately: it forms the broadened template
`T_B = T ⊗ B`, **censors and masks it in exactly the same way as `G`**, and only then
correlates with `T`. This requires one convolution and one correlation per fit iteration
rather than a single precomputed auto-correlation, and is the origin of the code's speed
problem.

Supporting machinery in W11:

- **§2.3 Masking.** Nebular emission, sky-subtraction residuals, cosmic rays, and telluric
  features are masked. Because `T_B` is treated identically to `G`, masking is as
  straightforward as in direct-pixel fitting. Masks defined in one frame are Doppler-shifted
  into the other and **broadened** to account for the fitted `σ`.
- **§2.4 FFT preparation.** Subtract the mean pixel value (over unmasked pixels only);
  zero-pad to a power of two and further to avoid convolution aliasing; deliberately **do not**
  apodize (W11 argues the symmetric treatment of `G` and `T_B` makes `B` robust without it).
- **§3 Fitting.** χ² in CC space using the Statler (1995) covariance:

  ```
  (δX_{j,k})² = Σ_n T_{n−j} T_{n−k} (δG_n)²                (W11 Eq. 4)
  χ²_X = [X − X_T] [(δX)²]⁻¹ [X − X_T]ᵀ                    (W11 Eq. 5)
  ```

  evaluated over a window centred on the CC peak, of width ≈ 2 × the CC-peak FWHM (empirically
  optimal, W11 §3).
- **Three iteration tiers** (W11 §3):
  - *Tier 1* — Levenberg–Marquardt minimization of χ²_X, with restarts from randomized guesses
    (characteristic scales 1, 100 km/s, 100 km/s for `a`, `V`, `σ`) to confirm a global minimum.
  - *Tier 2* — update the mask transcription using the fitted `V` and `σ`, then restart tier 1.
  - *Tier 3* — fit a low-order Legendre polynomial to the continuum difference `G − T_B` and
    restart tier 2. Both tiers 2 and 3 are hard-limited to ten iterations.
- **Sub-Nyquist broadening.** Discrete convolution is systematically in error when `σ` is
  under-sampled. W11 requires `σ ≥ 0.85 px` (Gaussian FWHM ≥ 2 px); below this the template is
  **block-replicated** — each pixel divided into `N` sub-pixels of the same value — convolved,
  then block-averaged back. `maxblk = 8`, so the effective floor is `0.85·dv/8 ≈ 0.106 px`;
  below that `σ` is set to exactly zero.
- **Appendix A — instrumental broadening.** `σ_LOS = (σ_obs² − δσ_inst²)^{1/2}` (Eq. A2), with
  `δσ_inst² = σ_B² − σ_obs² − σ_off²` (Eq. A8), where `σ_off` is a constant offset introduced to
  keep the resolution-matching kernel numerically stable.

---

## 3. Build system and compilation

### 3.1 Structure

`make` is invoked in `dc3/`, with a four-file include chain:

```
dc3/makefile
  ├── ../make.directory_defs      # scripts=..  libdir, bindir, hdrdir, nr3dir, gpcdir
  ├── $(libdir)/makefile          # rebuild libraries/ if needed
  ├── $(nr3dir)/makefile          # rebuild nr3/ if needed
  ├── $(gpcdir)/makefile          # rebuild gpc232/ if needed
  └── $(scripts)/make.binary_defs
        ├── ../make.linkage_defs  # -L/-l flags
        ├── -include $(src:%.cpp=obj/%.d)    # auto dependency files
        └── -include $(exe:%=lnk/%.lnk)      # per-executable object lists
```

`CC = g++`; compile flag `opt = -O`. Objects go to `dc3/obj`, binaries to `scripts/bin`.
Targets: `make all`, `make clean`, `make proglist`.

### 3.2 Linkage

`/Users/westfallold/Work/scripts/make.linkage_defs` requires eight environment variables —
`CFITSIO_DIR`, `PGPLOT_DIR`, `EIGEN_DIR`, `WCSLIB_DIR`, `FFTW3_DIR`, `GSL_DIR`, `HDF5_DIR`,
`MPI_DIR` — and defines:

```make
linkfit  = -L$(CFITSIO_DIR)/lib -lcfitsio
linkfft  = -L$(FFTW3_DIR)/lib   -lfftw3
linkwcs  = -L$(WCSLIB_DIR)/lib  -lwcs
linkgsl  = -L$(GSL_DIR)/lib     -lgsl -lgslcblas
linkcpg  = -L$(PGPLOT_DIR)/lib  -lcpgplot -lpgplot
linkhdf  = -L$(HDF5_DIR)/lib    -lhdf5
linkmpi  = -L$(MPI_DIR)/lib     -lmpi
linkmath = -lm
```

Every binary is linked with *all* of these, so the link line substantially over-declares
(see §10). Per-executable object lists are the real dependency declaration and live in
`lnk/*.lnk` as plain make rules.

### 3.3 Installation and use

Per `doc/README.install` (last updated 2007, and already out of date — it still refers to
`svanal` and `g2c`): edit the makefiles for the local CFITSIO/PGPLOT/X11 paths, `make all`,
put `scripts/bin` on `PATH`, and run the binary.

### 3.4 Current build state

| State | Executables |
|---|---|
| **Builds** (11) | `asym_renorm`, `DC3_express`, `DC3_setflags`, `extperr`, `fitrms`, `mkasym`, `redoasym`, `stack_pfile`, `test_int`, `test_convolve`, `test_spec` |
| **Does not build** (8) | `DC3`, `app_siginst`, `combspec`, `dither_stack`, `prob_stack`, `recoverv`, `stack`, `stack_sim` |

The `makefile`'s `src` list is itself stale: it names `DC3_specmanip.cpp` (moved to
`libraries/specmanip.cpp`), `instrconv.cpp` (lives in `../spakrdx/`), and misspells
`test_convolve.cpp` as `test_conolve.cpp`.

Eight source files in `dc3/` are referenced by neither the `src` list nor any `.lnk` file:
`DC3_chimap.cpp`, `DC3_contfit.cpp`, `DC3_dotprod.cpp`, `DC3_err.cpp`, `DC3_filt.cpp`,
`DC3_fitbroad.cpp`, `DC3_fiteval.cpp`, `DC3_fitfunc.cpp`. These are frozen pre-class (v1/v2)
code, retained as documentation of the original algorithm.

---

## 4. Where the algorithm actually lives

This is the single most important structural fact for a port.

| Class | Header | Implementation | Lines | Role |
|---|---|---|---|---|
| `Spectrum` | `headers/spectrum.h` | `libraries/spectrum.cpp` | 4,487 | Log-λ sampled spectrum with error, mask, continuum, and instrumental-resolution vectors |
| `Correlation`, `AutoCorrelation`, `CrossCorrelation` | `headers/correlation.h` | `libraries/correlation.cpp` | 2,600 | FFT correlation, censoring, masking, resolution matching, peak metrics |
| `VDF` | `headers/vdf.h` | `libraries/vdf.cpp` | 2,623 | The three-tier fitter, Statler covariance, LM minimization |

with supporting library modules `cnvlv` (convolution + block replication), `contfit`
(Legendre/Chebyshev continuum), `fft` (FFTW wrapper `FFT_1D`), `fitfunc` (Gaussian /
Gauss–Hermite), `apodize`, `specmanip`, `param`, `eigen`.

The `dc3/` directory now contains:

- a working driver (`DC3_express.cpp`),
- a stale interactive program (`DC3_main/menu/plot/userio/fileio/par/fft/win/util/monte/stack`),
- a frozen historical implementation of the same algorithm (`DC3_fitbroad`, `DC3_err`,
  `DC3_fiteval`, `DC3_chimap`, `DC3_contfit`, `DC3_dotprod`, `DC3_fitfunc`, `DC3_filt`),
- and a set of small post-processing utilities.

`DC3_fitbroad.cpp` (3,001 lines) is the direct ancestor of `libraries/vdf.cpp` and is the code
actually described in W11 §3. It is valuable reading — it contains three generations of the
fitter side by side (amoeba, LM, and masked-LM) and its edit log records the design decisions.

---

## 5. The 19 executables

### 5.1 Active

#### `DC3_express` — the production driver ✅

`DC3_express.cpp`, 2,093 lines, `main` at line 147. Detailed in §6.

Replaces the interactive program's menu sequence with a single batch invocation. Version
history: v1.0 (8 Jan 2013) → v2.0 (VDF 3.0) → v2.1 (VDF 3.2) → v3.0 (instrumental broadening,
1 Feb 2013) → v3.1 (fixed velocity window, 5 Mar 2013). Prior versions in
`oldversions/DC3_express_v{1.0,2.0,2.1,3.0}.cpp`.

Links (`lnk/DC3_express.lnk`): `myfuncs, nr_2.11, param, str_manip, stat, term_messages,
stopwatch, apodize, cnvlv, contfit, fitfunc, fitsfuncs, mywcs, eigen, myplot, correlation,
spectrum, fft, spline, vdf`. It does **not** link `autocorrelation.o` or `montecarlo.o` — no
explicit AC products and no Monte Carlo in express mode (errors come from the LM covariance).

#### `DC3_setflags` ✅

`DC3_setflags.cpp`, 152 lines, `main` at 28. Post-processes an `extperr` table and attaches
quality flags.

- Args: `-I` extperr output; `-b` list of fibers whose fits are "too broad"; `-n` list "too
  noisy"; `-O` output; `-f` number of fibers (default 331 — a SparsePak/PPak fiber count);
  `-R` image name, which makes it emit IRAF `imreplace <img>[*,i] 0` commands for unfitted
  fibers; `-h`.
- Flow: `readcommandline` → `confirmifile`/`confirmofile` → `readdat` (must be 15 or 19 columns,
  i.e. Gaussian or Gauss–Hermite) → `selectlist` → flag 1 (ok) / 2 (broad) / 3 (noisy) → rewrite
  the table with a trailing `FLAG` column.

#### `extperr` ✅

`extperr.cpp`, 123 lines, `main` at 46. Positional usage: `extperr <root> <output>`.

Joins five ASCII databases produced by the interactive `DC3` — `<root>.gaufit.db`,
`.gaucovar.db`, `.anorm.db`, `.anbtxc.db`, `.fitrms.db` — into one table:
`GID TID I Ierr V Verr VSIG VSIGerr [H3 H3err H4 H4err] RCHI2_XC RCHI2_LAM N_XC A_N N_BT AN_BT
XCRMS`. The number of VDF parameters is inferred from the column count (3 → Gaussian, 5 →
Gauss–Hermite). The file header quotes the e-mail from M. Bershady that drove the column
renaming.

#### `fitrms` ✅

`fitrms.cpp`, 188 lines, `main` at 31. RMS of the normalized CC residual `(X − X_T)/peak(X)`
over the full lag range.

- Args: `-X` list of XC FITS; `-B` list of BTXC FITS; `-R` an existing `fitrms` output (rederives
  the names via `changeext`); `-O`; `-h`.
- Flow: `read1Dimage` both; `norm = max(xc)`; trim to the common power-of-two length (v = 0 is
  always at `npix/2`); `diff = (xc − btxc)/norm`; strip the leading/trailing exactly-zero lag
  regions (`|x| < 1e2·EPSDP`); print `BTXC_FILE NORM RMS`.

#### `mkasym` ✅

`mkasym.cpp`, 184 lines, `main` at 31; helper `asymmetry()` at 147, commented *"Yanked from
correlation.cpp"*.

Builds the CC asymmetry (fold-difference) function `A(X)(i) = X(v+i·dv) − X(v−i·dv)` with
linear interpolation on both sides, restricted to the non-zero lag support, plus its normalized
RMS. This is the quantitative template-mismatch diagnostic of DiskMass Paper II.

- Args: `-I` two-column list `<corrfits> <foldv>`; `-v0` (default `CRVAL1`); `-dv` (default
  `CDELT1`); `-O`; `-h`.
- Writes `<name>.asym.fits` + `writespechead`, and reports `a = rms(asym/norm)`.

#### `redoasym` ✅

`redoasym.cpp`, 141 lines, `main` at 30. Recomputes the asymmetry RMS from an existing DC3
asymmetry database after stripping trailing zeros. Parses the DB by locating `".fits"` in each
line, reads `NORM` with `extract2n`, reloads each `*.asym.fits`.

#### `asym_renorm` ✅

`asym_renorm.cpp`, 131 lines, `main` at 29. Renormalizes existing asymmetry functions by the
peak of a *different* correlation — typically to renormalize `A(X)` by `peak(X_T)` rather than
`peak(X)`. Args `-X`, `-A`, `-O`, `-h`; requires equal-length lists.

#### `stack_pfile` ✅

`stack_pfile.cpp`, 255 lines, `main` at 86. A **parameter-file generator**, not science: it
exposes the 26 `stack` options as command-line tags (28 in total: `-pf -P -S -ES -E -diffe
-werr -mask -stmask -guessv -gsig -hif -lof -O -V -v0 -mv -T -tplv -tweak -sigerr -vtol -viter
-L -W -nw -echo -h`) and writes them as `stack.par` in `%9s = %17s //%s` format. Defaults are
in the `stackopt_str` literal at line 52; the output matches `progfiles/stack.par` verbatim and
is consumed by `readstackopt()` in `DC3_stack.cpp:34`.

#### `test_int` ✅

`test_int.cpp`, 123 lines. A numerical-behaviour probe: prints `int(±0.5 ± ε)` and `int(±1.5 ± ε)`
for ε ∈ {1e-1, 1e-5, 1e-10, 1e-15, EPSDP}. It exists because pixel↔velocity conversions
throughout `vdf.cpp` and `correlation.cpp` use bare `int()` truncation (e.g.
`win.invs = int((win.fvs − v0)/dv)`), and the author needed to confirm the half-pixel rounding
behaviour.

#### `test_convolve` ✅

`test_convolve.cpp`, 471 lines, `main` at 289. Exercises the **variable-σ Gaussian convolution**
used for instrumental-resolution matching.

- Args: `-D` 1-D FITS of instrumental dispersion σ_inst(λ); `-g [n] [σ]` number and dispersion
  of test Gaussians (default 10, 10 km/s); `-h`.
- Flow: `fits2spec` → build a `MultiGaussian` comb across the wavelength range → spline it →
  build `VariableSigmaGaussian<SplineInterpolator>` from `10·siginst` → `Spectrum::convolve` →
  dump `X F G CNV CNVe`.
- **Committed reference outputs** live alongside it: `test_convolve.out` (125 KB),
  `test_convolve.eps`, `test_convolve.sm`, `test_siginst.fits`, `test_siginst_log.fits`. These
  are the only usable golden files in the repository.

#### `test_spec` ✅

`test_spec.cpp`, 335 lines, `main` at 43. A scratch harness. The live body builds 100
Gaussian-noise spectra, velocity-shifts the first and last by ∓10,000 km/s, stacks them, and
writes the result. A large commented block (lines ~160–240) is the **best minimal example of the
VDF API in the repository**:

```cpp
vdffitwin vwin;  vwin.type = NFWHM;  vwin.fwhm = 2.0;
CrossCorrelation xc(ifile, 0, 1, 0,0,0,0, 0, 0, false, obj2, obj, 0, Mat_DP(), ffts, fftf);
VDF vdf("", xc, 1, GAUSS, vwin, &cvp, nmax, alambda, fititer, miter, mvdiff, citer, cp,
        false, false, true, true, fixp, p0, lam, del, &ffts, &fftf);
if (!vdf.fit()) return false;
```

### 5.2 Stale (do not build)

#### `DC3` — the interactive program ❌

`DC3_main.cpp`, 156 lines, `main` at 74. A thin wrapper: `print_intro()` → `readcommandline`
with six tags (`-P -S -ES -norm -outprep -h`) → `setparname()` → `readpars()` → `mainmenu()` →
`cpgend()`. Everything is in `DC3_menu.cpp` (3,313 lines), `DC3_userio.cpp` (1,638), and
`DC3_plot.cpp` (2,959). See §7.

#### `app_siginst` ❌

`app_siginst.cpp`, 273 lines, `main` at 51. Applies the instrumental-dispersion correction of
W11 Appendix A. The header states the algebra directly:

```
dsig^2     = sig^2_B - sig^2_obs - sig^2_off
sig^2_corr = sig^2_obs - dsig^2
```

Args: `-R` raw dispersions from DC3; `-rc a d e` column indices; `-C` corrections file;
`-cc o b` columns for σ_off and σ_B; `-GI` 2-D FITS of galaxy instrumental σ (one row per
aperture); `-TI` text template instrumental σ; `-O`; `-e` fractional σ_inst error (default 0.04);
`-o` aperture↔row offset; `-fe l u` error-limit flags; `-ff` fractional-error flag; `-h`.
Helper `calc_inst_disp_err()` (line 199) approximates σ_inst as the mean of the FITS row.

Note this program predates `Correlation::match_resolution()` (Feb 2013), which subsumes much of
the workflow it was written to support.

#### `combspec` ❌

`combspec.cpp`, 146 lines, `main` at 31. Weighted, velocity-offset coaddition. `-I` three-column
list `<spec> <voff> <weight>`; `-E` error list; `-M` mask list; `-O` root. The real work is
`combspec_mask()` in `libraries/specmanip.cpp`.

#### `stack` ❌

`stack.cpp`, 445 lines, `main` at 61. The substantial stacking program, with five documented
modes:

1. No registration.
2. Register to a systemic `v0`.
3. Register to velocities from a file.
4. Register to velocities derived from input templates (`initvfromtpl`).
5. Register by iteratively adjusting velocities from template fits to the running stack
   (`tweakv`).

Reads `stack.par` via `readstackopt()`. Its 26 options are in `struct stackopt` (`DC3.h:199`).
The file header carries a long, candid to-do/bugs list. Twelve prior versions are archived in
`dc3/stack/` (`stack_v1.cpp` through `stack_v7.5.cpp`, 2003–2010), along with `stack_help.txt`
and `stack.funcd.txt`.

#### `dither_stack` ❌

`dither_stack.cpp`, 723 lines, `main` at 203. A simulation: convolve a template to a known σ,
generate N noise realizations at fixed S/N, assign velocity offsets spread over a multiple of the
Gaussian FWHM, stack with S/N weighting, cross-correlate, and measure the resulting degradation
of the CC peak/width/integral. v2.0 added a GSL `multimin` search (`minimize_stack_width`,
`stack_width`) that attempts to *recover* the offsets by minimizing the stacked CC peak width.

#### `prob_stack` ❌

`prob_stack.cpp`, 336 lines, `main` at 49. Determines the posterior PDF of σ for a set of spectra
assumed to share a single dispersion but different velocity offsets. Key routines
`determine_dispPDF()` (192) and `GaussPDF()` (330); returns `pmodel`, `likelihood`, `disptry`,
and an MCMC-style `dispchain`.

#### `recoverv` ❌

`recoverv.cpp`, 734 lines, `main` at 38. A recovery simulator: broadens supplied templates to a
grid of input VDFs at a grid of S/N, refits, and tabulates recovery statistics. 19 tags including
`-P` parfile, `-T` templates, `-B` VDF list, `-n` MC realizations, `-s [s0 s1 n]` S/N grid, `-O`
root, `-wf` write MC FITS. This is the program that produced W11 Figures 4–6.

#### `stack_sim` ❌

`stack_sim.cpp`, 330 lines, `main` at 37. A controlled test of `stack`: one template, one known σ
(`-d`), velocity offsets uniform over `-v delv`, S/N `-s`, `-nf` fibers, `-ns` noise realizations,
`-nv` velocity realizations.

### 5.3 Purpose summary

| Executable | Purpose | Build | Port disposition |
|---|---|---|---|
| `DC3_express` | Batch CC kinematics | ✅ | **Port — the reference implementation** |
| `DC3` | Interactive CC kinematics | ❌ | Drop (CLI + QA plots replace it) |
| `app_siginst` | Instrumental-σ correction (W11 App. A) | ❌ | **Port — as library functions** |
| `extperr` | Join `.db` tables | ✅ | Subsumed by a results `DataContainer` |
| `DC3_setflags` | Attach quality flags | ✅ | Subsumed (bitmask) |
| `fitrms` | CC-residual RMS | ✅ | Subsumed (computed in-fit) |
| `mkasym` | CC asymmetry function | ✅ | Subsumed (computed in-fit) |
| `redoasym` | Recompute asymmetry RMS | ✅ | Subsumed |
| `asym_renorm` | Renormalize asymmetry | ✅ | Subsumed |
| `stack` | Spectral stacking, 5 modes | ❌ | **Future extension** |
| `combspec` | Weighted coaddition | ❌ | Future extension |
| `dither_stack` | Dither-stacking simulation | ❌ | Future extension |
| `prob_stack` | σ posterior from a stack | ❌ | Future extension |
| `stack_sim` | Stacking simulation | ❌ | Future (test suite) |
| `stack_pfile` | Generate `stack.par` | ✅ | Obsolete (ParSet) |
| `recoverv` | Recovery simulation (W11 Figs 4–6) | ❌ | Becomes the acceptance test suite |
| `test_convolve` | Variable-σ convolution probe | ✅ | Becomes a unit test (has golden files) |
| `test_spec` | Class scratch harness | ✅ | Becomes unit tests |
| `test_int` | `int()` rounding probe | ✅ | Obsolete |

---

## 6. `DC3_express` — control flow in detail

The file header (lines 5–43) is a literal transcript of the interactive-DC3 menu sequence this
program replaces.

### 6.1 Argument parsing (lines 150–254)

19 tags, parsed by `readcommandline(arglist, argc, tags, tagn, value)`, where `tagn[i]` gives the
argument count for tag `i`, and `value` is a `Mat_STR` with `value[i][0]` set to the tag if
present and `value[i][1..n]` holding its arguments.

| Tag | Args | Default | Meaning |
|---|---|---|---|
| `-P` | 1 | `dc3_express.par` | parameter file |
| `-G` | 1 | — | galaxy FITS (`@list` for batch) |
| `-E` | 1 | — | error FITS |
| `-T` | 1 | — | template FITS |
| `-MG` | 1 | — | galaxy pixel-mask FITS |
| `-MT` | 1 | — | template pixel-mask FITS |
| `-MS` | 1 | — | spectral-region mask table |
| `-v` | 2 | 0.0, 30.0 | guess `V`, `σ` |
| `-w` | 1 | 2.0 | fitting window, in CC-peak FWHM |
| `-r` | 1 | −1 | σ-rejection for CC statistics |
| `-l` | 3 | −1,−1,−1 | CC-acceptance criteria (`dvlim`, `peaklim`, `snlim`) |
| `-O` | 1 | `DC3_fit` | output root |
| `-no_overwrite` | 0 | false | abort rather than clobber |
| `-a` | 0 | false | solve the CC amplitude `I` analytically |
| `-SG` | 1 | — | galaxy instrumental σ |
| `-ST` | 1 | — | template instrumental σ |
| `-no_fit` | 0 | false | CCs + plots only |
| `-f` | 2 | −1,−1 | fixed velocity window |
| `-h` | 0 | | help |

Three argument-indexing bugs are present here; see §14.

### 6.2 Setup (lines 256–345)

1. `StopWatch timer;`
2. `confirmifile()` on `pfile`, `mgfile`, `mtfile`, `msfile`. The checks on `gfile`/`efile`/
   `tfile` are commented out because `@`-batch names are not real files.
3. `ParSet par(pfile);` — the key/value reader (§11.1).
4. `read_spectra(par, gfile, efile, mgfile, sgfile, gal)`, then `gal[i].settype(OBJECT)` and
   `ston[i] = gal[i].ston()`.
5. `read_spectra(par, tfile, "", mtfile, stfile, tpl)`, then `tpl[i].settype(TEMPLATE)`.
6. `apwin = 0, cosper = 2` **hardwired** (line 286) — no apodization, per `doc/develop.txt`
   ("For low numbers of lines, no apodization is better") and W11 §2.4.
7. `par.val("taper", taper)`.
8. `read_mask(msfile, rfmask)` — the spectral-region mask table (§11.3).
9. `set_cnvlv_par(par, cnvp)` (line 705) → `cnvlvpar{padtype, contin, minsig=0.85 hardwired,
   maxblk, taper}`.
10. `set_vdf_fit_win(par, vwin)` (line 714) → `winfac = 2.5`, `iter = 1` (obsolete),
    `smn = funcsmbin`, `pad = 0` hardwired, `minwin = noisywin`; then `vwin.type = FIXV` or
    `NFWHM` with `vwin.fwhm = nfwhm`.
11. `check_output_files(oroot, no_overwrite)` (line 737).
12. Open `<oroot>_ccstat.db`, write a timestamp and a 16-column header.

`set_cnt_fit_par()` (line 726) is **entirely commented out** — the `cntpar` path is dormant in
express mode; continuum order and iteration count are passed directly to the `VDF` constructor.

### 6.3 CC construction and statistics (lines 346–390)

A double loop over templates `i` and galaxies `j`, with `k = i·ngal + j`:

```cpp
name = oroot+"_g"+getnum(j+1,ndg)+"_t"+getnum(i+1,ndt)+"_xc";
xc[k] = CrossCorrelation(name, j+1, i+1, apwin, cosper, apwin, cosper, gv, gvs,
                         inp_guess, gal[j], tpl[i], taper, rfmask, &cnvp, fft0, fft1);
ccpeak_properties(xc[k], peakr[k], peakdv[k], peakdv2);
correlation_stats(xc[k], rsig, mean, stddev, stddev_0, peak, nuse, nuse_0);
if (inp_lims)
    fitxc[k] = fabs(peakdv[k]) <= dvlim && (peakr[k] < peaklim || ston[j] > snlim);
```

`ccpeak_properties` (767) locates the highest *secondary* local maximum and returns
`peakr = X₂/X₁`, `peakdv = v_peak − v_mask`, `peakdv2 = v_peak2 − v_mask`.
`correlation_stats` (783) is `DC3_util::istat` inlined: trim the leading/trailing exactly-zero
lag region, then `STAT::meanrej` and `STAT::fixmeanrej(data, 0.0, ...)` for the zero-mean variant.

`<oroot>_ccstat.db` has 16 columns:
`GID TID S/N_G MEAN STDDEV NU STDDEV_0 NU_0 PEAK P/S P/S_0 PEAKDV PEAKR PEAKDV2 RES_BASE F`.
`RES_BASE` is `xc[k].base_sig()`, the constant σ floor added during resolution matching (§8.3).

### 6.4 Fit setup (lines 392–425)

From the parameter file: `nmax`, `fititer`, `miter`, `mvdiff`, `citer`, `corder`. Then:

```cpp
bool pixfrac = mvdiff < 0;     // negative mvdiff ⇒ expressed in pixels
if (corder < 0) citer = 0;
if (citer == 0) corder = 0;
```

Hardwired (lines 406–418): `alambda = 0.001`, `func = GAUSS`, `np = 3` (the `+corder` is
commented out — VDF v3.2 removed continuum coefficients from the parameter vector),
`p0 = {1.0, gv, gvs}`, `lam = {1.0, 100.0, 100.0}`, `del = {0.01, 1.0, 1.0}`. The `lam` values
are precisely W11 §3's "characteristic scales of 1, 100 km/s, and 100 km/s for `a`, `V`, `σ`".

### 6.5 The fit loop (lines 426–451)

```cpp
for (int i = 0; i < nxc; ++i) {
    if (!fitxc[i]) continue;
    if (!inp_guess) {
        p0[1] = xc[i].peakv();
        p0[2] = xc[i].fw(xc[i].peakpix(), 0.5)/sig2fwhm;   // CC-peak FWHM → sigma
    }
    mvd = pixfrac ? fabs(mvdiff)*xc[i].dv() : mvdiff;
    vdf[i] = VDF(name, &xc[i], i, func, vwin, &cnvp, nmax, alambda, fititer, maskiter,
                 mvd, citer, corder, false, true, solve_norm, fixv, p0, lam, del, &fft0, &fft1);
    success[i] = vdf[i].fit();
}
```

**All three iteration tiers are inside `VDF::fit()`** (§8.1).

### 6.6 Outputs (lines 453–627)

`<oroot>_vdf_fits.db` — columns documented inline at lines 457–506:
`GID TID I Ie V Ve VSIG VSIGe [C1..Ccorder CC] RCHI2_XC RCHI2_LAM FE FC MC RES_BASE NX NT AX AT
AC RX F T`, where

- `NX = peak(X)`, `NT = peak(X_T)`
- `AX = rms(A(X))`, `AT = rms(A(X_T))`, `AC = rms(A(X) − A(X_T))` — the W11 asymmetry index,
  corrected for the template's inherent asymmetry
- `RX = rms(X − X_T)` over the full lag range
- `FE`/`FC`/`MC` = `vdf.error()` / `vdf.converged()` / `vdf.mskconv()`; `T` = elapsed seconds

The asymmetry loop (524–614):

```cpp
vwrap = fitxc[k] && success[k] ? vdf[k][V] : xc[k].peakv();
xc[k].asymmetry(vwrap, xc_a[k]);
xc_p[k] = xc[k].peak();
if (fitxc[k] && success[k]) {
    vdf[k].get_fitting_functions(btsp[k], btxc[k]);   // BTSP = T⊗B (+continuum); BTXC = X_T
    btxc[k].asymmetry(vwrap, btxc_a[k]);
    asym_diff[k][kk] = xc_a[k][kk] - btxc_a[k][kk];
    xc_diff[k][kk]   = xc[k][kk]   - btxc[k][kk];
}
```

`output_images(...)` (1739) writes, per input galaxy FITS × template:
`<oroot>_gf<i>_t<j>_{xc, xc_asym, tpl_rmatch, btxc, btxc_asym, btsp}.fits`, each with a matching
`.gpm.fits` good-pixel mask. Single-spectrum cases use `write1Dimage`/`writecorrhead`;
multi-spectrum cases use `write2Dimage`/`write2Dcorrhead`. Failed or unattempted fits are written
as zeros in both the data plane and the GPM.

`diagnostic_plots(...)` (797) opens `<oroot>_fit_plots.ps` and calls `create_plot()` (823) once
per (template, galaxy). Layout on a 16:9 page (`cpgpap(0.0, 9.0/16.0)`):

- `init_xc_panel` (945/979) — `X` vs `X_T` with a residual sub-panel; `plot_fit_window` marks
  `[win.invs, win.inve)`; default view is `maskv ± 10⁴ km/s`. This reproduces the top panel of
  W11 Figures 7 and 8.
- `init_asym_panels` (1101) — four panels: `A(X)`, its residual, `A(X_T)`, its residual, with
  `asym_diff` overplotted.
- `init_spec_panels` (1214/1236) — galaxy, galaxy−fit difference, template, with
  `plot_common_wave` marking the common rest-wavelength limits and `plot_mask_regions` shading
  the masks. It plots `xc.spec1()`/`xc.spec2()`, i.e. the **resolution-matched, masked** spectra
  actually used — not the raw inputs. This reproduces the bottom panel of W11 Figures 7 and 8.
- `write_statistics` (1479/1603) — the parameter and figure-of-merit block.

---

## 7. The interactive `DC3` program

Stale, but it defines the feature set the batch driver was distilled from.

### 7.1 `DC3_menu.cpp` (3,313 lines)

`mainmenu()` (line 66) owns all long-lived state — `Vec_SPEC rawspec, spectra; Mat_DP rfmask;
Vec_AC acorr; Vec_XC ccorr; Vec_VDF vdffits; Vec_MC mcsims;` plus `vdffitwin usrvdfwin;
cnvlvpar usrcnvlv; cntpar usrcnt;` initialized by `distributepars()` — and implements cascade
invalidation: new CCs erase VDF fits and MC simulations; new spectra erase everything; a
parameter or mask change warns and erases dependent products.

Top-level menu (lines 104–112):

```
1 Parameter Menu   2 Spectrum Menu   3 Masking Menu   4 Auto-correlation Menu
5 Cross-correlation Menu   6 (De-)Convolution Menu
7 Velocity broadening function Menu   8 Restart Menu   (unreachable — errormessage at line 220)
```

Only 1 and 2 are offered until spectra are loaded.

- **`parmenu()`** (486) — list, write to log, change a parameter.
- **`specmenu()`** (249) — read spectra / error spectra; list; create a template from a list
  (`mktemplate`); frequency filter (`cosfilt`/`hifilt`/`lofilt`); generate estimated error
  spectra; interactively alter a spectrum; write altered spectra; reset to raw.
- **`maskmenu()`** (541) — 12 options: enter/list/erase mask regions, change the
  template/object designation, change the mask resize flag (`setmaskresize`), set mask velocity
  offsets (`getmbfv`), read masks from file, interactive masking.
- **`autocorrmenu()`** (807) — 11 options: AC individual or all; wavelength-region contribution;
  fit the AC peak; statistics; W20/W50/W80 (`getfullws`); plot; erase.
- **`crosscorrmenu()`** (1538) — 24 options, including CC pairs, CC many-vs-one, CC errors, fit
  the CC peak, statistics, asymmetry about a given velocity, fit RMS for given broadening
  parameters, and **option 21 (flag CCs by list or empirical criteria)** — the interactive
  analogue of `DC3_express`'s `-l dvlim peaklim snlim`.
- **`convmenu()`** (2032) — 18 options driven by a convolve/deconvolve toggle: convolve
  functions with spectra, build BTXC sets, convolve/deconvolve AC and CC functions, deconvolve
  `G` with best-fit VDFs, build best-fit BTXCs with asymmetries, and **option 17 (test
  multi-component convolution)** — the entry point used by `siginst_algorithm.notes` to build a
  resolution-matched template.
- **`veldispmenu()`** (2704) — the scientific core, 26 options: switch between Gaussian and
  Gauss–Hermite; switch window type; fix parameters; produce a χ² map; fit one pair / one object
  vs many templates / many objects vs one template; interactive fit; **Monte Carlo parameter
  errors**; examine and print MC simulations; output fits, errors, and asymmetry values; VDF
  W20/W50/W80; output fitted CC asymmetry functions; output fit RMS; print continuum fits and
  continuum-subtracted spectra; a single PostScript file of all VDF plots; **CC reduced χ² for a
  set of window sizes** (→ `VDF::rchiset`, i.e. the study that set the ≈2×FWHM window of W11
  §3); iteration statistics; fit flagged CCs.

Options 15, 17, 18, 19, 20, and 21 are exactly the outputs `DC3_express` now produces
automatically.

### 7.2 `DC3_userio.cpp` (1,638 lines)

Terminal I/O: `print_intro` (50); `defmask` (69); `gettflag` (104); `getrfmask`/`gettomask`/
`promptmask` (134/208/271); `setmaskresize` (321); `listmasks` (354); `getmbfv` (378); `ghstat`
(489); `mktemplate` (621); `setwintype` (777); `outputstats` (826); `getwidths` (865); `peakfw`
(979/994); `getfullws` (1015/1051/1087); `getpair` (1128/1189); `listccs`/`listvdfs`/`listmvdf`/
`listgvdf`/`printvdfs`; `readacindex` (1506); `xc2vdf_match` (1534/1583); `spfiltk` (1625).

### 7.3 `DC3_plot.cpp` (2,959 lines)

The PGPLOT layer. Generic helpers: `plotlim` (67/89), `opt_vdfplotlim` (138), `plotcorr` (184),
`plotcorr_interact` (220/251/284), `zoom` (428, factor √2), `lineplt`, `drawlines`, `pointplt`,
`plotspec`, `plotpspec`, `plotspec_interact` (2649/2664).

The centrepiece is **`fitb_interact()`** (618, ~790 lines) — the interactive VDF fit. It composes
a four-region display defined in `DC3.h:52–71`: `fxcvp` (CC), `fspvp` (object spectrum), `ftpvp`
(template), `fparvp` (parameters), indexed `XCPL=0, SPPL=1, TPPL=2, PRPL=3`. Supporting routines:
`prepvdfplot` (1409) with options `overdiff/showdiff/showprep/meansub/smdiff`; `plotfitwin`
(1576); `fitvdfplot` (1620); `vdfpl_mask` (1851) + `rect_overlap` (1885) + `addcolor` (1927),
which render overlapping mask regions with blended colours; `plotpars` (1960); `printfit`/
`logfit` (2070/2174); `convertcurs` (2308); **`readcommand`** (2372), the keystroke language;
`printfitboptions` (2524) / `printcorroptions` (2582); `writefplot` (2630); `markoutdate` (2284).

The interactive command set (authoritative listing in `doc/descrip.txt` §4; `doc/plot.readme` is
the implementation checklist):

- *Graph keys:* `X/Y/Z/D` zoom about the cursor (×√2); `x/y/z` two-click range or box; `r/R`
  redraw; `S` set plot region; `W` show only the fitting window; `f` (re)fit; `M` mark the fitted
  peak and window limits; `p` mark the peak to fit; `w` define the fitting window (two clicks);
  `v` fix the velocity to the cursor; `F` free all fixed variables; `m` define a masking region;
  `b/B` set/reset the baseline; `s` compute peak, W20, W50, W80; `d` toggle residuals; `e` toggle
  mean-subtracted; `c` toggle spectra-as-prepared-for-correlation; `q` quit; `?` help.
- *Colon commands:* `:x`, `:y`, `:fixi`, `:fixv`, `:fixs`, `:fixh3`, `:fixh4`, `:maskT`,
  `:maskG`, `:setwin`, `:setvdf`, `:retmask`, `:nomask`.
- Not implemented: `o` (overplot), `n` (normalize to peak).

---

## 8. The algorithmic core

### 8.1 `VDF` — `libraries/vdf.cpp`, `headers/vdf.h`

The version history in `vdf.h` is the best available summary of the algorithm's evolution:

| Version | Date | Change |
|---|---|---|
| v1.1 | 27 Nov 2006 | Hold fitted continuum coefficients |
| v1.4 | 20 Dec 2006 | Limit the common wavelengths to exclude convolution contamination; add mask-width adjustment based on `VSIG`; add reduced-χ²-vs-window-size |
| v2.0 | 8 Dec 2012 | Accommodate the new `Correlation` class; use Eigen for linear algebra |
| v3.0 | 23 Jan 2013 | Fit the continuum **simultaneously** with the kinematics; apply it to the BTSP, not to the input CC |
| v3.1 | 24 Jan 2013 | **Force the constant offset fixed** — "the first-order component of the continuum is irrelevant to the XC chi-square because the spectrum means are subtracted before correlation. Thus, allowing this component to be fit destabilizes the fit." |
| v3.2 | 28 Jan 2013 | Revert to solving the continuum iteratively, but loop **inside** the mask loop rather than outside it |
| v4.0 | 8 Feb 2013 | Account for the object/template resolution difference, updated during the mask iteration |

Constants (`vdf.h:61–66`): `conviter = 2`, `mrqconv = 1.0e-3`, `alfac = 10.0`, `vsigmax = 600`,
`maxiter = 10` (the "hard limit of ten iterations" of W11 §3).
Window types (`vdf.h:73–81`): `NWINCHNG=0, XZERO=1, PMIN=2, NFWHM=3, FIXV=4, NVSIG=5, NVSIGND=6,
FULL=7`. Iteration types (`vdf.h:84–87`): `ITER_DONE=0, ITER_MASK=1, ITER_CONT=2, ITER_WIND=3`.

#### The driver — `VDF::fit()` (line 1473)

```cpp
int totiter = fiter * 5;            // total restarts before giving up
int maxmi = maxiter;
bool minimize_mask = miter < 0;     // iterate the mask to convergence rather than a fixed count
int ii, jj, mm = miter, cc = citer;
for ( ; ; ) {                                                   // ── outer: tiers 2 & 3
    improved = false;
    for (jj = 0, ii = 0; ii < fiter && jj < totiter; ++ii, ++jj) {   // ── restarts
        if (ii > 0 || newguess) { newguess = false; randomizepars(peakv); }
        if (!dofit(newguess, minimize_mask ? miter : miter-mm, cc, ii))   // ── TIER 1
            return !(fiterr = true);
        if ((ii == 0 && cc == citer) || (chi_xc < _chi_xc && !hitnmax) || (_hitnmax && !hitnmax)) {
            _p = p; _coeff = cf.coeff; _covar = covar; _chi_xc = chi_xc;
            _hitnmax = hitnmax; improved = true;
        }
    }
    check_mask_convergence(_p[V]-xc->maskv(), _p[VSIG]-xc->maskvs());
    if (improved) {
        p = _p; cf.coeff = _coeff;
        get_fitting_functions(&btxc);
        fit_continuum();                                        // ── TIER 3
        check_continuum_convergence(_coeff);
    }
    itype = itertype(minimize_mask, mm, maxmi, cc, _coeff, improved);
    if      (itype == ITER_DONE) break;
    else if (itype == ITER_CONT) { --cc; randomizepars(peakv); }
    else if (itype == ITER_MASK) {                              // ── TIER 2
        initpars();
        updatexc(_p[V], _p[VSIG]);   // re-transcribe masks, re-match resolution, re-correlate
        getwindow();
        calcsig();
        peakv = xc->peakv();
        _chi_xc = 1.0/EPSDP; _hitnmax = false; cnt_converged = false;
        cc = citer;
        if (!minimize_mask) --mm; else ++miter;
    }
}
```

After the loop (1618–1656): set `hitnmax`, `p = _p`; if `fitp[VSIG] && p[VSIG]² − base_sig² <
EPSDP`, **fix σ to 0 and recursively re-`fit()`**; restore `_coeff` and `_covar`; take the square
root of the covariance diagonal so `cov(i,i)` reports the 1σ error; rebuild the fitting functions;
evaluate both χ²; report; return `!hitnmax`.

The ordering policy — `itertype()` (667):

```cpp
if (cc > 0 && !cnt_converged && nonzero && improved)              return ITER_CONT;
if (mm > 0 || (minimize_mask && !msk_converged && miter < maxmi)) return ITER_MASK;
                                                                  return ITER_DONE;
```

i.e. **tier 3 is exhausted before each tier-2 update** — the v3.2 change. `nonzero` guards
against wasting iterations when the previous continuum solution was identically zero.

#### Tier 1 — Levenberg–Marquardt: `VDF::dofit()` (line 714)

A hand-rolled LM, not NR's `mrqmin`:

```cpp
mrqabc(alpha, beta);                        // first α, β, χ²
double al = alambda, convlimit = mrqconv;
for ( ; ; ) {
    AA = alpha;
    for (int j = 0; j < mfit; ++j) AA(j,j) = alpha(j,j)*(1.0 + al);
    da = AA.partialPivLu().solve(beta);
    if (al == 0.0) { CC = AA.inverse(); set_covar(CC); return true; }   // converged
    for (int j = 0, i = 0; i < np; ++i) {
        if (!fitp[i]) continue;
        p[i] = bestp[i] + da(j++);
        if (isnan(p[i])) p[i] = bestp[i];
    }
    mrqabc(AA, da);  ++nfunk;
    if (fabs(chi_xc - o_chi_xc) < convlimit) ++dchiiter; else dchiiter = 0;
    if      (chi_xc <  o_chi_xc && dchiiter < 2) { al /= alfac; /* accept */ }
    else if (chi_xc >= o_chi_xc && dchiiter < 2) { al *= alfac; /* reject */ }
    else if (dchiiter == 2)                       al = 0.0;    // converged: conviter = 2
    if (al > 1e20) { al = alambda; nfunk = 1; convlimit += mrqconv;
                     if (nrestart < fiter) ++nrestart; else { al = 0.0; hitnmax = true; } }
    if (nfunk >= nmax) { al = 0.0; hitnmax = true; }
}
```

Convergence is Δχ² < 1e-3 on two consecutive iterations. If the LU solve throws, χ² is set to
1/EPSDP, the covariance is zeroed, the restart counter is decremented, and a new random start is
requested — the modern form of the old *"require a new parameter set if gaussj() not successful
for 'da'"*.

#### The LM kernel — `VDF::mrqabc()` (line 883)

```cpp
if (fitp[I] && p[I] < 0) p[I] *= -1;                 // amplitude must be positive
if (fitp[VSIG]) check_vsig();                        // 0 < VSIG <= vsigmax (600 km/s)
if (win.type == NVSIG) { setwin(...); setinvsig2(); }// window tracks the fitted sigma
double vmin = v0+win.invs*dv, vmax = v0+win.inve*dv; // V must lie inside the window
if (fitp[V] && p[V] > vmax) p[V] = vmax;
if (fitp[V] && p[V] < vmin) p[V] = vmin;

get_fitting_functions(&btxc);                        // X_T = (T⊗B)' ∘ T
evalchi(&btxc);                                      // chi_xc

int nw = wsize();  dp.resize(nw, mfit);
for (int k, i = 0, j = 0; j < np; ++j) {             // CENTRAL FINITE DIFFERENCES
    if (!fitp[j]) continue;
    p[j] -= del[j]/2;  get_fitting_functions(&btxc_n);
    p[j] += del[j];    get_fitting_functions(&btxc_p);
    p[j] -= del[j]/2;
    for (k = 0; k < nw; ++k)
        dp(k,i) = (btxc_p[win.invs+k] - btxc_n[win.invs+k]) / del[j];
    ++i;
}
if (fitp[VSIG] && p[VSIG] < cvp->minsig*dv/cvp->maxblk) p[VSIG] = 0;   // AFTER the derivative

BB = invsig2.rows() == 0 ? dp : invsig2 * dp;        // Σ⁻¹ · ∂X_T/∂p
aa = dp.transpose() * BB;                            // α = (∂X_T/∂p)ᵀ Σ⁻¹ (∂X_T/∂p)
for (int k, i = 0, j = 0; j < np; ++j) {             // β = (X − X_T)ᵀ Σ⁻¹ (∂X_T/∂p)
    if (!fitp[j]) continue;
    bb(i) = 0.0;
    for (k = 0; k < nw; ++k)
        bb(i) += (xc->operator[](win.invs+k) - btxc[win.invs+k]) * BB(k,i);
    ++i;
}
```

**This is the performance bottleneck.** Each derivative costs **two full convolve + correlate
cycles**, so one LM step costs `1 + 2·mfit` forward-model evaluations. With `mfit = 3` that is 7
model evaluations ≈ 28 FFTs per step, multiplied by up to `nmax = 50` steps × `fititer = 5`
restarts × tier-2 × tier-3 iterations. The FFTW plans `fft0`/`fft1` are passed by pointer
precisely so they can be reused. `deriv_vsig()` (868) enlarges `del[VSIG]` when the negative
half-step would push σ below the block-replication floor.

W11 §3 describes this as *"The derivatives of the fitting function `X_T` required by this
minimization routine are determined via a finite-differencing method."*

#### The χ² — `VDF::evalchi_xc()` (line 1169)

```cpp
dx.resize(wsize());
for (int i = win.invs; i < win.inve; ++i)
    dx[i - win.invs] = xc->operator[](i) - btxc_ptr->operator[](i);
ndx = invsig2.rows() > 0 ? dx*invsig2 : dx;
chi_xc = ndx.dot(dx);
```

W11 Eq. 5 over `[win.invs, win.inve)`. If no errors were supplied, `invsig2` is empty and this
degenerates to unweighted least squares. `rchi()` = `chi_xc / (win.inve − win.invs − mfit)`.
`evalchi_wave()` computes an auxiliary χ² in wavelength space over the unmasked pixels.

#### The covariance — `calcsig`/`setsig2`/`setinvsig2` (348/399/427)

```cpp
// calcsig: apodize the object errors exactly as the object spectrum was apodized
for (int i = 0; i < nn; ++i) { mo[i] = xc->spec1().mask(i);
                               ee[i] = mo[i] ? 0.0 : xc->spec1().error(i); }
apodize(ee, mo, xc->apod1(), xc->cos1());
xc->prep_fft2(tt);                  // the template as it enters the FFT

// setsig2:  W11 Eq. 4
int start = win.sigs - nn/2, end = win.sige - nn/2;
for (int k, j = start; j < end; ++j)
    for (k = 0; k < nn; ++k)
        SS(k, j-start) = sp2[shift_index(nn, j, k)] * sp1err[k];
sig2 = SS.transpose() * SS;

// setinvsig2:
int is = win.invs - win.sigs;
invsig2 = sig2.block(is, is, nw, nw).inverse();
```

`shift_index(size, x, k)` is the circular lag index `k − x` wrapped into `[0, size)`. `sig2` is
built over the wider `[sigs, sige)` window (`winfac` × the fit window when `win.type == NVSIG`)
so the inverse sub-block can be re-extracted cheaply as σ changes.

> **Note for the port:** `(δX)²` depends only on `T` and `δG`, never on `B`. It is therefore
> constant throughout a tier-1 fit. The C++ nonetheless calls `.inverse()` explicitly — an O(W³)
> operation — whenever the window changes. A single Cholesky factorization per tier-2 iteration,
> with χ² evaluated by triangular solve, is both faster and better conditioned.

#### The forward model

```cpp
void VDF::get_fitting_functions(CrossCorrelation *btxc_ptr) {
    getbtsp();                 // T ⊗ B  (unnormalized)
    getbtxc(btxc_ptr);         // (T⊗B)' ∘ T
    renormalize_functions(btxc_ptr);
}
```

**`getbtsp()`** (1007) — the physics:

```cpp
// copy only the pixels that existed in the original template
for (int j = 0, i = 0; j < tnn; ++i) {
    wave = l0 + i*dl;
    if (wave < tl0) continue;
    btsp_small[j++] = xc->spec2()[i] - xc->spec2().getmean();
}
ap[V] = 0.0;                                             // shift applied separately
ap[VSIG] = SQR(p[VSIG]) - SQR(xc->base_sig());           // remove the resolution-matching floor
ap[VSIG] = ap[VSIG] < 0 ? 0.0 : sqrt(ap[VSIG]);
if (ap[VSIG] < cvp->minsig*dv) cnvlv_fft(btsp_small, dv, ap, ftype, true, *cvp);
else                           cnvlv_fft(btsp_small, dv, ap, ftype, true, *cvp, *fft0, *fft1);
// re-embed on the correlation grid
for (...) btsp[i] = (wave < tl0 || j >= tnn) ? 0.0 : btsp_small[j++];
shift_lininterp(btsp, dv, p[V], 0.0);                    // apply the velocity shift
if (cfit)                                                // add the Legendre continuum
    for (int j, i = 0; i < nn; ++i)
        for (j = 0; j < cf.order; ++j)
            btsp[i] += cf.coeff[j]*cf.basis[j][i];
// subtract the mean over unmasked pixels
```

The `ap[VSIG] = sqrt(p[VSIG]² − base_sig²)` step is the v4.0 change: the *fitted* σ is the total
broadening, and what is actually convolved into the template is the total minus the constant
floor introduced by resolution matching.

**`getbtxc()`** (1076) — replaces the object plane of the model's spectrum 1 with `btsp + mean(G)`,
calls `setmean()`, then `btxc_ptr->update(*fft0, *fft1)`, which re-runs **the full masking,
apodization, and FFT correlation**.

> This is the operational meaning of W11 Eq. 2. The model is `(T⊗B)' ∘ T`, not `(T∘T) ⊗ B`,
> *because* re-running `update()` applies to the model precisely the censoring and masking that
> was applied to the data. `doc/develop.txt` records: *"1 and 2 should be identical when no masks
> are applied. If masks are applied, 2 provides symmetry that is more reliable."*

**`calc_norm()`** (1132) — the analytic amplitude, documented in-code:

```
I = (X_T · Σ⁻¹ · X) / (X_T · Σ⁻¹ · X_T)
```

obtained by pulling `I` out of `X_T`, differentiating χ² with respect to `I`, and setting the
derivative to zero (Statler 1995). Requires the normalization used in building `X_T` to be
exactly 1.0. Enabled by `-a`/`solve_norm`; `setfix()` then forces `p[I] = 1.0; fitp[I] = false`
and refuses to both fix and solve for `I`. When enabled, `DC3_express` writes `Ie = −1` because
no covariance element exists for it.

> **Note for the port:** this is exactly the `K = 1` case of a non-negative generalized
> least-squares solve for template weights. Extending `X_T` from a vector to an `N × K` design
> matrix turns `calc_norm` into optimal template mixing with no change of principle.

#### Tier 2 — mask transcription: `VDF::updatexc()` (line 693)

```cpp
void VDF::updatexc(double gv, double gvs) {
    xc->set_mt_cw(gv, gvs, 6.0, *fft0, *fft1);   // 6-sigma taper of the common wavelength range
    init_cnt();                                  // re-initialize the continuum basis and weights
    btxc = (*xc);  btxc_n = (*xc);  btxc_p = (*xc);
}
```

`Correlation::set_mt_cw` does the whole job: `shift_input_masks()` → `update_region_masks()` →
`commonwave()` → shrink by `nsig·mvs` pixels at each end → `reset_masks()` → `match_resolution()`
→ `correl()`. Convergence: `msk_converged = !(|vdiff| > mvdiff || |vsdiff| > mvdiff)`.

#### Tier 3 — continuum: `VDF::fit_continuum()` (line 561)

```cpp
double fac, facsave = solve_norm ? fnorm : 1.0;
const int max_iterations = 20;   const double fac_conv = 1e-5;
for (int j = 0; j < max_iterations; ++j) {
    cf.coeff = 0.0;                        // remove the old continuum
    getbtsp();
    for (int i = 0; i < nn; ++i)
        cy[i] = (xc->spec1()[i] - xc->spec1().getmean())/facsave - btsp[i];   // G − T_B
    refit(cf, cx, cy, cw);                 // Legendre, error-weighted
    for (int i = 0; i < cf.order; ++i)
        if (fabs(cf.coeff[i]) < 1e-4) cf.coeff[i] = 0.0;
    get_fitting_functions(&btxc);
    fac = solve_norm ? fnorm : 1.0;
    if (isnan(fac)) { cf.coeff = 0.0; get_fitting_functions(&btxc); break; }
    if (fabs(fac - facsave) < fac_conv) break;    // normalization and continuum co-converged
    facsave = fac;
}
```

`init_cnt()` (179) sets up the fit once per mask iteration — `cx = coovec()`, `cw[i] = 1/σ_i²`
(0 for masked pixels), `cr = crinit(cp)`, `cf = cfinit(cp, ...)`, `eqnpts(...)`,
`chofac(cf.a, cf.chodiag)` — so the Cholesky factorization of the normal-equation matrix is
computed once and `refit()` only re-solves the right-hand side.
`check_continuum_convergence()` (619) tests, per coefficient: pass if (not small and
`|new/old − 1| < 0.05`) or (small and `|new − old| < 1e-3`).

The continuum is added to the **BTSP** and thence to the model — **not** subtracted from the
input CC. This is the v3.0 design decision recorded at `vdf.h:27–28`.

### 8.2 `Correlation` — `libraries/correlation.cpp`

Invariants documented at `correlation.h:7–45`: `v = 0` is always at pixel `nn/2`; `nn` is a power
of two; the continuum is subtracted before the spectra are copied in; pixels are tapered on
input; **spectrum 1 is always the object and spectrum 2 always the template**; `setspec()` does
*not* apply the common-wavelength mask — `setmask()` must be called.

> ⚠️ **The "power of two" invariant is stale documentation.** `Correlation::setsize()`
> (`correlation.cpp:395–410`) actually computes
>
> ```cpp
> nn = int(2.2*(int((ln-l0p)/dlp)+1));   // Length for the correlation
> if (nn % 2 != 0) ++nn;                 // Make nn even
> dvp = spec[0].dv();
> v0p = double(-nn)/2 * dvp;             // Initial velocity of the correlation
> ```
>
> with the in-code comment: *"A minimum factor of 2 is required to treat one spectrum as the
> response function for the other. I increased the length by 10% for further precaution. I also
> make the length even so that v=0 is always in pixel nn/2."* Only **evenness** is enforced.
> Verified against production output in §16: 2048-pixel spectra give `nn = 4508`, exactly
> `2.2 × 2049 → 4507 → 4508`. `libraries/fft.cpp` uses `fftw_plan_dft_r2c_1d` /
> `fftw_plan_dft_c2r_1d`, which accept arbitrary lengths; there is no radix-2 restriction
> anywhere in `cnvlv.cpp`, `fft.cpp`, or `correlation.cpp`. W11 §2.4 and footnote 5 describe
> power-of-two padding ("FFT algorithms exist that do not require the number of discrete samples
> to be a power of two; however, we have not yet implemented them in our code"); the code has
> since moved on and the comments and `doc/descrip.txt` were never updated.

Documented call order:

```
Correlation(): setspec() → correlcheck(), setsize(), maps(), mapm(), correl()
               guessvdf() → gv, gvs
               commonwave() → ls, le
               setmask() → adjmask(), mapm(), clipwave(), mask()
               correl()
```

**`correl_d()`** — the FFT correlation itself:

```cpp
fft0.forward();               // FFT of the object
fft1.kernel_reorder();        // rotate the template so its origin is at index 0
fft1.forward();
fft1.complex_conjugate();
if (!fft0.multiply_transform(fft1)) throw Handle_Err("Bad correlation.");
fft0.backward();
fft0.multiply_data(1.0/double(fft0.size()));
```

**`prep_fft(ii, dd)`** — masking, mean subtraction, apodization (W11 §2.4):

```cpp
for (int i = 0; i < nn; ++i) {
    dm[i] = spec[ii].mask(i);
    dd[i] = dm[i] ? 0.0 : spec[ii][i] - spec[ii].cont(i);
    if (!dm[i]) { mean += dd[i]; ++nm; }
}
if (nm == 0) throw Handle_Err("Entire spectrum has been masked!");
mean /= nm;
for (int i = 0; i < nn; ++i) if (!dm[i]) dd[i] -= mean;
apodize(dd, dm, apod[ii], cosp[ii]);
```

The mean is over unmasked pixels only, and `apodize` treats each unmasked block independently
(`doc/descrip.txt` §2.C.f).

**`commonwave()`** — the detector-censoring step that gives the code its name:

```cpp
double z = calcz(mv, false);
ls = max(sl0[0] - log10(1+z), sl0[1]);
le = min(sl0[0] + (snn[0]-1)*dlp - log10(1+z), sl0[1] + (snn[1]-1)*dlp);
```

`VDF::updatexc` then shrinks this further by `6σ` pixels at each end to keep convolution
wrap-around out of the correlation — the practical implementation of W11 §2.2's argument that
the galaxy OSR must be limited to accommodate the template OSR plus the fitted convolution width.

**`adjmask(imask, omask)`** — the live tier-2 region transcription:

```cpp
double mmvs = SQR(mvs) - SQR(res_base);          // remove the resolution-matching floor
mmvs = mmvs < 0 ? 0.0 : sqrt(mmvs);
for (int i = 0; i < nm; ++i) {
    omask[i] = imask[i];
    sign = (imask[i][0] == TEMPLATE) ? +1 : -1;
    omask[i+nm][0] = (imask[i][0] == TEMPLATE) ? OBJECT : TEMPLATE;
    mfac = NSIGMSK * mmvs / dvp;                  // NSIGMSK = 2.0
    npvs = fabs(mfac) > EPSDP ? int(mfac)+1 : 0;
    if (npvs != 0) {
        npvs *= int(imask[i][3]);                 // -1 shrink, 0 fix, +1 grow
        omask[i][2] = (omask[i][1] + omask[i][2]/2.0)*pow(10.0,  sign*npvs*dlp)
                    - (omask[i][1] - omask[i][2]/2.0)*pow(10.0, -sign*npvs*dlp);
        if (omask[i][2] < 0) omask[i][2] = 0;
    }
    mfac = 1 + calcz(mv, false);                  // redshift the copy into the other frame
    omask[i+nm][1] = sign > 0 ? omask[i][1]*mfac : omask[i][1]/mfac;
    omask[i+nm][2] = sign > 0 ? omask[i][2]*mfac : omask[i][2]/mfac;
    omask[i+nm][3] = imask[i][3];
}
```

This implements W11 §2.3 and `doc/descrip.txt` §2.D: a region masked in one frame is mapped into
the other by `(1+z)^±1` and, if flagged to grow, widened to account for the stellar broadening —
W11 Figure 2 shows the effect for the [O III] and [N I] lines.

**`guessvdf()`** — the initial estimate:

```cpp
gv = peakpv;
fft0.set_data(dd); fft1.set_data(dd); correl_d(fft0, fft1); fft0.data(dd);  // AC of the template
double acfwhm = fw(dd, nn/2, 0.5);
double xcfwhm = fwhm(peakpi);
gvs = (xcfwhm - acfwhm < EPSDP) ? 0
    : sqrt(SQR(xcfwhm) - SQR(acfwhm)) / (2*sqrt(2*log(2.0)));
```

**`asymmetry(v, diff)`** — the CC asymmetry function:

```cpp
int ri = int((v-v0p)/dvp);
int ndiff = nn-ri < ri ? nn-ri : ri;
diff.resize(ndiff);  diff[0] = 0;
for (int jn, jp, i = 1; i < ndiff; ++i) {
    jn = ri-(i-1);  vn = v-i*dvp;
    jp = ri+(i-1);  vp = v+i*dvp;
    diff[i] = linear_interpolate(..., vp) - linear_interpolate(..., vn);
}
```

Commented-out GSL cubic-spline variants sit alongside the live linear interpolation.

### 8.3 `Correlation::match_resolution()` (v3.0, Feb 2013)

```cpp
if (!spec[0].hasres() || !spec[1].hasres()) { res_match.erase(); res_base = 0.0; return; }
double zw = log10(1+calcz(mv, false));
SplineInterpolator spl0(&x, &r0);                     // object resolution vs log lambda
for (...) {
    res_match[j] = SQR(spl0.sample(wave+zw)) - SQR(spec[2].resolution(i));   // σ_G² − σ_T²
    min = min(min, res_match[j]);
}
res_base = min < SQR(MIN_SIG_RESMATCH) ? SQR(MIN_SIG_RESMATCH) - min : 0.0;  // MIN_SIG_RESMATCH = 1 km/s
for (...) res_match[i] = sqrt(res_match[i] + res_base);
res_base = res_base > 0 ? sqrt(res_base) : 0.0;

SplineInterpolator spl(&vel, &res_match);
VariableSigmaGaussian<SplineInterpolator> f(0.0, &spl);
if (!cnvlv_integral(&f, tpl, dvp, *cvp)) throw Handle_Err("Matching resolution problem");
```

The constant floor `res_base` is added so the kernel σ is never below 1 km/s (the convolution
integral is unstable as σ → 0). It is removed again inside `VDF::getbtsp` and
`Correlation::adjmask`, and reported as the `RES_BASE` column in both `DC3_express` databases.
`spec[2]` holds the **original, unmatched** template alongside the working copy.

This is the `σ_off` of W11 Appendix A (Eq. A1), and its introduction in Feb 2013 supersedes most
of the 18-step manual workflow documented in `siginst_algorithm.notes`.

> ⚠️ **`res_base` is a hard floor on the measurable velocity dispersion.** This consequence is
> nowhere documented in the code or the paper, and it matters.
>
> After matching, the template's instrumental resolution is
> `sqrt(σ_T² + σ_G² − σ_T² + res_base²) = sqrt(σ_G² + res_base²)` — uniformly **broader** than
> the galaxy by `res_base`, independent of wavelength. `VDF::getbtsp` then convolves by
> `ap[VSIG] = sqrt(p[VSIG]² − base_sig²)`, so the total model broadening is
> `sqrt(σ_G² + σ_fit²)` and **`res_base` cancels exactly**: the fitted `σ_fit` is the
> astrophysical σ, with no residual instrumental term.
>
> But the subtraction requires `σ_fit ≥ res_base`. Below that the argument goes negative,
> `getbtsp` clamps it to zero, and `VDF::fit()` (§8.1, after the main loop) responds by fixing
> σ ≡ 0 and recursively re-fitting. **Every spectrum whose true dispersion falls below
> `res_base` is therefore reported as σ = 0, not as an upper limit.**
>
> The floor is not small. `res_base² = MIN_SIG_RESMATCH² − min_λ(σ_G² − σ_T²)` grows with the
> extent to which the template is *broader* than the galaxy somewhere in the range, and the
> February 2013 campaign recorded `RES_BASE` between **0 and 9.50 km/s** (§16.4) — against W11
> §4 simulations that probe down to `σ_m = 5` km/s. It bites only when resolution vectors are
> supplied: without them `match_resolution` returns early with `res_base = 0` and there is no
> floor, which is why the `_od` production runs are unaffected and the `_cd` runs are not.
>
> Note this is a *separate* floor from the block-replication limit of §8.4
> (`0.85·dv/maxblk`), which is set by pixelization rather than by resolution. A port should
> report both, per spectrum.

### 8.4 `cnvlv` — convolution and block replication

**Block replication** — the sub-Nyquist trick of W11 §3:

```cpp
int cnvlv_blk(double dv, double sig, const cnvlvpar &cvpar) {
    double minsigv = cvpar.minsig * dv;                 // minsig = 0.85 px ⇒ FWHM = 2 px
    int ndiv;
    for (ndiv = 1; sig < minsigv && ndiv <= cvpar.maxblk; minsigv /= 2, ndiv *= 2);
    return ndiv;
}
```

and its use in `cnvlv_fft` (66):

```cpp
int ndiv = cnvlv_blk(dv, p[VSIG], cvpar);
if (ndiv > cvpar.maxblk) {                              // sigma is unmeasurably small
    f *= p[I];
    if (fabs(p[V]) > EPSDP) shift_lininterp(f, dv, p[V], 0.0);
    return;                                             // pure velocity shift, no broadening
}
if (ndiv > 1) { blkrep(f, ndiv);  dv /= double(ndiv); } // sub-sample
int nn = f.size();
cnvlv_prep(f, cvpar, nn/2);                             // pad
Vec_DP rf(f.size());
returnfunc(p, dv, f.size(), rf, f.size(), functype);    // build the Gaussian / GH kernel
cnvlv_fft(f, rf, conv, fftf, fftr);                     // FFT multiply (or divide, to deconvolve)
int nrm;  cnvlv_chop(f, nn/2, cvpar, nn, nrm);          // unpad
if (ndiv > 1) blkav(f, ndiv);                           // re-bin
```

`blkrep` is nearest-neighbour up-sampling; `blkav` averages back down with correct handling of a
ragged final block. `maxblk = 8` gives an effective σ floor of `0.85·dv/8 ≈ 0.106 px`; below that
σ is set to exactly zero (`vdf.cpp:953`).

**Padding** — `cnvlv_prep(f, cvpar, padlen)` pads by `nn/2` at each end with `constant`,
`nearest`, `wrap`, or `reflect` (the default; reflects repeatedly until filled). These map
exactly onto `numpy.pad(mode='constant'|'edge'|'wrap'|'reflect')`.

**Variable-σ convolution** (v3.0, 7 Feb 2013) — the templated machinery used by
`match_resolution`: `VariableSigmaGaussian<T>` (a Gaussian whose σ is sampled from an
interpolator at the running position), `Convolution_Integrand<T,U>` (a GSL `gsl_integration_qagi`
wrapper with tolerances hardwired to 1e-10 / 1e-6 and `cnv_intervals = 1000`), and
`cnvlv_integral(k, f, dx, cvpar)`. **This per-pixel adaptive-quadrature loop is the other major
performance liability**; a vectorized banded-matrix convolution replaces it.

---

## 9. Module-by-module inventory

### 9.1 `dc3/` — live

| File | Lines | Role |
|---|---|---|
| `DC3_express.cpp` | 2,093 | The production driver (§6) |
| `DC3_setflags.cpp` | 152 | Quality flags |

### 9.2 `dc3/` — required by the stale `DC3` binary

| File | Lines | Role |
|---|---|---|
| `DC3_menu.cpp` | 3,313 | The menu tree (§7.1) |
| `DC3_fileio.cpp` | 3,041 | Readers, single-object writers, `.db` writers, multi-extension collation writers, `vdftops`, `post_xc_rchi` |
| `DC3_plot.cpp` | 2,959 | PGPLOT layer; `fitb_interact` (§7.3) |
| `DC3_userio.cpp` | 1,638 | Terminal prompts (§7.2) |
| `DC3_stack.cpp` | 1,077 | `readstackopt`, `stack_warning`, `readval_slink`/`readval_flink`, `initvfromtpl`, `tweakv`, `getstackspec`, `getstacklist`, `maskspectrum` |
| `DC3_util.cpp` | 730 | Index bookkeeping (`pairname`, `findpair`, `setxcpairs`, `getaci`, `getvdfi`), the original `adjmask`, statistics (`givestat`, `moment`, `istat`), `matchcontinuum`, `msktype`/`mskstring` |
| `DC3_par.cpp` | 726 | The legacy 40-line positional parameter reader; `setparname`, `checkpars`, `changepars`, `setpar`, `distributepars` |
| `DC3_win.cpp` | 191 | `getwindow` ×3, `winedges` — superseded by `VDF::getwindow` |
| `DC3_fft.cpp` | 181 | `autocorr`, `crosscorr`, `getbtxc` batch wrappers — **the file that no longer compiles**, because it calls `AutoCorrelation::setac(...)` and an old `Correlation(...)` constructor |
| `DC3_monte.cpp` | 178 | `veldispmonte` — Monte Carlo parameter errors |

`DC3_monte.cpp` carries two live warnings: *"Monte Carlo routine allows for adjustment of the
fitting window. Updates should remedy this situation!"* and *"CHECK HOW TO REPLACE GALAXY
SPECTRUM WITH BTSP — NOT JUST HERE BUT IN VDF.CPP"*. `doc/develop.txt` records that MC and
covariance-matrix errors were previously found equivalent, with the covariance route much
faster — hence `DC3_express` uses covariance errors exclusively — but asks for the comparison to
be redone under the new fitting scheme. This is an open scientific question.

### 9.3 `dc3/` — frozen legacy (not compiled)

| File | Lines | Role |
|---|---|---|
| `DC3_fitbroad.cpp` | 3,001 | **The historical fitting engine — the code described in W11 §3.** Three generations side by side: amoeba (`amotry_veldisp`, `fitbroad`, `getbac`), LM (`fitbroadmrq`, `mrqabc`), and masked-LM (`fitbprepmask`, `fitbmrqmask`, `fitbmrqmaski`, `mrqabcmask`, `getbtxc`, `getbtsp`, `parsinit`, `setwin`) |
| `DC3_contfit.cpp` | 1,489 | A direct transliteration of the IRAF `onedspec` `continuum` task, credited to AURA (1986). Keeps the IRAF `cv*`/`ic_*` names. Superseded by `libraries/contfit.cpp` |
| `DC3_chimap.cpp` | 756 | `mkchimap`, `chi2grid` — brute-force χ² maps over a parameter grid |
| `DC3_err.cpp` | 720 | `calcsigmats`, `calcsigmatsmask`, `getsig2` ×2, `getinvsig2`, `prepinvsig2` — the Statler covariance in its pre-class form |
| `DC3_dotprod.cpp` | 345 | A brute-force alternative to the FFT path (`autocorrdp`, `crosscorrdp`, `correldp`, `convlvdp`), kept for validation. The code was switched to dot products in March 2005 and then switched back |
| `DC3_fitfunc.cpp` | 344 | `findfit`, `amotry_func` — direct amoeba fitting of an analytic function to a correlation peak |
| `DC3_filt.cpp` | 199 | `cosfilt`, `hifilt`, `lofilt` — FFT frequency filtering |
| `DC3_fiteval.cpp` | 92 | `evalchi`, `evalchinerr` |

### 9.4 `libraries/` — modules linked by DC3

Port carefully (real algorithm content):

| Module | Lines | Content |
|---|---|---|
| `spectrum.cpp` | 4,487 | Log-λ grid conventions, flux-conserving resample, variable-σ convolution driver, masking semantics, S/N, resolution bookkeeping. About half the file is FITS I/O boilerplate |
| `correlation.cpp` | 2,600 | §8.2 |
| `vdf.cpp` | 2,623 | §8.1 |
| `contfit.cpp` | 1,690 | The rejection/iteration *policy* is science; the Cholesky normal-equation machinery is not |
| `fitfunc.cpp` | 1,609 | Gaussian and Gauss–Hermite profiles with analytic derivatives (the NR `mrqmin` callback convention), plus `ghmoment`/`gauss2moment` error propagation |
| `cnvlv.cpp` + `cnvlv.h` | 293 + 321 | §8.4 |
| `specmanip.cpp` | 234 | `xcmask`, `combspec`, `combine_rows` — weighted, velocity-offset stacking |
| `apodize.cpp` | 445 | Only the ~40 lines of window *formulae* matter |
| `nr_2.11.cpp`: `cosfilt`, `hifilt`, `lofilt` | ~150 | **Custom KBW additions, not Numerical Recipes.** Reproduce the filter shape exactly |
| `montecarlo.cpp` | 1,355 | Only the growth-curve and rejection *policy* (~150 lines) is science |
| `stat.cpp` | 1,454 | Only `moment_wgt` and `meanchirej` semantics matter |

Delete outright (NumPy/SciPy/Astropy equivalents):

`myfuncs.cpp` (5,526) · `nr_2.11.cpp` (3,062, minus the three filters) · `myplot.cpp` (2,054) ·
`mywcs.cpp` (1,285) · `fitsfuncs.cpp` (4,189 — but keep the *header conventions*) ·
`fft.cpp` (717) · `param.cpp` (616) · `spline.cpp` (433) · `str_manip.cpp` (353) ·
`ellipse.cpp` (262) · `eigen.cpp` (256) · `autocorrelation.cpp` (224) ·
`term_messages.cpp` (208) · `stopwatch.cpp` (87).

Not used by DC3 at all (verified by grepping every `.lnk` and every `#include` in the dependency
closure): `interp`, `sigfunc`, `lmfitstat`, `nr3_apps`, `linear_set`, `variable_gaussian_kernel`,
`chidist`, `trig`, `distance`, `Rectangle`, `caustic`, `diskgalaxy`, `velfield`, `rotcurve`.
`mrqfunc` and `legfunc` are used only by `test_spec`; `kernel` only by `dither_stack`/
`prob_stack`.

Note: `headers/autocorrelation.h` is a **stale duplicate** — the live `AutoCorrelation` class is
declared inside `headers/correlation.h` (lines 362–375).

---

## 10. Dependencies

### 10.1 External libraries

| Library | Used? | Where | Python replacement |
|---|---|---|---|
| **cfitsio** | ✅ heavily | `libraries/fitsfuncs.cpp` (~500 `fits_*` calls), `mywcs.cpp`, `correlation.cpp` | `astropy.io.fits` |
| **fftw3** | ✅ | **only** `libraries/fft.cpp` (all `fftw_*` calls encapsulated in `FFT_1D`/`FFT_2D`) | `scipy.fft` / `numpy.fft`; `pyfftw` for plan reuse |
| **wcslib** | ✅ narrowly | `libraries/mywcs.cpp` `class FITS_WCS` only. The `class WCS` used elsewhere is hand-rolled and does not use wcslib | `astropy.wcs.WCS` (which *is* wcslib) |
| **gsl** | ✅ | `spline.cpp` (`gsl_interp`, `gsl_bspline`), `cnvlv.cpp` (`gsl_integration_qagi`), `eigen.cpp` (`gsl_eigen_symmv`), `dither_stack.cpp` (`gsl_multimin`) | `scipy.interpolate`, `scipy.integrate.quad`, `numpy.linalg.eigh`, `scipy.optimize.minimize` |
| **pgplot/cpgplot** | ✅ | `myplot.cpp` (~2,000 lines), `DC3_plot.cpp` (2,959), `DC3_express.cpp`, `DC3_main.cpp`, `DC3_fileio.cpp`; ~350 `cpg*` calls | `matplotlib` |
| **Eigen 3** (header-only) | ✅ | `libraries/vdf.cpp` + `headers/vdf.h` **only** — `MatrixXd`, `VectorXd`, `.partialPivLu().solve()`, `.inverse()`, `.block()` | `numpy` + `scipy.linalg` |
| **hdf5** | ❌ | Zero `H5*` symbols in the DC3 closure | — |
| **GPC (gpc232)** | ❌ | Zero `gpc_*` calls in the DC3 closure | — |
| **MPI** | ❌ | `linkmpi` defined but never referenced | — |

**Net: six C libraries collapse to three Python packages** (numpy, scipy, astropy), with
matplotlib for plotting and optional pyfftw.

### 10.2 Numerical Recipes

#### NR 2nd edition — `libraries/nr_2.11.cpp` (3,062 lines, `namespace NR`)

This is the one that matters. Actual call sites:

| NR routine | Called from | Python equivalent |
|---|---|---|
| `mrqmin`, `mrqcof`, `covsrt` | `myfuncs.cpp::fitfuncmrq`, `DC3_fitbroad.cpp` | `scipy.optimize.least_squares(method='lm')` / `curve_fit` |
| `gaussj` | `myfuncs.cpp::invertgj`, `DC3_fitbroad.cpp` | `numpy.linalg.solve` / `inv` |
| `ludcmp`, `lubksb` | `myfuncs.cpp::invertlu` | `scipy.linalg.lu_factor` / `lu_solve` |
| `ran1`, `ran2` | `myfuncs.cpp`, `montecarlo.cpp`, `DC3_fitbroad.cpp`, `DC3_fitfunc.cpp` | `numpy.random.default_rng()` |
| `gasdev` | `DC3_monte.cpp`, `recoverv.cpp` | `rng.normal()` |
| `sort` (6 overloads) | `myfuncs.cpp`, `stat.cpp`, `DC3_fileio.cpp`, `DC3_plot.cpp` | `numpy.sort` / `argsort` |
| `realft`, `four1` | `DC3_filt.cpp`, `spectrum.cpp` | `numpy.fft.rfft` / `irfft` |
| `convlv` | `DC3_dotprod.cpp`, `fitfunc.cpp` | `scipy.signal.fftconvolve` |
| `correl` | `DC3_dotprod.cpp` | `scipy.signal.correlate(method='fft')` |
| `spctrm` | `spectrum.cpp::powspec` | `scipy.signal.welch` |
| `cosfilt`, `hifilt`, `lofilt` | `spectrum.cpp` | **Custom KBW additions, not real NR** — reproduce the filter shape |
| `choldc`, `cholsl` | *hand-copied* into `contfit.cpp` as `chofac`/`choslv` | `scipy.linalg.cho_factor` / `cho_solve`, or `numpy.polynomial.*.fit` |

Declared but never called: `polint`, `ratint`, `spline`/`splint`, `twofft`, `hpsort`, `qromb`,
`savgol`, `svdcmp`, `svdfit`, `lfit`, `fit`, `gammq`, `rk4`, `mnbrak`, `brent`, `zbrent`,
`pythag`, and others. Do not port them.

#### NR 3rd edition — `nr3/`

| Header | Used by | Routines |
|---|---|---|
| `nr3.h` | transitively (types only — see §10.3) | `Doub`, `VecDoub`, `MatDoub`, `SQR`, `MAX`, `MIN`, `SIGN`, `SWAP` |
| `ran.h` | `vec.h:71` (!), `vdf.cpp:57`, `app_siginst`, `prob_stack`, `stack_sim`, `test_spec` | `struct Ran` |
| `deviates.h` | `app_siginst`, `stack_sim`, `dither_stack`, `prob_stack`, `test_spec` | `Normaldev`, `Gammadev`, `Poissondev` |
| `gamma.h` | via `deviates.h` | `gammln`, `Gamma::gammp/gammq` |
| `sort.h` | `stat.cpp`, `montecarlo.cpp` | `sort`, `Indexx`, `select` |
| `interp_1d.h`, `interp_linear.h` | `montecarlo.cpp` | `Linear_interp`, `Spline_interp` |
| `ludcmp.h`, `qrdcmp.h` | via `roots_multidim.h` | `LUdcmp`, `QRdcmp` |
| `roots_multidim.h` | `ellipse.cpp` | `newt`, `broydn`, `lnsrch`, `fdjac` |
| `eigen_sym.h` | `mywcs.cpp` | `Jacobi`, `Symmeig` |
| `pointbox.h` | `myplot.cpp` | `Point<DIM>`, `Box` |

NR3's `fourier.h`, `convlv.h`, `correl.h`, `mins_ndim.h` (amoeba), `fitmrq.h`, `svd.h`,
`gaussj.h`, `fitab.h`, `moment.h`, `plegendre.h` are **not** used — DC3 uses the 2nd-edition
equivalents, or FFTW/GSL/Eigen. Note `mins_ndim.h` (amoeba) *was* used historically: `DC3.h`
still has commented-out `ftol`/`tiny` "for amoeba", and `svanalpar.nmax` is still documented as
*"Maximum number of function calls for amoeba algorithm"*, but the live path is the custom LM in
`VDF::dofit()`.

**Name-collision warning:** `nr3/spectrum.h` and `nr3/weights.h` collide with `headers/spectrum.h`
and `headers/weights.h`. The include order in `make.binary_defs` (`$(inchdr)` before `$(incnr3)`)
makes `headers/` win.

### 10.3 The custom container types — `headers/vec.h`

**The single biggest porting liability, and the one that vanishes most completely.**

`vec.h` (1,772 lines) is *not* NR3's `nr3.h`. It is a hand-maintained fork: it re-declares the
NR3 scalar typedefs (`Int`, `Doub`, `Complex`, `Bool` — lines 41–60), the NR3 macros (`SQR`,
`MAX`, `MIN`, `SIGN`, `SWAP` — lines 74–130), `#include "ran.h"` at line 71 (so **every**
translation unit that touches a vector drags in the NR3 RNG), and three class templates:

| Class | Line | Semantics |
|---|---|---|
| `template<class T> class Vector` | 159 | 1-D; `size()`, `resize(val,n)`, `operator[]`, `append()`, `erase()` |
| `template<class T> class Matrix` | 584 | 2-D, row-pointer array; `nrows()`, `ncols()`, `operator[][]` |
| `template<class T> class Matrix3D` | 1,231 | 3-D |

followed by ~60 typedef triples (1,645–1,702) using the NR2 const-qualification convention:

```cpp
typedef const Vector<double> Vec_I_DP;
typedef Vector<double> Vec_DP, Vec_O_DP, Vec_IO_DP;
typedef const Matrix<double> Mat_I_DP;
typedef Matrix<double> Mat_DP, Mat_O_DP, Mat_IO_DP;
// ... same for INT, LONG, STR, BL, FL, CPLX_DP; Mat3D_* for INT, STR, BL, FL, DP
```

Domain classes add their own: `Vec_SPEC`, `Vec_XC`, `Vec_AC`, `Vec_VDF`, `Vec_MC`, `Vec_PAR`,
`Vec_STAT`, `Vec_WCS`, `Vec_PGL`.

Occurrence counts of `Vec_*`/`Mat_*`/`Mat3D_*` tokens: `dc3/*.cpp` 2,844; all `libraries/*.cpp`
5,735 (~2,000 in the DC3-linked subset); `headers/*.h` 7,797. **Over 10,000 occurrences across
the code that would be touched.** Every one maps to `numpy.ndarray`:

| C++ | Python |
|---|---|
| `Vec_DP v(0.0, n)` | `v = np.zeros(n)` |
| `Vec_I_DP &x` (const in) | `x: np.ndarray` |
| `Vec_O_DP &y` (out param) | **return it** |
| `Mat_DP a(ni, nj)`, `a[i][j]` | `a = np.zeros((ni, nj))`, `a[i, j]` |
| `Vec_BL m` | `np.zeros(n, dtype=bool)` |
| `Vector<Spectrum>` | `list[Spectrum]` |
| `Eigen::MatrixXd` / `VectorXd` | `np.ndarray` 2-D / 1-D |

The `_I_`/`_O_`/`_IO_` out-parameter convention is the main *stylistic* refactor: dozens of DC3
functions return `void` and write through `Vec_O_DP&` arguments. In Python these should return
tuples. This changes every call site — but it is also where most of the remaining boilerplate
(`erase()`, `resize()`, `append()`, `del_row()`, `submatrix()`) evaporates.

---

## 11. File formats

### 11.1 `ParSet` — used by `DC3_express`

`headers/param.h`, `libraries/param.cpp`. Line format:

```
<key>  =  <value(s)>  //<comment>
```

Values may be a `{ a, b, c }` list. Lookup is **by key**, order-independent, via
`par.val("key", out)` overloaded for `string/int/double/bool` with an optional element index.
`ParSet::check(keys)` validates a required-key list.

`progfiles/dc3_express.par` — the complete set of 15 keys `DC3_express` reads:

```
WAVE1     = CRVAL1     Header keyword: initial log wavelength
DISP      = CDELT1     Header keyword: pixel size (d log lambda)
dispaxis  = 1          1 = rows, 2 = columns
padtype   = reflect    wrap | reflect | nearest | constant
contin    = 1.0        continuum level if padtype = constant
maxblk    = 8          maximum block replication for small sigma
taper     = 0          spectral pixels zeroed at each edge
nmax      = 50         maximum LM function calls
fititer   = 5          number of fit restarts (global-minimum check)
miter     = 0          allowed mask-velocity adjustments (-1 => iterate to mvdiff)
mvdiff    = 1.0        mask/fit velocity convergence criterion (negative => pixels)
citer     = 0          maximum continuum-fitting iterations
corder    = 0          Legendre order for an added continuum (0 = none)
funcsmbin = 0          smoothing size for the CC fitting window
noisywin  = 30         minimum window size (important for noisy data)
```

Mapping to W11 §3: `fititer` = tier-1 restarts; `miter`/`mvdiff` = tier 2; `citer`/`corder` =
tier 3; `maxblk` = block replication; `nmax` = the LM iteration cap.

**Hardwired in `DC3_express`, not exposed:** `apwin = 0`, `cosper = 2`, `minsig = 0.85`,
`vwin.pad = 0`, `vwin.winfac = 2.5`, `alambda = 0.001`, `func = GAUSS`, `lam = {1,100,100}`,
`del = {0.01,1,1}`.

### 11.2 Legacy positional format — used by interactive `DC3`

`progfiles/dc3.par`, exactly **40 lines in the order given by `setparname()`**
(`DC3_par.cpp:307`). `readpars` reads line *n* into parameter *n*; **one malformed line reverts
every parameter to `defconsts`**. The full key list with defaults is at `DC3.h:234–277` and is
documented with commentary in `doc/descrip.txt` §1 and `doc/svanal_help.txt` §2.

### 11.3 Spectral-region mask table (`-MS`)

Four whitespace-separated columns, read by `DC3_express::read_mask` (line 673):

```
TYPE      CWL      DL     GROW
OBJECT    5893.0   20.0   1
TEMPLATE  5175.0   30.0   0
```

- `TYPE` ∈ {`OBJECT`, `TEMPLATE`} — the frame in which the region is defined.
- `CWL` — central wavelength (Å); `DL` — full width (Å).
- `GROW` ∈ {`-1`, `0`, `1`} — shrink / fix / grow the width by `2·NSIGMSK·σ_B` when transcribed
  (`NSIGMSK = 2.0`, `correlation.h:114`).

Stored internally as `Mat_DP rfmask(nm, 4)` with `rfmask[i][0] ∈ {TEMPLATE=0, OBJECT=1}`.

### 11.4 Spectrum lists

Interactive `DC3` (`doc/sv_new.inp`, read by `readspecs`):

```
<fits file>                           <object name>   <ms row>  <T|O>
HR6561_F0IIIp_Mg.vs.ncont.log.fits    HR6561_F0IIIp   0         T
N6703_Mg.p2.ncont.ms_log.fits         N6703_p2        1         O
```

`DC3_express` instead uses IRAF-style `@list` batch files, one filename per line, with parallel
lists for `-E`, `-MG`, `-SG`.

### 11.5 FITS I/O

Via cfitsio through `libraries/fitsfuncs.cpp` + `mywcs.cpp`. Entry points:
`ms2spec(msfile, emsfile, mmsfile, rmsfile, spectra, f0, dispaxis, WAVE1, DISP)`,
`fits2spec(...)`, `spec2ms(...)` read and write a multi-spectrum or single-spectrum image plus
optional error, mask, and **resolution** planes.

Requirements on input spectra (`doc/descrip.txt` §2.A): **log-linear in wavelength**, **continuum
normalized to 1**, with `WAVE1`/`DISP` keywords (defaults `CRVAL1`/`CDELT1`). `Spectrum` stores
`l0p`, `dlp` as `log10(λ)` and derives `dvp` from `dlp`.

Output keyword dictionaries are in `doc/keyword.lst` and `doc/headerkey.lst`:

- `*.xc.fits`: `DATEMADE, FILE1, FILE2, POMIT, APOD, COSP, MASK???, MASKF???, MASKV, MASKVS,
  RL_S, RL_E`
- `*.asymxc.fits` / `*.asymbtxc.fits`: the above plus `XCFILE, FOLDV, RMS, NORM, NRMS`
- `*.btsp.fits` / `*.btxc.fits`: plus `VDF, FIXP, FWINT, FWINV1, FWINV2, FITER, FITI0, FITV,
  FITVS, FITH3, FITH4, CHI, RCHIX, RCHIL, CITER, CFUNC, CORDER, CLOR, CHIR, CREJI, CFIT???`

`DC3_express` uses a flatter convention: `<oroot>_gf<i>_t<j>_<kind>.fits` with a companion
`..._<kind>.gpm.fits`; `<kind>` ∈ {`xc`, `xc_asym`, `tpl_rmatch`, `btxc`, `btxc_asym`, `btsp`};
plus `<oroot>_ccstat.db`, `<oroot>_vdf_fits.db`, `<oroot>_fit_plots.ps`.

### 11.6 ASCII databases

Space-delimited, `#`-commented, timestamped, with a two-line header (names, then column numbers).
Consumed by `extperr` (`.gaufit.db`, `.gaucovar.db`, `.anorm.db`, `.anbtxc.db`, `.fitrms.db`) and
`DC3_setflags`. `doc/svanalout.mail` (26 Jul 2007) documents the downstream IRAF
`fields`/`paste`/`convlst` pipeline that motivated `extperr`.

---

## 12. `DC3.h`

637 lines. Include guard is still `_SCKAT_H_` — never renamed.

Constants (40–73): `nsvpar = 40`; `defparfile = "sckat.par"`; `minsigtry = 30`; `maxdim = 2`;
`parray = maxnp` (= 5). Plot geometry: `fxcvp = {0.06,0.98,0.62,0.95}`,
`fspvp = {0.06,0.75,0.34,0.56}`, `ftpvp = {0.06,0.75,0.08,0.30}`,
`fparvp = {0.76,1.0,0.15,0.5}`, `chsize = 0.75`. Print precisions: `Iprec=3, Vprec=2,
VSIGprec=3, H3prec=3, H4prec=3, CHIprec=3, RCHIprec=3, WINVprec=2`. Panel ids:
`INDEFPL=-1, XCPL=0, SPPL=1, TPPL=2, PRPL=3`.

`struct svanalpar` (76–171) — the 40-member monolithic parameter block: `msout, log, pad, padl,
padw, taper, filttap, padtype, contin, lam[5], funclam[5], alambda, par0[5], del[5], nmax, WAVE1,
DISP, dispaxis, apwin, cosperc, sigrej, noisywin, maskwin, mwave, miter, mvdiff, fititer,
cplytype, corder, clor, chir, crejiter, citer, winfac, usebacwin, witer, funcsmbin, minsig,
maxblk, seed`, plus `inline bool equal(...)` for change detection.

`struct svclpar` (175–181) — `speclist, errspeclist, normcorr, outprep`.

`struct stackopt` (199–232) — 26 members, one-to-one with `stack.par`.

Prototype blocks (286–636), still grouped under the original `sv.*.cpp` file names. The
`sv.specmanip.cpp` block (411–427) and the `sv.chimap.cpp` block (531–548) are entirely commented
out.

Top-of-file to-do: **"CHECK ALL ROUTINES TO SEE IF THEY USE (dl/l = v/c) instead of
calcz()!!!!!"** — a correctness concern about non-relativistic vs. relativistic Doppler
conversion. Grep shows `calcz()` *is* used consistently in `DC3_express.cpp`, `DC3_fitbroad.cpp`,
`DC3_plot.cpp`, `DC3_stack.cpp`, `DC3_util.cpp`, `correlation.cpp`, and `spectrum.cpp`, so the
concern appears to have been resolved.

---

## 13. Documentation digest

### `doc/descrip.txt` (12.8 KB, 27 Jan 2006) — the closest thing to a specification

- §1 the legacy parameter file with defaults.
- §2.A input requirements: log-linear binning, continuum normalized to 1, `WAVE1`/`DISP`.
- §2.B convolution prep: pad to a power of two; pad value from `padtype`; errors padded
  identically.
- §2.C correlation prep, in order: taper the ends; subtract the mean (excluding the taper);
  register initial wavelengths (unobserved → zero); pad to a power of two; apply masks and
  re-subtract the mean from unmasked pixels only; apodize (`apwin` 0–6: none, cosine bell, Hann,
  Bartlett, Welch, Blackman, Hamming) block-by-block between masks.
- §2.D masking, with the tier-2 algebra in closed form:

  ```
  w_o = w (σ_o/σ_w) (1+z)^{±1}
  w_o = sqrt(w² ± 2 ln2 (2n)² σ_b²) (1+z)^{±1}
  ```

  with `σ_w = w/(2n√(2ln2))` and `σ_o = sqrt(σ_w² ± σ_b²)`; the negative sign applies when
  converting from the broader object frame, and if `w_o` comes out negative only the redshift
  correction is applied. §2.D.e warns: *"In this scheme, the saved parameters used to adjust the
  mask are from the previous fitting iteration."*
- §3 the fitting algorithm, steps (a)–(k) — a prose version of `VDF::fit()`, including the
  rationale for `(T⊗B)∘T` over `(T∘T)⊗B`.
- §4 the full interactive command list.

### `doc/develop.txt` (7 KB, 10 Mar 2006) — the experiment log

The empirical basis for the hardwired choices:

| Question | Finding |
|---|---|
| Baseline | **Mean subtraction is best** — "most correct recovery of FWHM at all line numbers" |
| Padding | **Subtract the mean, then pad with zeros** |
| Apodization | **None, for low line numbers**; type matters little otherwise — hence `apwin = 0` |
| External continuum | **Normalize to 1** |
| Masking | Four schemes tried; **INCOMPLETE** — "1-3 increase precision but still not great" |
| Errors | MC vs. covariance matrix: "Previously found to be same (2 is much quicker). **Need to redo with new fitting scheme.**" Open: "Are our chi² too small still? ... Covariance between pixels important?" |
| Window | **1.0–1.7 FWHM best (NEED TO REVISIT)** — note `DC3_express` defaults to 2.0 and W11 §3 reports ≈2 |
| Minimizer | Amoeba gives better parameter control but sticks in local minima of a noisy χ²; **LM is faster and more reliable but hard to constrain** — the reason for the parameter clamps in `mrqabc` |
| Minimum dispersion | **σ = 0.85 px (FWHM = 2 px), "makes sense in a Nyquist sense"** — the origin of `minsig` |

It also lists the validation datasets, all on `/d/bebop{2,3}/westfall` (no longer accessible).

### `doc/sckat.todo.readme` (7 KB, 2 Sep 2008) — the issue tracker

Notable open items:

- Libraries: implement and hold a bad-pixel mask for spectra; more exception handling.
- Parameters: force `mwave = true` always and remove the parameter; add a verbose CLI option;
  allow arbitrary parameter ordering in the file.
- Masking: *"What happens in translating the masks when a galaxy is correlated against another
  galaxy? Should that be disallowed?"*
- Fitting: **"Fix chimap routine for new class system"**; determine the covariance matrix when
  directly fitting AC/CC; **"allow fitting for the best template combination"**.
- Added later: **"Incorporate continuum fitting and mask alterations directly into fitting
  iterations, such that so many fitting iterations do not need to be performed!!!!!!!"** — this
  is precisely the v3.0–v3.2 restructuring of `VDF` in January 2013.
- Checks on current algorithms: *"WHY THE HELL DOES IT TAKE SO MUCH RAM TO DO THE CONTINUUM
  FITTING?"*; *"Is the current determination of the errors in the correlation function done
  correctly?"*; *"Is the output window size correct considering inclusion/exclusion of the end
  pixel?"*; *"Does the window have to be odd/even?"* (relevant — `VDF::getwindow` forces an odd
  window at line 542).
- *"Using the convolve function to produce a btxc for a given template will not produce the same
  thing as when producing the btxc from a fit, because the former doesn't have the wavelength
  limitations from the galaxy spectrum."*

### `doc/notes` (23 Jun 2006)

- "blk replicate when sigma is between `minsig*dv` and `minsig*dv/maxblk`"
- "cannot compute `da` happens when the functions found when performing the derivative are
  identical. This gives 0 for some diagonal elements."
- "any fitted sigma below `minsig*dv/maxblk` actually has sigma = 0"
- Lists parameters that are **unused**: `logfile, funclam, samel0, sigrej, usebacwin, winiter`.

### `dc3/notes_improve_dc3` (28 Feb 2013) — three open wishes

1. *"Include a number of 'catch' calls to make sure that the error reporting is sensible."*
2. *"Allow the use to provide a template for each galaxy spectrum."* (Currently the full
   `ngal × ntpl` outer product is computed and fitted.)
3. *"Allow the user to provide their own 'base level' response function that can then be
   convolved with a Gaussian during the fitting process?"*

### `dc3/notes` (28 Feb 2013)

The `SCKAT → DC3` rename map, plus the `DC3_EXPRESS` design brief — which matches the delivered
program, except that "Determine the instrumental-dispersion correction" and
"instrumental-dispersion correction files" were absorbed into `Correlation::match_resolution()`
instead. It also asks: *"Use this program to perform some timing benchmarks"* and *"Update
libraries to improve runtime: autocorrelation, correlation, spectrum, vdf"* — neither of which
appears to have been done.

### `dc3/siginst_algorithm.notes` (14 Jan 2011)

An 18-step procedure for the instrumental-dispersion correction, driving the interactive `DC3`
by **piping a scripted keystroke file** (`sckat < inp`). Inputs (a)–(f); output
`σ_los = sqrt(2·σ_B² + σ_off² − σ_B'²)`. Uses convolution-menu option 17 with `instrconv`'s
`outc`/`outr` files to build a resolution-matched template, then option 4 to broaden it, then a
three-spectrum cross-correlation and VDF fit. **This entire workflow was superseded in February
2013** by `Correlation::match_resolution()` and the `-SG`/`-ST` options of `DC3_express`.

### Other

`doc/keyword.lst`, `doc/headerkey.lst` — FITS keyword dictionaries.
`doc/svanal_help.txt` (24.7 KB, 25 Apr 2005) — per-parameter prose; note its defaults
(`nmax = 5000`, `apwin = 5`) have since moved to 50 and 0.
`doc/plot.readme` — the interactive-command implementation checklist with `DONE` markers.
`doc/namingconv.mail_11.14.06` — an output-naming proposal.
`doc/svanalout.mail` — the downstream database-assembly pipeline.
`doc/sv.funcdepend.txt` (69 KB) — a full function-dependency dump of the v1 code base.

---

## 14. Defects and known issues

Found by reading; none are fixed in the source.

1. **`DC3_express.cpp:215`** — `bool fixed_window = value[7][0].size() > 0;` tests `-v` (index 7)
   instead of `-f` (index 17). **Supplying a guess velocity silently switches the code to `FIXV`
   windowing with `fvs = fve = -1`** instead of the intended `NFWHM`, and `-w` is ignored.

   Tracing the consequence: `VDF::getwindow` (`vdf.cpp:481–493`) computes
   `invs = int((-1 − v0)/dv)`, `inve = invs + 1`. Since `v0 = −(nn/2)·dv`, that places a
   1-pixel window at `v ≈ 0`. It is then below `win.minwin` (= `noisywin`, 30 in production),
   so the fallback recentres it: a **30-pixel window centred on zero velocity lag**, regardless
   of where the CC peak is. `VDF::mrqabc` then clamps the fitted `V` into that window, so for
   any galaxy with `|V| ≳ 15·dv` the velocity is pinned at a window edge.

   **This bug was introduced in v3.1 (5 Mar 2013) and has apparently never been exercised.**
   `oldversions/DC3_express_v3.0.cpp` has `ntags = 18` with no `-f` tag and no `fixed_window`
   variable at all; the February 2013 production campaign (§16) ran v3.0 and used `NFWHM`
   correctly. But every invocation in that campaign passes `-v`, so **re-running any of those
   commands against the current source would silently produce garbage velocities.**
2. **`DC3_express.cpp:212–213`** — `sgfile` reads `value[13][1]` (the `-a` slot) and `stfile`
   reads `value[14][1]` (the `-SG` slot); both are off by one, so `-SG` and `-ST` do not work as
   documented. `readcommandline` (`myfuncs.cpp:3085`) initializes `value` to empty strings and
   fills only `narg[j]` entries per tag, so for the zero-argument `-a` tag `value[13][1]` is
   always `""`. **This bug is present in v3.0** and is confirmed by production failures — see
   §16.4.
3. **`vdf.cpp:487`** — `if (win.inve > nn) win.invs = nn;` should assign `win.inve`. Clamping the
   upper edge of a `FIXV` window instead corrupts the lower edge.
4. **`DC3_express.cpp:1769, 1775, 1781, 1795, 1801, 1807`** —
   `(overwrite(ofile,false) || overwrite(ofile,false))`; the second argument should be `omask`,
   so a pre-existing GPM file is never detected under `-no_overwrite`.
5. **`DC3_express.cpp:767–779`** — `ccpeak_properties` leaves `peakdv2` uninitialized when no
   secondary maximum exists; it is then written to `_ccstat.db` column 14.
6. **`DC3_express.cpp:1809`** — `write_btsp(...)` is called unconditionally, unlike the five
   preceding blocks which are guarded by `no_overwrite`.
7. **`makefile`** — `src` lists `test_conolve.cpp` (typo), plus `DC3_specmanip.cpp` and
   `instrconv.cpp`, both of which have moved out of the directory. `test_int`, `test_convolve`,
   and `test_spec` appear in *both* the active and the commented-out `exe` blocks.
8. **`lnk/{DC3,combspec,recoverv,stack,stack_sim}.lnk`** reference ten `obj/DC3_*.o` files that
   cannot be produced (§1).
9. **`res_base` silently floors the measurable velocity dispersion** (§8.3). Because
   `VDF::getbtsp` convolves by `sqrt(σ_fit² − base_sig²)`, any spectrum with a true dispersion
   below `res_base` is driven to σ ≡ 0 and re-fitted, rather than being reported as an upper
   limit. Production `RES_BASE` values reached 9.50 km/s (§16.4). This is arguably a design
   choice rather than a bug — in the DMS it was read as an observational limitation, since a
   dispersion below the floor was not measurable with the data in hand — but it is undocumented
   in both the code and the paper, and the affected spectra are indistinguishable in the output
   from genuine σ = 0 measurements.
10. **Unresolved scientific questions** carried in the documentation: whether the CC-function
   errors are determined correctly; whether χ² is systematically too small; whether inter-pixel
   covariance from resampling is being double-counted; whether Monte Carlo and covariance-matrix
   errors remain equivalent under the post-2013 fitting scheme; whether the fit window should be
   1.0–1.7 FWHM (the 2006 finding) or ≈2 FWHM (the shipped default).

---

## 15. Implications for the port

### What must be ported carefully (~11,000 lines of C++)

`vdf.cpp`, `correlation.cpp`, `cnvlv.cpp` + `cnvlv.h`, the scientific half of `spectrum.cpp`, the
rejection policy of `contfit.cpp`, the Gauss–Hermite parameterization in `fitfunc.cpp`, the three
custom filters in `nr_2.11.cpp`, `specmanip.cpp`, the window logic in `DC3_win.cpp`/
`VDF::getwindow`, and `DC3_express.cpp` as the pipeline specification.

### What disappears (~20,000 lines)

`vec.h` (1,772) · `myfuncs.cpp` (5,526) · `nr_2.11.cpp` (3,062) · the PGPLOT/terminal UI
(`myplot` 2,054 + `DC3_plot` 2,959 + `DC3_menu` 3,313 + `DC3_userio` 1,638) · `mywcs.cpp` (1,285)
· `fft.cpp` (717) · `param.cpp` (616) · `spline.cpp` (433) · `str_manip.cpp` (353) ·
`eigen.cpp` (256) · `stopwatch.cpp` (87) · all of `nr3/` · all of `gpc232/`.

### Licensing

Removing Numerical Recipes removes the barrier to open-sourcing. DC3's NR usage reduces to LM
fitting, linear algebra, sorting, RNGs, and FFT helpers — all of which have direct, permissively
licensed replacements in NumPy/SciPy. The three "NR" filter routines DC3 actually depends on
(`cosfilt`, `hifilt`, `lofilt`) are the author's own additions to `nr_2.11.cpp`, not NR code.

### Performance

Two identified bottlenecks, both structural rather than incidental:

1. **Finite-difference derivatives** in `VDF::mrqabc` — `1 + 2·mfit` forward-model evaluations
   per LM step, each a convolution plus a full FFT correlation.
2. **Per-pixel adaptive quadrature** in `cnvlv_integral` for the variable-σ resolution-matching
   convolution.

Plus two smaller ones: the explicit `.inverse()` of the covariance sub-block inside the fit loop,
and the absence of any parallelism despite the problem being embarrassingly parallel over
(galaxy × template) pairs.

### Validation

There is no live C++ oracle (§1). The available anchors are:

- `test_convolve.out`, `test_siginst.fits`, `test_siginst_log.fits` — committed reference outputs
  for the variable-σ convolution.
- **The Monte Carlo simulations of W11 §4** — 240 simulation sets over
  ⟨S/N⟩ ∈ {1, 1.4, 2, 2.8, 4, 8, 16, 64}, `V_m` ∈ {1000, 2500, 3600, 4160, 4640} km/s,
  `σ_m` ∈ {5, 10, 20, 40, 80, 110} km/s, 50 realizations each. These are fully synthetic and
  self-contained, and their results are published in Figures 4 and 5 with seven numbered findings
  in §4 — a reproducible acceptance test.
- **The `pPXF` comparison of W11 §4.1 and §5.1** (Figure 6, Table 1). `ppxf` is actively
  maintained and installable, so this comparison can be repeated directly.
- **The UGC 6918 demonstration of W11 §5** (Figures 7–9, Table 1) — ✅ **located**, at
  `/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/` (§16.8), together with the K1 III
  template HD 167042 (= HR 6817), the June 2010 `ppxf` comparison of §5.1, and the Appendix A
  instrumental-dispersion workflow. Table 1's published `V` and `σ` are build-independent and
  are the strongest available acceptance test; note they report `σ_obs`, *not* the
  instrument-corrected `σ_LOS`.
- **Real production inputs and outputs on `/Volumes/seshat`** (§16). `PPK_rdx/*/vdf_feb13/` gives
  per-galaxy inputs, a production parameter file, a mask table, and the corresponding
  `_ccstat.db` / `_vdf_fits.db` / FITS products for ~30 galaxies — a regression corpus for the
  `_od` (no instrumental-dispersion) path. The `_cd` products are **not** usable as references
  (§16.4).

### Features the port should add

From `notes_improve_dc3` and `sckat.todo.readme`, the author's own long-standing requests:

- Per-galaxy templates instead of the full `ngal × ntpl` outer product.
- **Fitting for the best template combination** — i.e. optimal mixing of a template library.
- A user-supplied base response function convolved with the Gaussian during fitting — i.e. an
  **empirical instrumental LSF** in place of the Gaussian assumption. **Low priority.** The
  port should not *preclude* this, but need not implement it now; see the port plan, Phase 3,
  for the seam that keeps it cheap to add later.
- Sensible exception handling throughout.
- Timing benchmarks.

---

## 16. Cross-check against a production run

Everything above was derived from source. This section checks it against a real campaign:

- **Driver script:** `/Volumes/seshat/data/diskmass/PPK_rdx/feb13_dc3_inst_03.scr` (11 Feb 2013),
  one of a numbered series (`feb13_dc3_inst_01.scr`, `_02`, `_03`, …).
- **Example output:** `/Volumes/seshat/data/diskmass/PPK_rdx/U06918/vdf_feb13/` (291 MB).

This is **PPak** (PMAS) data from the DiskMass Survey, not the SparsePak data of W11 §5, and the
template is **HR 6654**, not the K1 III HD 167042 used in the paper. So it is a Paper IV-era
campaign exercising the same software.

### 16.0 ⚠️ The archived outputs were *not* produced by the code now on disk

The campaign ran 11–14 February 2013. Comparing that against source modification times shows
that **most of the algorithmic core was edited afterwards** — in some cases by years:

| File | Modified | Relative to the campaign |
|---|---|---|
| `headers/fitfunc.cpp` → `libraries/fitfunc.cpp` | 10 Jan 2013 | before ✅ |
| `libraries/contfit.cpp` | 28 Jan 2013 | before ✅ |
| `headers/cnvlv.h` | 8 Feb 2013 | before ✅ |
| `headers/correlation.h` | 9 Feb 2013 | before ✅ |
| `libraries/correlation.cpp` | 28 Feb 2013 | **+2 weeks** ❌ |
| `libraries/fft.cpp` | 15 Mar 2013 | **+1 month** ❌ |
| `dc3/DC3_express.cpp` (v3.0 archived 5 Mar 2013) | 26 Jul 2016 | **+3.4 years** ❌ |
| `libraries/vdf.cpp`, `headers/vdf.h` | 9 Dec 2014 | **+22 months** ❌ |
| `libraries/cnvlv.cpp` | 24 Dec 2016 | **+3.9 years** ❌ |
| `libraries/spectrum.cpp`, `headers/spectrum.h` | 29 Dec 2016 | **+3.9 years** ❌ |

So the fitter, the spectrum class, the convolution, the correlation, the FFT wrapper, and the
driver have all changed since these products were written. Worse, `vdf.h`'s own version history
**ends at v4.0 (8 Feb 2013)** even though `vdf.cpp` and `vdf.h` were modified on 9 Dec 2014 —
meaning there are changes to the fitting core that the in-file changelog does not record. The
version log is therefore not a complete account of the code's evolution.

There are at least **three distinct versions** of DC3 in play, and none of them is reproducible:

1. **February 2013** — produced the archived data. Source not preserved as a snapshot; the
   nearest artifact is `oldversions/DC3_express_v3.0.cpp` (archived 5 Mar 2013), which is what
   the working file became *after* the campaign.
2. **June 2015** — `scripts/bin/DC3_express`, the surviving binary. Predates the Dec 2016
   `spectrum.cpp`/`cnvlv.cpp` edits, so it matches neither (1) nor (3). It does not run.
3. **Jul/Dec 2016** — the current source, which is what §§1–15 of this report describe.

**Consequence for validation.** The archived **inputs** (galaxy and template spectra, mask table,
parameter file) are version-independent and remain fully valuable. The archived **outputs** are
only an indicative reference: a disagreement between the Python port and these tables is
evidence of *a* difference, but cannot be attributed to the port without first establishing
which version produced the table. Do not set tight numerical regression tolerances against them.
The reproducible specification is Westfall et al. (2011) §4, whose Monte Carlo acceptance
criteria are stated in the paper and do not depend on any particular build.

### 16.1 The invocation pattern

Each galaxy gets four runs, distinguished by two binary choices encoded in the output root:

```
DC3_express -P dc3_express.par -G U06918_merge_c.fits -E U06918_merge_c.err.fits \
            -T HR6654.c_log.fits -MS tplmask.db -v 1140 70 -O U06918_c4_an_od -w 2.0 -a
```

| Suffix | Meaning |
|---|---|
| `an` | **a**nalytic **n**ormalization — `-a` given, so `VDF::calc_norm` solves `I` in closed form |
| `fn` | **f**itted **n**ormalization — `I` is a free LM parameter |
| `od` | **o**riginal **d**ispersion — no `-SG`/`-ST` |
| `cd` | **c**orrected **d**ispersion — `-SG`/`-ST` given |

Eight galaxies per script (`U04380`, `U04458`, `U04555`, `U04622`, `U06903`, `U06918`, `U07244`,
`U07917`), guess velocities 1140–12500 km/s, guess σ = 70 km/s throughout, window `-w 2.0`.

### 16.2 Confirmed without change

*Read these in light of §16.0: they confirm that the described behaviour was present in the
February 2013 build. For the file formats, naming conventions, and numerical identities below
that is strong evidence the description is right, because these are exactly the things unlikely
to have changed. It is weaker evidence for anything touching the fitter.*

| Claim | Evidence |
|---|---|
| Output naming `<oroot>_gf<i>_t<j>_<kind>.fits` with `.gpm.fits` companions (§6.6) | `U06918_c4_an_od_gf1_t1_{xc,xc_asym,tpl_rmatch,btxc,btxc_asym,btsp}.fits` + all six `.gpm.fits` |
| Three text/PS products (§6.6) | `_ccstat.db`, `_vdf_fits.db`, `_fit_plots.ps` (93 MB) |
| `_ccstat.db` has 16 columns (§6.3) | Header and data rows match the documented list exactly |
| `_vdf_fits.db` column layout (§6.6) | 27 columns: with `corder = 4`, the `[C1..Ccorder CC]` block expands to `C1 C2 C3 C4 CC`, giving `8 + 5 + 14 = 27` |
| `Ie = −1` when `-a` is used (§8.1) | `an` runs: `Ie = -1` in every row. `fn` runs: real values (e.g. `8.31e-03`) |
| Mean subtraction over unmasked pixels (§8.2) | `_ccstat.db` `MEAN` column ≈ `1e-16` — machine zero |
| Zero-lag regions trimmed from CC statistics (§6.3) | `NU = 3888` of `nn = 4508` pixels |
| `v0p = −(nn/2)·dvp` (§8.2) | XC header `CRVAL1 = −24823.7478`, `CD1_1 = 11.0132`; `−24823.7478/11.0132 = −2254 = −4508/2` |
| `dvp` derived from `dlp` | Input `CD1_1 = 1.5954274e-05` in log₁₀λ → `c·ln(10)·dlogλ = 11.013 km/s`, matching the XC `CD1_1` |
| Region-mask table format (§11.3) | `tplmask.db` is exactly `TEMPLATE 5006.84 5.0 1` / `TEMPLATE 5197.90 5.0 1` / `TEMPLATE 5200.26 5.0 1` — i.e. [O III] λ5007 and [N I] λλ5198, 5200, 5 Å wide, `GROW = 1`. These are the same lines masked in W11 §5 and Figure 2. Note they are defined in the **template (rest) frame** and transcribed into the galaxy frame by `(1+z)`, exactly as §8.2's `adjmask` describes |
| `ParSet` is key-based and order-independent (§11.1) | The production file omits `dispaxis`-adjacent keys in a different order than the `progfiles` template and still works |
| One resolution-matched template per galaxy spectrum | `tpl_rmatch` is 352 × 2048 for a single 1-D input template — the template is re-prepared per spectrum because the mask and common-wavelength range differ |

### 16.3 Corrections to this report

**(a) The correlation length is not a power of two.** Corrected in §8.2. The production XC is
4508 pixels for 2048-pixel inputs, which reproduces `nn = int(2.2·(int((ln−l0p)/dlp)+1))`
exactly. `fft.cpp` plans are `fftw_plan_dft_r2c_1d`/`c2r` with **`FFTW_ESTIMATE`**, so the code
already uses real-to-complex transforms on arbitrary lengths.

**(b) The `fixed_window` bug post-dates this campaign.** Corrected in §14.1. The bug requires the
`-f` tag, which first appears in v3.1; the working file at the time was the state later archived
as `oldversions/DC3_express_v3.0.cpp` (`ntags = 18`, no `-f`, no `fixed_window`). So these runs
used `NFWHM` with `-w 2.0` as intended — corroborated directly by the fitted velocities
clustering at 1050–1155 km/s around the `-v 1140` guess rather than being pinned near zero lag.
This dating is inference from the archived version files plus the output behaviour, not from a
preserved build record.

**(c) The production parameter file differs substantially from the `progfiles` template**, and
carries one key the code ignores:

| Key | `progfiles/dc3_express.par` | `U06918/vdf_feb13/dc3_express.par` |
|---|---|---|
| `DISP` | `CDELT1` | **`CD1_1`** |
| `fititer` | 5 | **2** |
| `miter` | 0 | **−1** (iterate the mask to convergence) |
| `mvdiff` | 1.0 | **−0.10** (negative → 0.1 *pixel*) |
| `citer` | 0 | **10** |
| `corder` | 0 | **4** |
| `padw` | *absent* | **0** — present in the file but **inert**: `set_vdf_fit_win` has `par.val("padw", vwin.pad)` commented out and hardwires `vwin.pad = 0` |

So the shipped template's defaults are *not* the production configuration: in production the
continuum (tier 3) and mask (tier 2) iterations are both fully enabled, while the tier-1 restart
count is *reduced* from 5 to 2. The port's ParSet defaults should follow the production file, not
`progfiles`.

### 16.4 The instrumental-dispersion runs failed

For `U06918`, both `-SG`/`-ST` runs produced **no output at all**. The complete spool files are:

```
$ cat spool_c4_an_cd spool_c4_fn_cd
ERROR: The size of the resolution and ms data is different.
ERROR: The size of the resolution and ms data is different.
```

(the message is raised at `libraries/spectrum.cpp:3874`, inside the multi-spectrum reader).

Two independent causes are in play:

1. **The §14.2 argument off-by-one**, which is present in v3.0. `sgfile` resolves to `""` (the
   zero-argument `-a` slot is always empty), and `stfile` receives the `-SG` argument — i.e. the
   **galaxy's** instrumental-dispersion file is handed to the **template**.
2. **A file-shape mismatch independent of the bug:** `U06918_merge.instrdisp_log.fits` is
   352 × **2047**, whereas `U06918_merge_c.fits` is 352 × **2048**. Even with correct argument
   handling this would trip the same check.

Across the whole campaign the outcome is mixed. Galaxies whose σ_inst file has the same pixel
count as the data (e.g. `U00448_r3_7h_sigmaInst_log.fits`, 331 × 2048) **did** complete, with
non-zero `RES_BASE` (0 to 9.50 km/s), so `Correlation::match_resolution` ran. Galaxies whose
σ_inst file is a pixel short (U06918, and others in `feb13_dc3_inst_03.scr`) aborted.

> **Unresolved.** For the runs that completed, I could not establish from the outputs alone which
> file ended up as which resolution vector. Two simple hypotheses — that the template received
> row 0 of the galaxy file, or that the arguments behaved as documented — both predict `RES_BASE`
> values (1.00 and 2.87 km/s for `U00448` GID 1) that disagree with the observed 6.62 km/s. The
> discrepancy is plausibly explained by the Doppler shift and common-wavelength restriction
> applied inside `match_resolution`, which my hand calculation ignored. **Resolving this needs a
> controlled re-run, which is not possible without rebuilding the C++.** Treat any
> `*_cd_*` product from this campaign as provenance-uncertain, and do not use it as a validation
> reference for the port's instrumental-dispersion path.

### 16.5 Performance baseline

From `U06918` (352 spectra, 1 template, 2048 pixels, `nn = 4508`, single-threaded, 2013 hardware):

| Quantity | `an_od` | `fn_od` |
|---|---|---|
| Wall-clock for the whole run | **47 m 39 s** | — |
| Σ of per-fit times (column `T`) | 2826 s (0.78 hr) | 3747 s (1.04 hr) |
| Median per-fit time | 6.5 s | 8.8 s |
| Fits attempted | 352 / 352 | 352 / 352 |
| `FC` (LM converged) | 344 | 338 |
| `MC` (mask converged) | 344 | 338 |

Three things follow:

- **~8.1 s of wall time per spectrum**, of which the fit itself is ~59%; the remainder is CC
  construction, the 93 MB PostScript QA file, and FITS I/O.
- **Solving the normalization analytically is ~25% faster** than fitting it (2826 s vs 3747 s) —
  direct support for making `-a` the default and for generalizing `calc_norm` to the multi-template
  weight solve rather than adding free parameters.
- The whole campaign is embarrassingly parallel and was run serially.

### 16.6 Two scientific observations

**Median reduced χ²_X is 0.82 (`an`) and 0.85 (`fn`) — systematically below 1.** This is direct
empirical support for the open question recorded in `doc/develop.txt`: *"Are our chi² too small
still? If so, could resampling the data be causing overestimate of errors? Covariance between
pixels important?"* The port should treat this as a live issue, not a settled one.

**Failed fits are flagged by a `−1` sentinel** in `V` and `VSIG`, not by `FE`. Across the 352
U06918 spectra, `V` spans −1 to 6852 km/s about a guess of 1140, so the tails are unphysical
outliers that downstream analysis must reject; `FC`/`MC` are the useful discriminants.

### 16.7 Data located for the port

`PPK_rdx/` holds `vdf_feb13/` directories for ~30 galaxies, plus earlier `vdf_aug09/`,
`vdf_may10/`, `vdf_jun10/`, `vdf_nov10/`, and `vdf_jan13/` campaigns — a substantial
regression corpus.

### 16.8 ✅ The W11 §5 demonstration data — found

§15 listed the UGC 6918 SparsePak demonstration data (Figures 7–9, Table 1) as not found, and
§16.7 originally guessed at `/Volumes/seshat/data/diskmass/SPSPK_rdx/`. **That guess was wrong
about the path.** The data is at

```
/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/     (78 MB, Nov 2008 – Jan 2009)
```

— one level below the mount point, *not* under `data/diskmass/`. The identification is solid:

| File | Significance |
|---|---|
| `HR6817_K1III_Mg_log.fits`, `.err.fits` | **HR 6817 = HD 167042**, the K1 III template named in W11 §5 |
| `U06918_merge.mg.{me,ms,msn}_smc_log.fits` | UGC 6918, **Mg** region, SparsePak merged spectra (`me`/`ms`/`msn` = the error, science, and S/N-normalized variants) |
| `U06918_merge.msn.stat` | per-fiber S/N statistics |
| `ppxf_jun10/U06918_merge.ppxf.db` | the **ppxf comparison** of W11 §5.1, run June 2010 |
| `sircorrect_feb10/U06918_instrs_{01..18}.db` | eighteen files, one per step of the **18-step instrumental-dispersion workflow** documented in `siginst_algorithm.notes` (W11 Appendix A) |
| `inp`, `inp_refit` | the **keystroke scripts** that drove the interactive `DC3` binary |
| `U06918_vdf{,_refit,_final}.*.db` | per-stage outputs: `gaufit`, `gaucovar`, `vdfit`, `anorm`, `anbtxc`, `cntcoeff`, `fitrms`, `comb` |
| `xc/`, `btxc/`, `btsp/`, `cont/`, `asymxc/`, `asymbtxc/` | per-spectrum function products, one file per fiber |

**This run used the interactive `DC3` binary, not `DC3_express`** — §7's program, one of the
eight that no longer compile (§5). So the output layout described in §11 does not apply here:
many small `.db` files and per-stage subdirectories rather than two tables and six FITS files.

`inp` is worth reading in full as an independent check on the report. It confirms, purely from
the archived run and without reference to source: `tplmask.db` as the region-mask table
(matching §11.3); a velocity guess of `1110 60` km/s; a window factor of `2.0` (the `NFWHM`
default of §9.1); and an explicit good-fiber list
(`2,5,9,16,19,21,22,24-37,39-50,52-54,56,57,59,61-67,69,70,72,74-78,80,82,83`) — i.e. **fiber
selection in the paper was manual**, not an automatic S/N cut.

**Why this matters for the port.** Everything in §16 carries the §16.0 provenance problem: the
archived outputs cannot be attributed to any identifiable build. This directory has the same
problem — worse, if anything, since the code path is older still. But it is the input to a
*published* result, and W11 Table 1 gives `V` and `σ` for fibers 52 and 55. Those numbers are
in the literature, are independent of which binary wrote them, and can therefore anchor a
genuinely gating test. See the port plan's Test data and Verification sections.

`/Volumes/seshat/SPSPK_rdx/kinematics/mg/` contains many further galaxies, and there are
sibling trees under `/Volumes/seshat/SPSPK_rdx/` — a larger SparsePak archive than assumed.

---

## 17. Change log

- **2026-09-15** — Initial report, written from the source tree at
  `/Users/westfallold/Work/scripts/dc3` and its dependency closure, plus Westfall et al. (2011).
- **2026-09-15** — Cross-checked against the February 2013 PPak production campaign
  (`feb13_dc3_inst_03.scr`, `U06918/vdf_feb13/`). Added §16. Corrected §8.2: the correlation
  length is **not** constrained to a power of two (`setsize()` uses a 2.2× factor and enforces
  only evenness); the header comment, `doc/descrip.txt`, and W11 §2.4 are stale on this point.
  Corrected §14.1: the `fixed_window` bug was introduced in v3.1 (Mar 2013), after the campaign,
  and has never been exercised in production. Strengthened §14.2 with the `readcommandline`
  mechanism and the production evidence. Added §16.3(c) noting that production parameter values
  differ substantially from the `progfiles` template. Added a performance baseline (§16.5) and
  the χ² < 1 observation (§16.6). Noted in §16.7 that SparsePak trees exist, partly resolving the
  missing-demo-data item in §15.
- **2026-09-15** — Added §16.0 after it was pointed out that the code on disk may not be the code
  that produced the archived data. Source modification times confirm this: `vdf.cpp` (Dec 2014),
  `spectrum.cpp` and `cnvlv.cpp` (Dec 2016), `DC3_express.cpp` (Jul 2016), `correlation.cpp` and
  `fft.cpp` (Feb/Mar 2013) all post-date the 11–14 Feb 2013 campaign, and `vdf.h`'s changelog
  stops at Feb 2013 despite the file being edited in Dec 2014 — so the in-file version history is
  incomplete. At least three mutually inconsistent versions exist (Feb 2013 source, Jun 2015
  binary, 2016 source), none rebuildable. Downgraded the archived **outputs** from a regression
  reference to an indicative one, while keeping the archived **inputs** as fully trustworthy.
  Added qualifiers to §16.2 and restated §16.3(b)'s version dating as inference rather than
  record.
- **2026-09-15** — ✅ **Located the W11 §5 demonstration data.** New §16.8 documents
  `/Volumes/seshat/SPSPK_rdx/kinematics/mg/U06918/vdf_dec08/` — one level below the mount
  point, not under `data/diskmass/` as §16.7 had guessed, which is why earlier searches missed
  it. Identified by `HR6817_K1III_Mg_log.fits` (HR 6817 = HD 167042, the paper's K1 III
  template), the `ppxf_jun10/` comparison of W11 §5.1, and `sircorrect_feb10/`'s eighteen
  `U06918_instrs_NN.db` files matching the 18-step Appendix A workflow. Noted that this run
  used the **interactive `DC3` binary**, so §11's output layout does not describe it, and that
  the `inp` keystroke script independently confirms `tplmask.db`, the `1110 60` velocity guess,
  the 2.0 window factor, and — a new finding — that the paper's **fiber selection was manual**.
  Trimmed §16.7 accordingly. Also marked the third bullet of "Features the port should add"
  (user-supplied base response function, i.e. an empirical instrumental LSF) as **low
  priority**: not to be precluded, not to be implemented now.
- **2026-09-16** — Three edits arising from `plan-reassessment.md`. Added a warning block to
  **§8.3** recording a consequence of `res_base` that is documented nowhere in the code or the
  paper: because `VDF::getbtsp` convolves by `sqrt(σ_fit² − base_sig²)`, `res_base` cancels
  exactly in the model but imposes a **hard floor on the measurable σ**, below which the fitter
  sets σ ≡ 0 and re-fits — so affected spectra are indistinguishable in the output from genuine
  zero-dispersion measurements. Production `RES_BASE` reached 9.50 km/s against W11 §4
  simulations probing to 5 km/s, and the floor is distinct from §8.4's block-replication limit.
  Added the corresponding entry as **§14.9** (renumbering the open-questions item to §14.10).
  Updated **§15 "Validation"**, which still described the W11 §5 SparsePak data as unlocated and
  pointed at the wrong directory; §16.8 supersedes it, and the entry now notes that Table 1
  reports `σ_obs` rather than the corrected `σ_LOS`. Removed a stray doubled horizontal rule
  between §15 and §16.

---

*Prepared from the source tree as of its last modification (26 Dec 2016 for `DC3_util.cpp` and
`stack.cpp`; 26 Jul 2016 for `DC3_express.cpp`), and cross-checked against the February 2013
production campaign on `/Volumes/seshat`.*
