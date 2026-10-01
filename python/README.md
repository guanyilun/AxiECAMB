# Cobaya wrapper for AxiECAMB

AxiECAMB has no Python bindings, so `axiecamb.py` wraps the compiled `camb`
executable as a [cobaya](https://cobaya.readthedocs.io) `Theory` class: each
likelihood evaluation writes a `params.ini` into a scratch directory, runs the
binary, and parses the output Cl files. One evaluation takes ~1 s on a laptop.

## Setup

1. Compile the code: `make` in the repo root (serial; `make -j` has dependency races).
2. Install cobaya (`pip install cobaya` or `pip install -e /path/to/cobaya`).
3. Point cobaya at the wrapper:

```yaml
theory:
  axiecamb.AxiECAMB:
    python_path: /path/to/AxiECAMB/python
```

See `example.yaml` for a full input file and `test_axiecamb.py` for a
programmatic check (it also serves as a template for a likelihood that
consumes the spectra).

## Provided products

- `Cl` (lensed): `tt`, `ee`, `bb`, `te`, `eb`, `pp`, `tp`, `ep`
- `unlensed_Cl`: `tt`, `ee`, `bb`, `te`, `eb`
- Derived parameters parsed from the feedback output:
  `age`, `zstar`, `rstar`, `theta_s_100`, `theta_MC_100`, `zdrag`, `rdrag`, `omegam`

Conventions follow cobaya's `BoltzmannBase`: `get_Cl()` returns raw `C_l` in
FIRAS-calibrated muK^2 by default.

Caveats:

- The lensing convolution does not act on the rotation-induced scalar B mode,
  so the wrapper returns `bb` = lensed BB + unlensed rotation BB, and `eb` is
  the unlensed scalar EB. TB is not computed by the Fortran code.
- Matter power spectra / transfer functions are not exposed (the code's
  `get_transfer` output exists, but `Pk_interpolator` etc. are not wired up).
- `sigma8` is not available since transfer output is disabled during sampling.

## Input parameters

Cosmological: `ombh2`, `omnuh2`, `omk`, `H0`, `tau`, `As`, `ns`, `nrun`,
`yhe`, `w`, `cs2_lam`.

Axion sector: `m_ax` (eV) or `log10_m_ax`, `g_axion` (= g_agamma * M_pl),
and depending on the `use_axfrac` option either (`omdah2`, `axfrac`) or
(`omch2`, `omaxh2`). Isocurvature: `Hinf`, with the `isocurvature` option below
(the amplitude is derived internally from `Hinf` and the initial field value).

Anything else in `params.ini` can be fixed through `extra_args` (raw ini keys),
e.g. `extra_args: {accuracy_boost: 1.5, massless_neutrinos: 2.044}`.

Defaults chosen to match CAMB under cobaya:
- `yhe` is BBN-consistent (CAMB's `camb.bbn` predictor, from `ombh2` and
  N_eff) unless `yhe` is given as a parameter or `helium_fraction` is set in
  `extra_args`; this needs the `camb` Python package.
- N_eff = 3.044 (`massless_neutrinos: 2.044` plus one massive species).
- With lensing, `k_eta_max_scalar` is at least 18000 (CAMB's
  `lens_potential_accuracy: 1`); `2 * l_max_scalar` alone leaves C_L^phiphi
  ~10% low at L ~ 2500.
- Lensing is linear (`do_nonlinear: 0`) unless set in `extra_args`. CAMB's
  cobaya default is nonlinear (Mead2020), which AxiECAMB does not have;
  `do_nonlinear: 2` with `halofit_version: 4` (Takahashi) is the closest, but
  halofit is not calibrated for strong axion suppression.

## Wrapper options

| option | default | meaning |
|---|---|---|
| `path` | repo root | directory containing the `camb` binary |
| `use_axfrac` | `true` | parametrize axion abundance by (`omdah2`, `axfrac`) |
| `lensing` | `true` | compute lensed Cls and the lensing potential |
| `accurate_bb` | `false` | sets `accurate_BB` in the ini (slower) |
| `lmax_margin` | `250` | extra ells computed beyond the requested lmax |
| `num_threads` | `0` | OpenMP threads per evaluation (0 = all cores) |
| `timeout` | `600` | seconds before an evaluation is declared failed |
| `run_dir` | none | keep run files here instead of a temp dir (debugging) |
| `isocurvature` | none | `adi`, `both` (adiabatic + axion isocurvature, lensed together) or `iso` (isocurvature only); none leaves it to `extra_args` |
| `extra_args` | `{}` | raw `params.ini` overrides (e.g. `movH_switch`) |

Failed or timed-out evaluations are reported to cobaya as invalid points
(rejected by the sampler) rather than crashing the chain.

The wrapper always sets `write_aniso_transfer = F` so the per-(k, tau)
anisotropic source file (`aniso_source_k_tau_*.dat`) is not written during
sampling; standalone `./camb params.ini` runs still write it by default.

For MPI runs, note that each chain spawns its own `camb` process; set
`num_threads` to roughly (cores / number of chains) to avoid oversubscription.
