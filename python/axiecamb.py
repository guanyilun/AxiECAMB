"""
Cobaya Theory wrapper for AxiECAMB: runs the ini-driven Fortran executable
once per likelihood evaluation and parses its output Cl files.
See README.md in this directory for usage, options and caveats.
"""

import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

from cobaya.log import LoggedError
from cobaya.theories.cosmo import BoltzmannBase

_T_FIRAS_MUK = 2.7255e6

PARAM_TO_INI = {
    "ombh2": "ombh2",
    "omch2": "omch2",
    "omnuh2": "omnuh2",
    "omk": "omk",
    "H0": "hubble",
    "w": "w",
    "cs2_lam": "cs2_lam",
    "yhe": "helium_fraction",
    "tau": "re_optical_depth",
    "As": "scalar_amp(1)",
    "ns": "scalar_spectral_index(1)",
    "nrun": "scalar_nrun(1)",
    "m_ax": "m_ax",
    "omaxh2": "omaxh2",
    "omdah2": "omdah2",
    "axfrac": "axfrac",
    "g_axion": "g_axion",
    "Hinf": "Hinf",
}

_BASE_INI = {
    "get_scalar_cls": True,
    "get_vector_cls": False,
    "get_tensor_cls": False,
    "get_transfer": False,
    "do_lensing": True,
    "do_nonlinear": 0,
    "halofit_version": 1,
    "l_max_scalar": 2700,
    "k_eta_max_scalar": 6000,
    "use_physical": True,
    "ombh2": 0.0224,
    "omch2": 0.12,
    "omnuh2": 0.6451439e-3,
    "omk": 0.0,
    "hubble": 67.36,
    "w": -1,
    "cs2_lam": 1,
    "m_ax": 1e-27,
    "use_axfrac": True,
    "omaxh2": 0.12,
    "omdah2": 0.12,
    "axfrac": 1.0,
    "g_axion": 0.0,
    "axion_isocurvature": False,
    "alpha_ax": 0,
    "Hinf": 13.7,
    "temp_cmb": 2.7255,
    "helium_fraction": 0.24,
    "massless_neutrinos": 2.046,
    "massive_neutrinos": 1,
    "share_delta_neff": True,
    "nu_mass_eigenstates": 1,
    "nu_mass_fractions": 1,
    "initial_power_num": 1,
    "pivot_scalar": 0.05,
    "pivot_tensor": 0.05,
    "scalar_amp(1)": 2.196e-9,
    "scalar_spectral_index(1)": 0.9655,
    "scalar_nrun(1)": 0,
    "tensor_spectral_index(1)": 0,
    "initial_ratio(1)": 0,
    "tens_ratio": 0,
    "reionization": True,
    "re_use_optical_depth": True,
    "re_optical_depth": 0.05,
    "re_delta_redshift": 0.5,
    "re_ionization_frac": -1,
    "RECFAST_fudge": 1.14,
    "RECFAST_fudge_He": 0.86,
    "RECFAST_Heswitch": 6,
    "RECFAST_Hswitch": True,
    "initial_condition": 1,
    "vector_mode": 0,
    "COBE_normalize": False,
    "CMB_outputscale": _T_FIRAS_MUK**2,
    "scalar_output_file": "scalCls.dat",
    "vector_output_file": "",
    "tensor_output_file": "",
    "total_output_file": "",
    "lensed_output_file": "lensedCls.dat",
    "lensed_total_output_file": "",
    "lens_potential_output_file": "lenspotentialCls.dat",
    "scalar_covariance_output_file": "",
    "FITS_filename": "",
    "do_lensing_bispectrum": False,
    "do_primordial_bispectrum": False,
    "feedback_level": 1,
    "derived_parameters": True,
    "lensing_method": 1,
    "accurate_BB": False,
    "massive_nu_approx": 1,
    "accurate_polarization": True,
    "accurate_reionization": True,
    "do_tensor_neutrinos": True,
    "do_late_rad_truncation": True,
    "number_of_threads": 0,
    "high_accuracy_default": True,
    "accuracy_boost": 1,
    "l_accuracy_boost": 1,
    "l_sample_boost": 1,
    "write_aniso_transfer": False,
}

_SCAL_COLUMNS = ["tt", "ee", "bb", "te", "eb"]
_LENSED_COLUMNS = ["tt", "ee", "bb", "te"]
_LENSPOT_COLUMNS = {"pp": (5, 2.0), "tp": (6, 1.5), "ep": (7, 1.5)}

_CL_SUPPORTED = {"tt", "te", "ee", "bb", "eb", "pp", "tp", "ep"}

_DERIVED_PATTERNS = {
    "age": r"Age of universe/GYr\s*=\s*([0-9.eEdD+-]+)",
    "zstar": r"zstar\s*=\s*([0-9.eEdD+-]+)",
    "rstar": r"r_s\(zstar\)/Mpc\s*=\s*([0-9.eEdD+-]+)",
    "theta_s_100": r"100\*theta\s+=\s*([0-9.eEdD+-]+)",
    "theta_MC_100": r"100 theta \(CosmoMC\)\s*=\s*([0-9.eEdD+-]+)",
    "zdrag": r"zdrag\s*=\s*([0-9.eEdD+-]+)",
    "rdrag": r"r_s\(zdrag\)/Mpc\s*=\s*([0-9.eEdD+-]+)",
    "omegam": r"Om_m \(1-Om_K-Om_L\)\s*=\s*([0-9.eEdD+-]+)",
}


def _canon_cl(cl):
    order = "tebp"
    if len(cl) != 2 or any(c not in order for c in cl):
        return cl
    return "".join(sorted(cl, key=order.index))


def _ini_value(v):
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return "%.12g" % v
    return str(v)


class AxiECAMB(BoltzmannBase):
    """Cobaya interface to the AxiECAMB executable."""

    path = None
    executable = "camb"
    use_axfrac = True
    lensing = True
    accurate_bb = False
    lmax_margin = 250
    num_threads = 0
    timeout = 600
    run_dir = None
    extra_args = {}
    speed = 0.8

    def initialize(self):
        super().initialize()
        self.path = self.path or os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self._exe = os.path.join(self.path, self.executable)
        if not (os.path.isfile(self._exe) and os.access(self._exe, os.X_OK)):
            raise LoggedError(
                self.log,
                "AxiECAMB executable not found at %s. Compile it with 'make' in %s, "
                "or set the 'path'/'executable' options.",
                self._exe,
                self.path,
            )
        self._highl_template = os.path.join(
            self.path, "HighLExtrapTemplate_lenspotentialCls.dat"
        )
        if not os.path.isfile(self._highl_template):
            raise LoggedError(
                self.log, "Missing template file %s", self._highl_template
            )
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
            self._tmpdir = self.run_dir
            self._tmpdir_is_temp = False
        else:
            self._tmpdir = tempfile.mkdtemp(prefix="axiecamb_")
            self._tmpdir_is_temp = True
        self._lmax_request = 0
        self._t_cmb = float(
            self.extra_args.get("temp_cmb", _BASE_INI["temp_cmb"])
        )
        # these ini keys would override the top-level options after the wrapper
        # has already based its validation and parsing on them
        if shadowed := {"use_axfrac", "do_lensing", "accurate_BB"} & set(
            self.extra_args
        ):
            raise LoggedError(
                self.log,
                "Do not set %s via extra_args; use the corresponding top-level "
                "option(s) ('use_axfrac', 'lensing', 'accurate_bb') instead.",
                sorted(shadowed),
            )

    def close(self, *args):
        if getattr(self, "_tmpdir_is_temp", False):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def initialize_with_params(self):
        super().initialize_with_params()
        # the Fortran reads only one abundance parametrization and silently
        # ignores the other, which would leave a sampled parameter unused
        ignored = (
            {"omch2", "omaxh2"} if self.use_axfrac else {"omdah2", "axfrac"}
        ) & set(self.input_params)
        if ignored:
            raise LoggedError(
                self.log,
                "Parameter(s) %s are ignored by AxiECAMB when use_axfrac=%s; "
                "use %s instead, or change the 'use_axfrac' option.",
                sorted(ignored),
                self.use_axfrac,
                ["omdah2", "axfrac"] if self.use_axfrac else ["omch2", "omaxh2"],
            )
        if {"m_ax", "log10_m_ax"} <= set(self.input_params):
            raise LoggedError(
                self.log, "Give either 'm_ax' or 'log10_m_ax', not both."
            )
        expected = {"omdah2", "axfrac"} if self.use_axfrac else {"omch2", "omaxh2"}
        defaulted = expected - set(self.input_params) - set(self.extra_args)
        if not {"m_ax", "log10_m_ax"} & set(self.input_params) and (
            "m_ax" not in self.extra_args
        ):
            defaulted.add("m_ax")
        if defaulted:
            self.log.warning(
                "Axion-sector parameter(s) neither sampled/fixed in the params "
                "block nor set in extra_args; using built-in defaults: %s",
                {k: _BASE_INI[k] for k in sorted(defaulted)},
            )

    def must_provide(self, **requirements):
        for k, v in list(requirements.items()):
            if k in ("Cl", "unlensed_Cl"):
                req = {_canon_cl(cl.lower()): lmax for cl, lmax in v.items()}
                if unsupported := set(req) - _CL_SUPPORTED:
                    raise LoggedError(
                        self.log,
                        "Cl spectra %s not computed by AxiECAMB (available: %s; "
                        "note TB is not implemented).",
                        sorted(unsupported),
                        sorted(_CL_SUPPORTED),
                    )
                if not self.lensing and {"pp", "tp", "ep"} & set(req):
                    raise LoggedError(
                        self.log,
                        "Lensing potential spectra requested but option "
                        "'lensing' is False.",
                    )
                requirements[k] = req
            elif v is not None:
                raise LoggedError(
                    self.log,
                    "Requisite '%s' is not implemented in the AxiECAMB wrapper "
                    "(only Cl, unlensed_Cl and derived parameters).",
                    k,
                )
        super().must_provide(**requirements)
        lmax = 0
        for k in ("Cl", "unlensed_Cl"):
            lmax = max(lmax, *(self._must_provide.get(k) or {0: 0}).values())
        self._lmax_request = lmax

    def get_can_provide_params(self):
        return list(_DERIVED_PATTERNS)

    def calculate(self, state, want_derived=True, **params_values_dict):
        ini = dict(_BASE_INI)
        ini["use_axfrac"] = self.use_axfrac
        ini["do_lensing"] = self.lensing
        ini["accurate_BB"] = self.accurate_bb
        ini["number_of_threads"] = self.num_threads
        if not self.lensing:
            ini["lensed_output_file"] = ""
            ini["lens_potential_output_file"] = ""
        ini.update(self.extra_args)
        lmax_calc = max(self._lmax_request + self.lmax_margin, 1000)
        ini["l_max_scalar"] = max(int(ini["l_max_scalar"]), lmax_calc)
        ini["k_eta_max_scalar"] = max(
            int(ini["k_eta_max_scalar"]), 2 * ini["l_max_scalar"]
        )
        ini["output_root"] = os.path.join(self._tmpdir, "ax")
        ini["highL_unlensed_cl_template"] = self._highl_template

        for p, v in params_values_dict.items():
            if p == "log10_m_ax":
                ini["m_ax"] = 10.0**v
            elif p in PARAM_TO_INI:
                ini[PARAM_TO_INI[p]] = v
            else:
                raise LoggedError(
                    self.log,
                    "Unknown input parameter '%s'. Parameters understood by the "
                    "AxiECAMB wrapper: %s (plus 'log10_m_ax'). For fixed settings "
                    "use the 'extra_args' option with raw params.ini keys.",
                    p,
                    list(PARAM_TO_INI),
                )

        # outputs of the previous evaluation must not survive: a failed run
        # would otherwise silently hand back stale spectra
        for f in glob.glob(ini["output_root"] + "_*"):
            os.remove(f)

        ini_path = os.path.join(self._tmpdir, "params.ini")
        with open(ini_path, "w") as f:
            f.writelines(f"{k} = {_ini_value(v)}\n" for k, v in ini.items())

        try:
            proc = subprocess.run(
                [self._exe, ini_path],
                cwd=self._tmpdir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            self.log.warning("AxiECAMB timed out for parameters %r", params_values_dict)
            return False
        if proc.returncode != 0:
            self.log.warning(
                "AxiECAMB failed (exit %d) for parameters %r. Output tail:\n%s",
                proc.returncode,
                params_values_dict,
                "\n".join((proc.stdout + proc.stderr).splitlines()[-10:]),
            )
            return False

        try:
            self._load_cls(state, ini)
        except (OSError, ValueError) as err:
            self.log.warning(
                "Could not read AxiECAMB output for parameters %r (%s). "
                "Output tail:\n%s",
                params_values_dict,
                err,
                "\n".join((proc.stdout + proc.stderr).splitlines()[-10:]),
            )
            return False

        if want_derived:
            state["derived"] = self._parse_derived(proc.stdout)
        return True

    def _load_cls(self, state, ini):
        root = ini["output_root"] + "_"
        lmax_out = self._lmax_request or int(ini["l_max_scalar"]) - self.lmax_margin
        ell = np.arange(lmax_out + 1)

        scal = np.atleast_2d(np.loadtxt(root + ini["scalar_output_file"]))
        unlensed = {"ell": ell}
        for i, key in enumerate(_SCAL_COLUMNS, start=1):
            unlensed[key] = self._dl_to_cl(scal[:, i], lmax_out)

        if "unlensed_Cl" in (self._must_provide or {}):
            state["unlensed_Cl"] = unlensed

        if "Cl" not in (self._must_provide or {}):
            return
        if not self.lensing:
            state["Cl"] = unlensed
            return

        lensed = np.atleast_2d(np.loadtxt(root + ini["lensed_output_file"]))
        cls = {"ell": ell}
        for i, key in enumerate(_LENSED_COLUMNS, start=1):
            cls[key] = self._dl_to_cl(lensed[:, i], lmax_out)
        # the lensing convolution does not act on the rotation-induced scalar
        # B mode, and EB exists only unlensed
        cls["bb"] = cls["bb"] + unlensed["bb"]
        cls["eb"] = unlensed["eb"]

        lenspot = np.atleast_2d(np.loadtxt(root + ini["lens_potential_output_file"]))
        for key, (i, ell_power) in _LENSPOT_COLUMNS.items():
            cls[key] = self._dl_to_cl(lenspot[:, i], lmax_out, ell_power=ell_power)
        state["Cl"] = cls

    @staticmethod
    def _dl_to_cl(col, lmax_out, ell_power=1.0):
        n = lmax_out - 1
        if len(col) < n:
            raise ValueError(
                f"output only reaches l={len(col) + 1} but l={lmax_out} was "
                "requested; increase the 'lmax_margin' option"
            )
        ls = np.arange(2, lmax_out + 1, dtype=float)
        cl = np.zeros(lmax_out + 1)
        cl[2:] = col[:n] * 2 * np.pi / (ls * (ls + 1.0)) ** ell_power
        return cl

    def _parse_derived(self, stdout):
        derived = {}
        for name, pattern in _DERIVED_PATTERNS.items():
            if match := re.search(pattern, stdout):
                derived[name] = float(match.group(1).lower().replace("d", "e"))
        if missing := set(self.output_params or []) - set(derived):
            raise LoggedError(
                self.log,
                "Derived parameter(s) %s not found in AxiECAMB feedback output.",
                sorted(missing),
            )
        return derived

    def _format_cls(self, cls_in, ell_factor, units):
        field_factor = self._cmb_unit_factor(units, self._t_cmb) / _T_FIRAS_MUK
        ells = cls_in["ell"]
        llp1 = ells * (ells + 1.0)
        out = {"ell": ells.copy()}
        for key, arr in cls_in.items():
            if key == "ell":
                continue
            n_p = key.count("p")
            cl = arr * field_factor ** (2 - n_p)
            if ell_factor:
                cl = cl * llp1 ** (1 + n_p / 2.0) / (2 * np.pi)
            out[key] = cl
        return out

    def get_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._format_cls(self.current_state["Cl"], ell_factor, units)

    def get_unlensed_Cl(self, ell_factor=False, units="FIRASmuK2"):
        return self._format_cls(self.current_state["unlensed_Cl"], ell_factor, units)
