"""
Quick end-to-end check of the cobaya wrapper:

    python python/test_axiecamb.py

Runs the wrapper through cobaya, compares the spectra against a direct
./camb invocation (if /tmp/axie/test_lensedCls.dat exists), and checks that
birefringence (g_axion != 0) produces EB/BB power.
"""

import os

import numpy as np
from cobaya.model import get_model

HERE = os.path.dirname(os.path.abspath(__file__))
LMAX = 2450

captured = {}


def dummy_like(_self=None):
    captured["Cl"] = _self.provider.get_Cl(ell_factor=True)
    captured["unlensed_Cl"] = _self.provider.get_unlensed_Cl(ell_factor=True)
    return 0.0


def make_info(g_axion, accurate_bb):
    return {
        "theory": {
            "axiecamb.AxiECAMB": {
                "python_path": HERE,
                "use_axfrac": True,
                "accurate_bb": accurate_bb,
            }
        },
        "likelihood": {
            "dummy": {
                "external": dummy_like,
                "requires": {
                    "Cl": {cl: LMAX for cl in ("tt", "ee", "bb", "te", "eb", "pp")},
                    "unlensed_Cl": {cl: LMAX for cl in ("tt", "ee", "bb", "te", "eb")},
                },
            }
        },
        "params": {
            "ombh2": 0.0224,
            "omdah2": 0.12,
            "axfrac": 1.0,
            "m_ax": 1e-27,
            "g_axion": g_axion,
            "H0": 67.36,
            "tau": 0.05,
            "As": 2.196e-9,
            "ns": 0.9655,
            "omnuh2": 0.6451439e-3,
            "omk": 0.0,
            "zstar": {"derived": True},
            "age": {"derived": True},
            "theta_MC_100": {"derived": True},
            "omegam": {"derived": True},
        },
        "debug": False,
    }


def run(g_axion, accurate_bb=True):
    model = get_model(make_info(g_axion, accurate_bb))
    logpost = model.logposterior([])
    derived = dict(zip(model.parameterization.derived_params(), logpost.derived))
    return dict(captured), derived


def main():
    cls0, derived0 = run(g_axion=0.0)
    print("derived:", derived0)
    ell = cls0["Cl"]["ell"]
    sel = ell >= 2

    ref_file = "/tmp/axie/test_lensedCls.dat"
    if os.path.exists(ref_file):
        ref = np.loadtxt(ref_file)
        n = min(len(ref), LMAX - 1)
        for key, i in [("tt", 1), ("ee", 2), ("te", 4)]:
            wrapper_dl = cls0["Cl"][key][2 : n + 2]
            ratio = wrapper_dl / ref[:n, i]
            assert np.allclose(ratio, 1, atol=1e-3), (key, ratio)
        print("PASS: lensed TT/EE/TE match direct ./camb run")

    assert np.all(np.abs(cls0["Cl"]["eb"][sel]) < 1e-12), "EB should vanish for g=0"
    print("PASS: EB = 0 when g_axion = 0")

    cls1, _ = run(g_axion=0.05)
    eb = cls1["Cl"]["eb"]
    ee = cls1["Cl"]["ee"]
    assert np.any(np.abs(eb[sel]) > 0), "EB should be nonzero for g != 0"
    assert np.all(np.isfinite(eb)), "EB should be finite"
    bb_rot = cls1["unlensed_Cl"]["bb"]
    assert np.any(bb_rot[sel] > 0), "rotation BB should be nonzero for g != 0"
    imax = np.argmax(np.abs(eb))
    print(
        f"PASS: g_axion=0.05 gives EB (peak D_l={eb[imax]:.3e} muK^2 at l={imax}), "
        f"max|EB|/EE there = {abs(eb[imax]) / ee[imax]:.3e}"
    )
    print("all checks passed")


if __name__ == "__main__":
    main()
