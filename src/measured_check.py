"""Check the optimizer's answer against measured points that lie on the same
volt-second constraint line.

The design sweep in optimize.py scores *hypothetical* operating points with a
surrogate; those points were never measured, so the "saving" it reports is a
model-vs-model number. But the MagNet grid is dense enough that, for a given
K = B_pk * f and temperature, near-sine measurements exist at most frequencies.
This script pulls them, bins by frequency and reports the measured
loss-vs-frequency curve, its minimum, and the measured saving versus the
frequency limit (which is where Steinmetz always sends the design).

Only the sine template can be checked this way: the triangle templates used
in optimize.py have no measured counterpart in the dataset.

    python src/measured_check.py --k 5e3 --temp 25
    python src/measured_check.py --k 4e4 --temp 90
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import extract_features
from train import load_material

ROOT = Path(__file__).resolve().parents[1]


def measured_curve(material: str, K: float, T: float, tol: float, purity_min: float):
    B, f, temp, P = load_material(ROOT / "data/extracted/final-training", material)
    X = pd.DataFrame(extract_features(B, f, temp))
    k = X["b_pk"].to_numpy() * f
    m = (temp == T) & (np.abs(k / K - 1) < tol) & (X["purity"].to_numpy() >= purity_min)
    if m.sum() == 0:
        raise SystemExit(f"no measured near-sine points within ±{tol:.0%} of K={K:g} at T={T:g}")
    df = pd.DataFrame({"f": f[m], "b_pk": X["b_pk"].to_numpy()[m], "P": P[m]})
    # ~12 % log-spaced bins so 199/200 kHz and 397/398 kHz land in one bin
    df["bin"] = np.round(np.log10(df["f"]) * 20) / 20
    g = (df.groupby("bin")
           .agg(f_kHz=("f", lambda s: float(np.median(s)) / 1e3),
                n=("P", "size"),
                P_med=("P", "median"),
                b_pk=("b_pk", "mean"))
           .reset_index(drop=True)
           .sort_values("f_kHz"))
    return g, int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Material B")
    ap.add_argument("--k", type=float, required=True, help="volt-second constant B_pk*f [T*Hz]")
    ap.add_argument("--temp", type=float, required=True)
    ap.add_argument("--tol", type=float, default=0.06, help="relative tolerance on K")
    ap.add_argument("--purity", type=float, default=0.95, help="min fundamental purity (near-sine)")
    ap.add_argument("--min-n", type=int, default=5, help="ignore bins with fewer points when locating the minimum")
    a = ap.parse_args()

    g, n = measured_curve(a.material, a.k, a.temp, a.tol, a.purity)
    solid = g[g.n >= a.min_n]
    lo = solid.loc[solid.P_med.idxmin()]
    hi = solid.loc[solid.f_kHz.idxmax()]
    saving = (1 - lo.P_med / hi.P_med) * 100

    print(f"{a.material} | K={a.k:g} T*Hz ±{a.tol:.0%} | T={a.temp:g} C | purity>={a.purity} | n={n}")
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nmeasured minimum : {lo.f_kHz:.0f} kHz  P={lo.P_med:.3e} W/m3  (bins with n>={a.min_n})")
    print(f"frequency limit  : {hi.f_kHz:.0f} kHz  P={hi.P_med:.3e} W/m3")
    print(f"measured saving  : {saving:.1f} %  (minimum vs frequency limit)")

    out = ROOT / "reports" / f"{a.material}-measured-check-K{a.k:g}-T{a.temp:g}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "material": a.material, "K": a.k, "T": a.temp, "tol": a.tol, "purity_min": a.purity,
        "n_points": n,
        "curve": g.round(6).to_dict(orient="records"),
        "measured_min_kHz": float(lo.f_kHz), "measured_min_P": float(lo.P_med),
        "f_limit_kHz": float(hi.f_kHz), "P_at_limit": float(hi.P_med),
        "measured_saving_pct": float(saving),
        "note": "sine template only; triangle templates have no measured counterpart",
    }, indent=2, ensure_ascii=False))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
