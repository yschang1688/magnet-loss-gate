"""JD-3 deliverable (strategy layer): an AI-derived control policy the firmware can execute.

The real-time loop (PWM, ADC, protection) stays in firmware on the MCU/DSP. What the
model side owns is the *strategy*: for each operating condition the converter may see
(volt-second demand K as the load/voltage proxy, core temperature T), which switching
frequency minimises core loss — with every cell of the policy having passed the same
physics gates as the design optimiser (saturation, in-distribution, Steinmetz sanity).

The output is a lookup table (CSV + C header) plus a validation report:
  * gate coverage      every cell's chosen point passed all gates
  * raw-optimum vetoes  how many cells the un-gated surrogate would have got wrong
  * smoothness         largest f* jump between neighbouring cells — a jumpy policy
                       chatters in the loop; firmware adds hysteresis, but the table
                       should not hand it a cliff to begin with

Usage: python src/policy.py --material "Material B"
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).parent))
from optimize import pick_templates, build_candidates, in_distribution  # noqa: E402
from train import steinmetz_pred  # noqa: E402


def gated_optimum(bundle, K, T, b_sat, n_freq=60):
    model, Xtr, feats = bundle["model"], bundle["X_train"], bundle["features"]
    freqs = np.geomspace(float(Xtr["freq"].min()), float(Xtr["freq"].max()), n_freq)
    C = build_candidates(freqs, pick_templates(Xtr), K, T)
    C["p_ml"] = np.exp(model.predict(C[feats]))
    se = LinearRegression(); se.coef_ = np.array([bundle["steinmetz"]["alpha"], bundle["steinmetz"]["beta"]]); se.intercept_ = bundle["steinmetz"]["log_k"]
    C["p_se"] = np.exp(steinmetz_pred(se, C))
    ok_id, dist, thr = in_distribution(Xtr, C, feats)
    C["in_dist"] = ok_id
    C["g4_ok"] = np.abs(np.log(C["p_ml"] / C["p_se"])) <= np.log(3.0)
    C["g5_ok"] = C["b_pk"] <= b_sat
    C["gate_pass"] = C["in_dist"] & C["g4_ok"] & C["g5_ok"]
    raw = C.loc[C["p_ml"].idxmin()]
    if not C["gate_pass"].any():
        return None, raw, C
    gated = C[C["gate_pass"]].loc[lambda d: d["p_ml"].idxmin()]
    return gated, raw, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Material B")
    ap.add_argument("--b-sat", type=float, default=0.40)
    ap.add_argument("--k-grid", default="3e3,5e3,8e3,1.2e4,2e4,3e4,4e4")
    ap.add_argument("--t-grid", default="25,50,70,90")
    ap.add_argument("--tol", type=float, default=0.20, help="loss tolerance band for smoothing (0.05 = within 5%% of cell optimum)")
    args = ap.parse_args()
    root = Path(__file__).parent.parent
    b = joblib.load(root / "models" / f"{args.material}.joblib")
    Ks = [float(x) for x in args.k_grid.split(",")]
    Ts = [float(x) for x in args.t_grid.split(",")]

    rows = []
    for T in Ts:
        for K in Ks:
            g, raw, C = gated_optimum(b, K, T, args.b_sat)
            rows.append({
                "temp_C": T, "K_T_Hz": K,
                "f_star_hz": None if g is None else round(float(g["freq"])),
                "b_pk_T": None if g is None else round(float(g["b_pk"]), 4),
                "waveform": None if g is None else g["waveform"],
                "p_v_W_m3": None if g is None else round(float(g["p_ml"]), 1),
                "no_feasible_point": g is None,
                "raw_optimum_vetoed": bool(not raw["gate_pass"]),
                "raw_f_hz": round(float(raw["freq"])), "raw_b_pk_T": round(float(raw["b_pk"]), 4),
                "n_gate_pass": int(C["gate_pass"].sum()),
            })
    P = pd.DataFrame(rows)

    # ---- smoothing: a policy is executed sequentially as load moves, so among the
    # gate-passing candidates whose loss is within `tol` of the cell optimum, pick the
    # frequency closest to the previous K cell (per temperature row). Loss cost is
    # bounded by tol; chatter is what we buy down. ----
    def smooth_row(cands_by_K, tol=args.tol):
        prev, out = None, {}
        for K in sorted(cands_by_K):
            C = cands_by_K[K]
            ok = C[C["gate_pass"]]
            if ok.empty:
                out[K] = None; continue
            band = ok[ok["p_ml"] <= ok["p_ml"].min() * (1 + tol)]
            pick = band.loc[band["p_ml"].idxmin()] if prev is None else band.loc[(np.log(band["freq"]) - np.log(prev)).abs().idxmin()]
            prev = float(pick["freq"]); out[K] = pick
        return out

    cands = {}
    for T in Ts:
        for K in Ks:
            _, _, C = gated_optimum(b, K, T, args.b_sat)
            cands.setdefault(T, {})[K] = C
    P["f_smooth_hz"] = np.nan; P["p_v_smooth_W_m3"] = np.nan
    for T in Ts:
        picks = smooth_row(cands[T])
        for K, pk in picks.items():
            m = (P["temp_C"] == T) & (P["K_T_Hz"] == K)
            if pk is not None:
                P.loc[m, "f_smooth_hz"] = round(float(pk["freq"])); P.loc[m, "p_v_smooth_W_m3"] = round(float(pk["p_ml"]), 1)
    P["loss_penalty_pct"] = ((P["p_v_smooth_W_m3"] / P["p_v_W_m3"]) - 1) * 100

    # ---- policy validation ----
    feasible = P[~P["no_feasible_point"]]
    def max_jump(col):
        js = []
        for T, g in feasible.groupby("temp_C"):
            f = g.sort_values("K_T_Hz")[col].to_numpy(dtype=float)
            if len(f) > 1: js.append(float(np.max(np.abs(np.diff(np.log(f))))))
        return round(max(js), 3) if js else None
    smooth = [max_jump("f_star_hz")]
    report = {
        "material": args.material, "b_sat_T": args.b_sat,
        "grid": {"K": Ks, "T": Ts, "n_cells": int(len(P))},
        "cells_feasible": int(len(feasible)),
        "cells_no_feasible_point": int(P["no_feasible_point"].sum()),
        "cells_raw_optimum_vetoed": int(P["raw_optimum_vetoed"].sum()),
        "veto_rate": round(float(P["raw_optimum_vetoed"].mean()), 3),
        "max_neighbour_jump_log_f_argmin": max_jump("f_star_hz"),
        "max_neighbour_jump_log_f_smoothed": max_jump("f_smooth_hz"),
        "smoothing_tol": args.tol,
        "max_loss_penalty_pct_from_smoothing": round(float(P["loss_penalty_pct"].max()), 2),
        "note": "jump is ln(f_i+1/f_i) between neighbouring K cells at fixed T; >0.7 (≈2x) means the loop would chatter — firmware hysteresis alone should not have to absorb that",
    }

    out = root / "reports"; out.mkdir(exist_ok=True)
    P.to_csv(out / f"{args.material}-policy-table.csv", index=False)
    (out / f"{args.material}-policy-report.json").write_text(json.dumps(report, indent=2))

    # ---- C header the firmware side can drop in ----
    inc = root / "include"; inc.mkdir(exist_ok=True)
    hdr = [f"/* auto-generated by src/policy.py — {args.material}; smoothed policy (tol={args.tol}), f*[T][K] in Hz, 0 = no feasible point */",
           f"#define POLICY_N_T {len(Ts)}", f"#define POLICY_N_K {len(Ks)}",
           "static const float POLICY_T_C[POLICY_N_T] = {" + ", ".join(f"{t:g}" for t in Ts) + "};",
           "static const float POLICY_K_THZ[POLICY_N_K] = {" + ", ".join(f"{k:g}" for k in Ks) + "};",
           "static const unsigned long POLICY_F_STAR_HZ[POLICY_N_T][POLICY_N_K] = {"]
    for T in Ts:
        g = P[P["temp_C"] == T].sort_values("K_T_Hz")
        hdr.append("  {" + ", ".join("0" if pd.isna(v) else str(int(v)) for v in g["f_smooth_hz"]) + "},")
    hdr.append("};")
    (inc / "policy_table.h").write_text("\n".join(hdr) + "\n")

    mlflow.set_tracking_uri("sqlite:///" + str((root / "mlflow.db").absolute()))
    mlflow.set_experiment("magnet-loss-gate")
    with mlflow.start_run(run_name=f"{args.material}-policy"):
        mlflow.log_params({"material": args.material, "b_sat": args.b_sat, "n_cells": len(P)})
        mlflow.log_metrics({"veto_rate": report["veto_rate"], "cells_feasible": report["cells_feasible"],
                            "max_jump_argmin": report["max_neighbour_jump_log_f_argmin"] or 0,
                            "max_jump_smoothed": report["max_neighbour_jump_log_f_smoothed"] or 0,
                            "max_loss_penalty_pct": report["max_loss_penalty_pct_from_smoothing"]})
        for f in (f"{args.material}-policy-table.csv", f"{args.material}-policy-report.json"):
            mlflow.log_artifact(str(out / f))
        mlflow.log_artifact(str(inc / "policy_table.h"))
    print(json.dumps(report, indent=2))
    print("argmin f*:"); print(P.pivot(index="temp_C", columns="K_T_Hz", values="f_star_hz").to_string())
    print("smoothed f*:"); print(P.pivot(index="temp_C", columns="K_T_Hz", values="f_smooth_hz").to_string())


if __name__ == "__main__":
    main()
