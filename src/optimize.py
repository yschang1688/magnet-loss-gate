"""JD-1 deliverable: AI-driven design-parameter optimisation for a magnetic component.

Design question a power engineer actually asks
------------------------------------------------
"For a transformer that must carry a given volt-second product, which switching
frequency and excitation waveform give the lowest core loss?"

Faraday ties the knobs together:  V = N * Ae * dB/dt  ->  for a fixed volt-second
budget, B_pk * f = K  (K set by the application). So the free design variables are
    f        switching frequency  (B_pk follows from the constraint)
    waveform excitation shape     (templates taken from the measured dataset)
    T        operating temperature (fixed per scenario)
and the objective is volumetric core loss predicted by the validated surrogate.

The optimiser is a transparent grid sweep (BO is a drop-in later). Every candidate
passes the same gate the training run used, plus an in-distribution check — because
an optimiser will happily converge on the region where the surrogate is most wrong.

Usage: python src/optimize.py --material "Material B" --k 1.0e4 --temp 50
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from train import steinmetz_pred  # noqa: E402
from mlp_surrogate import ResidualSurrogate  # noqa: E402,F401  (unpickle support)  (same log-space SE fit)

SHAPE_COLS = ["b_form", "crest_dB", "duty_pos", "purity"]


def pick_templates(Xtr: pd.DataFrame):
    """Three excitation templates drawn from real samples, chosen by nearest
    (purity, duty) match: sine; a symmetric trapezoid / near-square wave (flat
    ~80 % of the period, steep edges, hence 4-5x the loss of sine at equal K);
    and an asymmetric triangle with a 20 % rise time."""
    targets = {
        "sine":          {"purity": 1.00, "duty_pos": 0.50},
        "trapezoid_50":  {"purity": 0.80, "duty_pos": 0.50},
        "triangle_20":   {"purity": 0.80, "duty_pos": 0.20},
    }
    out = {}
    for name, t in targets.items():
        d = ((Xtr["purity"] - t["purity"]) ** 2 + (Xtr["duty_pos"] - t["duty_pos"]) ** 2)
        row = Xtr.iloc[int(d.idxmin())]
        out[name] = {c: float(row[c]) for c in SHAPE_COLS}
        out[name]["pp_ratio"] = float(row["b_pp_half"] / row["b_pk"])
    return out


def build_candidates(freqs, templates, K, temp):
    rows = []
    for wf, shape in templates.items():
        for f in freqs:
            bpk = K / f
            rows.append({"freq": f, "temp": temp, "b_pk": bpk,
                         "b_pp_half": bpk * shape["pp_ratio"],
                         "waveform": wf, **{c: shape[c] for c in SHAPE_COLS}})
    return pd.DataFrame(rows)


def in_distribution(Xtr, Xc, features, q=0.95):
    """kNN distance in standardised feature space; a candidate is 'in distribution'
    if its distance to the training set is below the q-quantile of train self-distances."""
    mu, sd = Xtr[features].mean(), Xtr[features].std().replace(0, 1)
    Z = ((Xtr[features] - mu) / sd).to_numpy()
    Zc = ((Xc[features] - mu) / sd).to_numpy()
    nn = NearestNeighbors(n_neighbors=2).fit(Z)
    self_d = nn.kneighbors(Z)[0][:, 1]                 # nearest *other* point
    thr = float(np.quantile(self_d, q))
    dc = nn.kneighbors(Zc, n_neighbors=1)[0][:, 0]
    return dc <= thr, dc, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Material B")
    ap.add_argument("--k", type=float, default=1.0e4, help="volt-second budget as B_pk*f [T*Hz]")
    ap.add_argument("--temp", type=float, default=50.0)
    ap.add_argument("--n-freq", type=int, default=60)
    ap.add_argument("--b-sat", type=float, default=0.40, help="G5: ferrite saturation bound for B_pk [T]")
    ap.add_argument("--bundle", default="", help="model bundle suffix, e.g. -mlp")
    args = ap.parse_args()
    tag = f"K{args.k:.0e}-T{args.temp:.0f}" + (args.bundle if args.bundle else "")

    root = Path(__file__).parent.parent
    b = joblib.load(root / "models" / f"{args.material}{args.bundle}.joblib")
    model, Xtr, feats = b["model"], b["X_train"], b["features"]

    fmin, fmax = float(Xtr["freq"].min()), float(Xtr["freq"].max())
    freqs = np.geomspace(fmin, fmax, args.n_freq)
    templates = pick_templates(Xtr)
    C = build_candidates(freqs, templates, args.k, args.temp)

    # ---- surrogate + baseline predictions (W/m^3) ----
    C["p_ml"] = np.exp(model.predict(C[feats]))
    from sklearn.linear_model import LinearRegression
    se = LinearRegression(); se.coef_ = np.array([b["steinmetz"]["alpha"], b["steinmetz"]["beta"]]); se.intercept_ = b["steinmetz"]["log_k"]
    C["p_se"] = np.exp(steinmetz_pred(se, C))

    # ---- gate: in-distribution + positivity + Steinmetz 3x sanity ----
    ok_id, dist, thr = in_distribution(Xtr, C, feats)
    C["in_dist"] = ok_id; C["nn_dist"] = dist
    C["g4_ok"] = np.abs(np.log(C["p_ml"] / C["p_se"])) <= np.log(3.0)
    C["g5_ok"] = C["b_pk"] <= args.b_sat                      # saturation: a design rule, not a data rule
    C["gate_pass"] = C["in_dist"] & C["g4_ok"] & C["g5_ok"] & (C["p_ml"] > 0)

    # ---- optimise: unconstrained argmin vs gated argmin ----
    raw = C.loc[C["p_ml"].idxmin()]
    gated = C[C["gate_pass"]].loc[lambda d: d["p_ml"].idxmin()]
    se_best = C.loc[C["p_se"].idxmin()]
    # what the surrogate says about the Steinmetz-recommended point
    se_best_ml = float(se_best["p_ml"])

    def pt(r):
        return {"freq_hz": round(float(r["freq"])), "b_pk_T": round(float(r["b_pk"]), 4),
                "waveform": r["waveform"], "p_ml_W_per_m3": round(float(r["p_ml"]), 1),
                "p_se_W_per_m3": round(float(r["p_se"]), 1),
                "in_dist": bool(r["in_dist"]), "nn_dist": round(float(r["nn_dist"]), 3),
                "below_saturation": bool(r["g5_ok"]), "gate_pass": bool(r["gate_pass"])}

    result = {
        "material": args.material, "constraint_Bpk_x_f": args.k, "temp_C": args.temp,
        "design_space": {"freq_range_hz": [round(fmin), round(fmax)], "n_candidates": int(len(C)),
                         "waveform_templates": templates},
        "gate": {"in_dist_threshold": round(thr, 3), "n_pass": int(C["gate_pass"].sum()),
                 "n_reject_out_of_dist": int((~C["in_dist"]).sum()),
                 "n_reject_g4": int((~C["g4_ok"]).sum()),
                 "n_reject_g5_saturation": int((~C["g5_ok"]).sum()), "b_sat_T": args.b_sat},
        "optimum_raw_surrogate": pt(raw),
        "optimum_gated": pt(gated),
        "steinmetz_recommendation": pt(se_best),
        "loss_saving_vs_steinmetz_pct": round(100 * (1 - float(gated["p_ml"]) / se_best_ml), 1),
        "raw_optimum_rejected_by_gate": bool(not raw["gate_pass"]),
    }

    out = root / "reports"; out.mkdir(exist_ok=True)
    (out / f"{args.material}-optimize-{tag}.json").write_text(json.dumps(result, indent=2))
    C.to_csv(out / f"{args.material}-candidates-{tag}.csv", index=False)

    # ---- chart for the deck: loss vs f per waveform, gated region, optima ----
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    colors = {"sine": "#0E6E78", "trapezoid_50": "#4B545C", "triangle_20": "#A2332A"}
    for wf, g in C.groupby("waveform"):
        ax.plot(g["freq"] / 1e3, g["p_ml"] / 1e3, color=colors[wf], lw=2, label=f"{wf}  (surrogate)")
        bad = g[~g["gate_pass"]]
        ax.scatter(bad["freq"] / 1e3, bad["p_ml"] / 1e3, color=colors[wf], marker="x", s=28, alpha=.8)
    ax.plot(C[C.waveform == "sine"]["freq"] / 1e3, C[C.waveform == "sine"]["p_se"] / 1e3,
            color="#7C858D", lw=1.2, ls="--", label="Steinmetz's own prediction (underestimates at high f)")
    ax.scatter([se_best["freq"] / 1e3], [se_best["p_ml"] / 1e3], s=130, marker="s", facecolor="none",
               edgecolor="#7C858D", lw=2.2, zorder=5, label="Steinmetz-recommended point, scored by surrogate")
    ax.scatter([gated["freq"] / 1e3], [gated["p_ml"] / 1e3], s=150, facecolor="none",
               edgecolor="#0E6E78", lw=2.5, zorder=6, label="gated optimum (surrogate)")
    ax.scatter([raw["freq"] / 1e3], [raw["p_ml"] / 1e3], s=110, marker="D", facecolor="none",
               edgecolor="#A2332A", lw=2, zorder=6, label="raw surrogate optimum")
    ymax = float(C["p_ml"].max()) / 1e3
    ax.annotate(f"−{result['loss_saving_vs_steinmetz_pct']:.1f}%  (same evaluator: surrogate at both points)",
                xy=(gated["freq"] / 1e3, gated["p_ml"] / 1e3),
                xytext=(gated["freq"] / 1e3 * 0.62, min(ymax * 0.97, gated["p_ml"] / 1e3 * 1.9)),
                fontsize=8.5, color="#0E6E78", ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#0E6E78", lw=1.2, connectionstyle="arc3,rad=-0.25"))
    ax.annotate("", xy=(se_best["freq"] / 1e3, se_best["p_ml"] / 1e3), xytext=(gated["freq"] / 1e3, gated["p_ml"] / 1e3),
                arrowprops=dict(arrowstyle="-", color="#0E6E78", lw=1, ls=":"))
    ax.set_xscale("log")
    ax.set_xlabel("switching frequency f [kHz]   (B_pk = K / f, K = %.0e T·Hz, T = %.0f °C)" % (args.k, args.temp))
    ax.set_ylabel("core loss P_v [kW/m³]")
    ax.set_title(f"{args.material} — design sweep under volt-second constraint  (× = gate reject)", fontsize=10)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (60,70,80,90,200,300,400) else ""))
    ax.grid(True, which="both", alpha=.25); ax.legend(fontsize=7.5, ncol=1, loc="upper left", framealpha=.92)
    fig.tight_layout(); fig.savefig(out / f"{args.material}-optimize-{tag}.png"); plt.close(fig)

    mlflow.set_tracking_uri("sqlite:///" + str((root / "mlflow.db").absolute()))
    mlflow.set_experiment("magnet-loss-gate")
    with mlflow.start_run(run_name=f"{args.material}-optimize-{tag}"):
        mlflow.log_params({"material": args.material, "K": args.k, "temp": args.temp, "n_candidates": len(C)})
        mlflow.log_metrics({"loss_saving_vs_steinmetz_pct": result["loss_saving_vs_steinmetz_pct"],
                            "gate_pass_frac": result["gate"]["n_pass"] / len(C),
                            "raw_optimum_rejected": int(result["raw_optimum_rejected_by_gate"])})
        mlflow.log_artifact(str(out / f"{args.material}-optimize-{tag}.json"))
        mlflow.log_artifact(str(out / f"{args.material}-optimize-{tag}.png"))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
