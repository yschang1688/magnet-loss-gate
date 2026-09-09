"""JD-3 PoC: a DSP-sized model that patches what the classical model gets wrong.

Classical baseline = Steinmetz with fixed coefficients (one calibration, one material).
Its known weaknesses are exactly the JD's list: non-linearity (non-sine excitation),
multiple operating points (temperature, frequency range), parameter drift (material
batch). The PoC keeps Steinmetz as the backbone and adds a *tiny residual MLP*:

    log P = Steinmetz(f, B_pk) + g(log f, log B_pk, T, purity, duty, crest)

g is 6 -> 16 -> 16 -> 1 with tanh: ~400 parameters, int8-quantisable, a few hundred MACs.
That is the shape of thing that fits next to a control loop on a C2000-class DSP and
can be re-fitted from a few hundred fresh samples when the material batch drifts.

Reports: accuracy vs Steinmetz / vs the big LightGBM, per temperature and per waveform
family; model size (float32 / int8) and MAC count; quantisation error; few-shot
adaptation to another material; physics gate (monotonicity probe) on the small model.
Exports include/residual_mlp.h with int8 weights + scales.

Usage: python src/embedded.py --material "Material B" --adapt-material "Material E"
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features
from train import load_material, steinmetz_fit, steinmetz_pred, rel_err

IN_COLS = ["freq", "b_pk", "temp", "purity", "duty_pos", "crest_dB"]
HIDDEN = (16, 16)


def design(X):
    Z = pd.DataFrame({"lf": np.log(X["freq"]), "lb": np.log(X["b_pk"]), "t": X["temp"],
                      "pu": X["purity"], "du": X["duty_pos"], "cr": X["crest_dB"]})
    return Z


class Scaler:
    def fit(self, Z): self.mu, self.sd = Z.mean(), Z.std().replace(0, 1); return self
    def __call__(self, Z): return ((Z - self.mu) / self.sd).to_numpy()


def fit_residual(Ztr_s, r, warm=None, max_iter=3000, seed=42):
    m = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="tanh", solver="lbfgs",
                     max_iter=max_iter, random_state=seed, alpha=1e-4)
    if warm is not None:
        m.warm_start = True
        m.coefs_, m.intercepts_ = [w.copy() for w in warm.coefs_], [b.copy() for b in warm.intercepts_]
        m.n_layers_ = warm.n_layers_; m.n_outputs_ = 1; m.out_activation_ = "identity"
        m._random_state = np.random.RandomState(seed)
    m.fit(Ztr_s, r)
    return m


def quantize_int8(m, bits=8):
    """Per-tensor symmetric intN; returns quantised copies + scales + byte count."""
    qmax = 2 ** (bits - 1) - 1
    dt = np.int8 if bits == 8 else np.int16
    qw, qb, scales = [], [], []
    for W, b in zip(m.coefs_, m.intercepts_):
        s = float(np.max(np.abs(W))) / qmax or 1.0
        qw.append(np.round(W / s).astype(dt)); scales.append(s)
        qb.append(b.astype(np.float32))                      # biases kept float (tiny)
    nbytes = sum(w.size * (bits // 8) for w in qw) + sum(b.size * 4 for b in qb) + 4 * len(scales)
    return qw, qb, scales, nbytes


def forward_q(qw, qb, scales, Zs):
    a = Zs
    for i, (W, b, s) in enumerate(zip(qw, qb, scales)):
        a = a @ (W.astype(np.float32) * s) + b
        if i < len(qw) - 1: a = np.tanh(a)
    return a.ravel()


def mono_violation(pred_fn, X, col, mult=(1.0, 1.3, 1.6, 2.0)):
    sub = X.sample(min(400, len(X)), random_state=0).reset_index(drop=True)
    P = []
    for k in mult:
        Xg = sub.copy(); Xg[col] = Xg[col] * k; P.append(pred_fn(Xg))
    P = np.stack(P)
    return float((np.diff(P, axis=0) < np.log(0.98)).any(axis=0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Material B")
    ap.add_argument("--adapt-material", default="Material E")
    ap.add_argument("--data", default="data/extracted/final-training")
    ap.add_argument("--few-shot", type=int, default=200)
    args = ap.parse_args()
    root = Path(__file__).parent.parent

    B, f, T, P = load_material(Path(args.data), args.material)
    X = extract_features(B, f, T); y = np.log(P)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

    se, se_p = steinmetz_fit(Xtr, ytr)
    r_tr = ytr - steinmetz_pred(se, Xtr)
    sc = Scaler().fit(design(Xtr))
    g = fit_residual(sc(design(Xtr)), r_tr)

    def pred_full(Xq): return steinmetz_pred(se, Xq) + g.predict(sc(design(Xq)))
    def pred_se(Xq):   return steinmetz_pred(se, Xq)

    Pte = np.exp(yte)
    res = {"material": args.material, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
           "steinmetz": rel_err(Pte, np.exp(pred_se(Xte))),
           "steinmetz_plus_tiny_mlp": rel_err(Pte, np.exp(pred_full(Xte)))}

    # ---- where does the patch help: per temperature, per waveform family ----
    fam = np.where(Xte["purity"] > 0.97, "sine", "non-sine")
    by = {}
    for key, mask_series in [("temp", Xte["temp"].astype(int).astype(str)), ("waveform", pd.Series(fam, index=Xte.index))]:
        by[key] = {}
        for v in sorted(mask_series.unique()):
            m = (mask_series == v).to_numpy()
            by[key][str(v)] = {"n": int(m.sum()),
                               "se_p95": round(rel_err(Pte[m], np.exp(pred_se(Xte[m])))["rel_err_p95"], 1),
                               "se+mlp_p95": round(rel_err(Pte[m], np.exp(pred_full(Xte[m])))["rel_err_p95"], 1)}
    res["breakdown"] = by

    # ---- size / cost ----
    n_params = int(sum(w.size for w in g.coefs_) + sum(b.size for b in g.intercepts_))
    macs = int(sum(w.size for w in g.coefs_))
    qw, qb, scales, nbytes = quantize_int8(g)
    q_pred = steinmetz_pred(se, Xte) + forward_q(qw, qb, scales, sc(design(Xte)))
    qw16, qb16, sc16, nb16 = quantize_int8(g, bits=16)
    q16_pred = steinmetz_pred(se, Xte) + forward_q(qw16, qb16, sc16, sc(design(Xte)))
    res["embedded"] = {"params": n_params, "macs_per_inference": macs,
                       "float32_bytes": n_params * 4, "int16_bytes": int(nb16), "int8_bytes": int(nbytes),
                       "est_us_at_100MHz_1MAC_per_cycle": round(macs / 100.0, 2),
                       "float32_p95": round(res["steinmetz_plus_tiny_mlp"]["rel_err_p95"], 1),
                       "int16_quantised_p95": round(rel_err(Pte, np.exp(q16_pred))["rel_err_p95"], 1),
                       "int8_quantised_p95": round(rel_err(Pte, np.exp(q_pred))["rel_err_p95"], 1),
                       "note": "per-tensor symmetric quantisation; int8 loses too much here -> ship int16 or float32 (C2000 has FPU)"}

    # ---- eval hygiene: hold out one whole temperature (cross-operating-point, no neighbours) ----
    hold = {}
    for t_hold in sorted(X["temp"].unique()):
        mtr = (X["temp"] != t_hold).to_numpy(); mte = ~mtr
        se_h, _ = steinmetz_fit(X[mtr], y[mtr])
        sc_h = Scaler().fit(design(X[mtr]))
        g_h = fit_residual(sc_h(design(X[mtr])), y[mtr] - steinmetz_pred(se_h, X[mtr]))
        p_h = steinmetz_pred(se_h, X[mte]) + g_h.predict(sc_h(design(X[mte])))
        hold[str(int(t_hold))] = {"n": int(mte.sum()),
                                  "se_p95": round(rel_err(np.exp(y[mte]), np.exp(steinmetz_pred(se_h, X[mte])))["rel_err_p95"], 1),
                                  "se+mlp_p95": round(rel_err(np.exp(y[mte]), np.exp(p_h))["rel_err_p95"], 1)}
    res["holdout_temperature"] = hold

    # ---- physics gate on the small model ----
    res["gate"] = {"G2_freq_mono_violation_rate": mono_violation(pred_full, Xte, "freq"),
                   "G3_flux_mono_violation_rate": mono_violation(pred_full, Xte, "b_pk")}

    # ---- parameter drift: another material, few-shot adaptation ----
    B2, f2, T2, P2 = load_material(Path(args.data), args.adapt_material)
    X2 = extract_features(B2, f2, T2); y2 = np.log(P2)
    X2a, X2te, y2a, y2te = train_test_split(X2, y2, test_size=0.5, random_state=7)
    X2a, y2a = X2a.iloc[:args.few_shot], y2a[:args.few_shot]
    P2te = np.exp(y2te)
    # (a) B-calibrated Steinmetz on E, (b) B model on E, (c) refit SE on 200 E samples + fine-tune residual
    se2, se2_p = steinmetz_fit(X2a, y2a)
    r2 = y2a - steinmetz_pred(se2, X2a)
    g2 = fit_residual(sc(design(X2a)), r2, warm=g, max_iter=300)
    res["drift_adaptation"] = {
        "adapt_material": args.adapt_material, "few_shot_n": int(len(X2a)), "n_test": int(len(X2te)),
        "stale_steinmetz_from_B_p95": round(rel_err(P2te, np.exp(steinmetz_pred(se, X2te)))["rel_err_p95"], 1),
        "stale_B_model_p95": round(rel_err(P2te, np.exp(pred_full(X2te)))["rel_err_p95"], 1),
        "refit_steinmetz_200_p95": round(rel_err(P2te, np.exp(steinmetz_pred(se2, X2te)))["rel_err_p95"], 1),
        "refit_steinmetz_200_plus_finetuned_mlp_p95": round(rel_err(P2te, np.exp(steinmetz_pred(se2, X2te) + g2.predict(sc(design(X2te)))))["rel_err_p95"], 1),
    }

    # ---- export C header ----
    inc = root / "include"; inc.mkdir(exist_ok=True)
    L = [f"/* auto-generated by src/embedded.py — {args.material}: log P = Steinmetz + tiny residual MLP",
         f"   Steinmetz: log_k={se_p['log_k']:.6f} alpha={se_p['alpha']:.6f} beta={se_p['beta']:.6f}",
         "   inputs (standardised): [ln f, ln B_pk, T, purity, duty, crest]; hidden tanh; int8 weights * scale + float bias */",
         f"#define RES_N_IN 6\n#define RES_H1 {HIDDEN[0]}\n#define RES_H2 {HIDDEN[1]}",
         "static const float RES_IN_MU[RES_N_IN] = {" + ", ".join(f"{v:.6f}" for v in sc.mu) + "};",
         "static const float RES_IN_SD[RES_N_IN] = {" + ", ".join(f"{v:.6f}" for v in sc.sd) + "};"]
    for i, (W, b, s) in enumerate(zip(qw, qb, scales)):
        L.append(f"static const float RES_S{i} = {s:.8f}f;")
        L.append(f"static const signed char RES_W{i}[{W.shape[0]}][{W.shape[1]}] = {{" + ", ".join("{" + ", ".join(str(int(v)) for v in row) + "}" for row in W) + "};")
        L.append(f"static const float RES_B{i}[{b.size}] = {{" + ", ".join(f"{v:.6f}f" for v in b) + "};")
    (inc / "residual_mlp.h").write_text("\n".join(L) + "\n")

    out = root / "reports"; out.mkdir(exist_ok=True)
    (out / f"{args.material}-embedded.json").write_text(json.dumps(res, indent=2))
    mlflow.set_tracking_uri("sqlite:///" + str((root / "mlflow.db").absolute()))
    mlflow.set_experiment("magnet-loss-gate")
    with mlflow.start_run(run_name=f"{args.material}-embedded-residual-mlp"):
        mlflow.log_params({"hidden": str(HIDDEN), "params": n_params, "int8_bytes": nbytes})
        mlflow.log_metrics({"se_p95": res["steinmetz"]["rel_err_p95"], "se_mlp_p95": res["steinmetz_plus_tiny_mlp"]["rel_err_p95"],
                            "int8_p95": res["embedded"]["int8_quantised_p95"],
                            "drift_stale_p95": res["drift_adaptation"]["stale_B_model_p95"],
                            "drift_adapted_p95": res["drift_adaptation"]["refit_steinmetz_200_plus_finetuned_mlp_p95"],
                            "G2": res["gate"]["G2_freq_mono_violation_rate"], "G3": res["gate"]["G3_flux_mono_violation_rate"]})
        mlflow.log_artifact(str(out / f"{args.material}-embedded.json")); mlflow.log_artifact(str(inc / "residual_mlp.h"))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
