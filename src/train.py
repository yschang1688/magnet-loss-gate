"""MagNet mini-demo: Steinmetz baseline vs LightGBM, with a physics validation gate.

One material, one day. The point is not leaderboard accuracy — it is that every
ML prediction ships with physics checks a magnetics engineer can veto:
  G1  positivity            predicted loss > 0 (log-target makes this structural)
  G2  frequency monotonicity loss must not decrease as f rises (others held)
  G3  flux monotonicity      loss must not decrease as B_pk rises
  G4  Steinmetz deviation    flag test samples where |log(P_ml/P_st)| > ln(3)

Usage: python src/train.py --material <name> --data data/extracted
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features

REL_ERR_Q = [50, 95, 99]


def load_material(root: Path, material: str):
    d = root / material
    def one(*names):
        for nm in names:
            hits = list(d.glob(nm))
            if hits:
                return pd.read_csv(hits[0], header=None).to_numpy(dtype=float)
        raise FileNotFoundError(f"{names} not in {d}")
    B = one("B_Field.csv", "B_waveform*.csv", "B[BH]*.csv")
    f = one("Frequency.csv").ravel()
    T = one("Temperature.csv").ravel()
    P = one("Volumetric_Loss.csv", "Volumetric_losses.csv").ravel()
    return B, f, T, P


def rel_err(y_true, y_pred):
    e = np.abs(y_pred - y_true) / y_true * 100
    return {f"rel_err_p{q}": float(np.percentile(e, q)) for q in REL_ERR_Q}


def steinmetz_fit(X, y_log):
    """log P = log k + a*log f + b*log Bpk  (classic SE, sine-dominant subset would
    be cleaner; fitting on all data keeps the baseline honest but imperfect)."""
    A = np.column_stack([np.log(X["freq"]), np.log(X["b_pk"])])
    m = LinearRegression().fit(A, y_log)
    return m, {"alpha": float(m.coef_[0]), "beta": float(m.coef_[1]),
               "log_k": float(m.intercept_)}


def steinmetz_pred(m, X):
    A = np.column_stack([np.log(X["freq"]), np.log(X["b_pk"])])
    return m.predict(A)


def monotonicity_violations(model, X, col, grid_mult=(1.0, 1.3, 1.6, 2.0)):
    """G2/G3 probe: scale one physical driver up on a sample subset; count
    predictions that DROP by more than 2% anywhere along the ramp."""
    sub = X.sample(min(400, len(X)), random_state=0).reset_index(drop=True)
    preds = []
    for mlt in grid_mult:
        Xg = sub.copy()
        Xg[col] = Xg[col] * mlt
        preds.append(model.predict(Xg))
    preds = np.stack(preds)                       # [grid, n] in log space
    drops = (np.diff(preds, axis=0) < np.log(0.98)).any(axis=0)
    return float(drops.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="N87")
    ap.add_argument("--data", default="data/extracted")
    args = ap.parse_args()

    B, f, T, P = load_material(Path(args.data), args.material)
    assert (P > 0).all(), "raw data contains non-positive losses"
    X = extract_features(B, f, T)
    y_log = np.log(P)

    Xtr, Xte, ytr, yte = train_test_split(X, y_log, test_size=0.25, random_state=42)
    Pte = np.exp(yte)

    mlflow.set_experiment("magnet-loss-gate")
    with mlflow.start_run(run_name=f"{args.material}-lgbm-vs-steinmetz"):
        mlflow.log_params({"material": args.material, "n_samples": len(X),
                           "features": ",".join(X.columns)})

        st_model, st_params = steinmetz_fit(Xtr, ytr)
        st_te = np.exp(steinmetz_pred(st_model, Xte))
        st_metrics = {f"steinmetz_{k}": v for k, v in rel_err(Pte, st_te).items()}
        mlflow.log_params({f"se_{k}": round(v, 4) for k, v in st_params.items()})
        mlflow.log_metrics(st_metrics)

        ml = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05,
                               num_leaves=63, random_state=42)
        ml.fit(Xtr, ytr)
        ml_te = np.exp(ml.predict(Xte))
        ml_metrics = {f"lgbm_{k}": v for k, v in rel_err(Pte, ml_te).items()}
        mlflow.log_metrics(ml_metrics)

        # ---- physics gate ----
        gate = {
            "G1_positivity_ok": bool((ml_te > 0).all()),           # structural via log target
            "G2_freq_mono_violation_rate": monotonicity_violations(ml, Xte, "freq"),
            "G3_flux_mono_violation_rate": monotonicity_violations(ml, Xte, "b_pk"),
        }
        dev = np.abs(np.log(ml_te / st_te))
        flagged = dev > np.log(3.0)
        gate["G4_steinmetz_3x_flag_rate"] = float(flagged.mean())
        mlflow.log_metrics({k: (float(v) if not isinstance(v, bool) else int(v))
                            for k, v in gate.items()})

        report = {"material": args.material, "n": len(X),
                  "steinmetz": st_metrics, "lgbm": ml_metrics,
                  "steinmetz_params": st_params, "physics_gate": gate}
        out = Path("reports"); out.mkdir(exist_ok=True)
        rp = out / f"{args.material}.json"
        rp.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(rp))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
