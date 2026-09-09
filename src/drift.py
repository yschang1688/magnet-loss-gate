"""JD-2 deliverable: drift check that decides when to retrain — without a human in the loop.

PSI (population stability index) per feature between the training distribution and a
new batch. PSI > 0.25 on any feature => retrain flag. The flag, per-feature PSI and the
batch identity are logged to MLflow so the decision is auditable, not tribal knowledge.

Two batches are exercised on purpose:
  --batch same     a fresh sample of the same material  -> expect no drift
  --batch other    a different material's waveforms     -> expect drift, retrain=1

Usage: python src/drift.py --material "Material B" --batch other --other "Material E"
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from features import extract_features
from train import load_material

PSI_RETRAIN = 0.25


def psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="Material B")
    ap.add_argument("--data", default="data/extracted/final-training")
    ap.add_argument("--batch", choices=["same", "other"], default="other")
    ap.add_argument("--other", default="Material E")
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    b = joblib.load(root / "models" / f"{args.material}.joblib")
    Xtr, feats = b["X_train"], b["features"]

    if args.batch == "same":
        B, f, T, P = load_material(Path(args.data), args.material)
        Xn = extract_features(B, f, T).sample(min(1500, len(f)), random_state=7)
        batch_id = f"{args.material} resample"
    else:
        B, f, T, P = load_material(Path(args.data), args.other)
        Xn = extract_features(B, f, T)
        batch_id = args.other

    per = {c: round(psi(Xtr[c].to_numpy(), Xn[c].to_numpy()), 4) for c in feats}
    worst = max(per, key=per.get)
    retrain = per[worst] > PSI_RETRAIN
    report = {"model_material": args.material, "batch": batch_id, "n_batch": int(len(Xn)),
              "psi_per_feature": per, "worst_feature": worst, "psi_threshold": PSI_RETRAIN,
              "retrain_triggered": bool(retrain)}

    out = root / "reports"; out.mkdir(exist_ok=True)
    (out / f"{args.material}-drift-{args.batch}.json").write_text(json.dumps(report, indent=2))

    mlflow.set_tracking_uri("sqlite:///" + str((root / "mlflow.db").absolute()))
    mlflow.set_experiment("magnet-loss-gate")
    with mlflow.start_run(run_name=f"{args.material}-drift-{args.batch}"):
        mlflow.log_params({"model_material": args.material, "batch": batch_id, "psi_threshold": PSI_RETRAIN})
        mlflow.log_metrics({f"psi_{k}": v for k, v in per.items()})
        mlflow.log_metrics({"psi_max": per[worst], "retrain_triggered": int(retrain)})
        mlflow.set_tag("action", "retrain" if retrain else "keep")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
