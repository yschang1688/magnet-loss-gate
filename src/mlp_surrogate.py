"""Wrap Steinmetz + residual MLP as a drop-in surrogate for optimize.py / policy.py
(.predict(X) returns log loss, same contract as the LightGBM bundle)."""
import sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).parent))
from train import steinmetz_pred


class ResidualSurrogate:
    def __init__(self, se, g, scaler, design):
        self.se, self.g, self.sc, self.design = se, g, scaler, design
    def predict(self, X):
        return steinmetz_pred(self.se, X) + self.g.predict(self.sc(self.design(X)))


def build(material="Material B", data="data/extracted/final-training"):
    from embedded import design, Scaler, fit_residual
    from train import load_material, steinmetz_fit
    from features import extract_features
    from sklearn.model_selection import train_test_split
    B, f, T, P = load_material(Path(data), material)
    X = extract_features(B, f, T); y = np.log(P)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    se, se_p = steinmetz_fit(Xtr, ytr)
    sc = Scaler().fit(design(Xtr))
    g = fit_residual(sc(design(Xtr)), ytr - steinmetz_pred(se, Xtr))
    base = joblib.load(Path("models") / f"{material}.joblib")
    bundle = dict(base); bundle["model"] = ResidualSurrogate(se, g, sc, design); bundle["surrogate"] = "steinmetz+residual_mlp"
    joblib.dump(bundle, Path("models") / f"{material}-mlp.joblib")
    print("saved models/%s-mlp.joblib" % material)


if __name__ == "__main__":
    build()
