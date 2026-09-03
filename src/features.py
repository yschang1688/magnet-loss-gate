"""Waveform feature extraction for MagNet core-loss samples.

Each sample: B(t) waveform (single period, uniformly sampled), excitation
frequency f [Hz], temperature T [degC]. Features are deliberately physical and
few — the demo argues for validatable models, not leaderboard models.
"""
import numpy as np
import pandas as pd


def extract_features(B: np.ndarray, freq: np.ndarray, temp: np.ndarray) -> pd.DataFrame:
    n = B.shape[1]
    bpk = np.abs(B).max(axis=1)                      # peak flux density [T]
    bpp = (B.max(axis=1) - B.min(axis=1)) / 2.0      # half peak-to-peak
    brms = np.sqrt((B ** 2).mean(axis=1))

    dB = np.diff(B, axis=1, append=B[:, :1])         # per-sample dB
    dB_rms = np.sqrt((dB ** 2).mean(axis=1))
    dB_pk = np.abs(dB).max(axis=1)
    crest_dB = np.divide(dB_pk, dB_rms, out=np.zeros_like(dB_pk), where=dB_rms > 0)

    # duty proxy: fraction of the period where dB/dt > 0 (0.5 for sine/symmetric tri)
    duty = (dB > 0).mean(axis=1)

    # fundamental purity: |FFT_1| energy share (1.0 = pure sine)
    spec = np.abs(np.fft.rfft(B, axis=1))
    spec[:, 0] = 0.0
    total = (spec ** 2).sum(axis=1)
    fund = spec[:, 1] ** 2
    purity = np.divide(fund, total, out=np.ones_like(fund), where=total > 0)

    return pd.DataFrame({
        "freq": freq,
        "temp": temp,
        "b_pk": bpk,
        "b_pp_half": bpp,
        "b_form": np.divide(brms, bpk, out=np.zeros_like(brms), where=bpk > 0),
        "crest_dB": crest_dB,
        "duty_pos": duty,
        "purity": purity,
    })
