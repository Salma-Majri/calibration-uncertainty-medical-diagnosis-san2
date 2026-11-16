"""Temperature scaling, ajusté sur `val` uniquement (un T par checkpoint / seed).

La température ne change pas l'argmax : l'accuracy et le macro-F1 restent identiques.
Elle sert à recalibrer les probabilités (NLL, ECE) et les scores conformes.

Usage :
    python src/calibration.py
Sortie : outputs/results/temp_scaling.json

Diagnostic : l'ECE avant/après est aussi rapporté sur `calib` (jeu non utilisé pour ajuster T,
et sur lequel aucun choix n'est fait).
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from metrics import ece, nll, softmax

ROOT = Path(__file__).resolve().parent.parent


def load_logits_labels(split, seed, tag=""):
    """Charge outputs/logits/{split}_seed{seed}{tag}.npy et {split}_labels.npy."""
    d = ROOT / "outputs" / "logits"
    logits = np.load(d / f"{split}_seed{seed}{tag}.npy")
    labels = np.load(d / f"{split}_labels.npy")
    assert len(logits) == len(labels), f"{split}: logits et labels de tailles différentes"
    return logits, labels


def nll_from_logits(logits, y, T):
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / T, axis=1)
    return float(-lp[np.arange(len(y)), y].mean())


def fit_temperature(logits, y, log_bounds=(-3.0, 3.0)):
    """T > 0 minimisant la NLL (recherche bornée sur log T, T entre ~0.05 et ~20)."""
    res = minimize_scalar(lambda t: nll_from_logits(logits, y, float(np.exp(t))),
                          bounds=log_bounds, method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits, T):
    return np.asarray(logits, dtype=np.float64) / T


def main():
    from utils import ensure_dir, load_config

    cfg = load_config()
    out = {}
    for seed in cfg["seeds"]["training"]:
        lv, yv = load_logits_labels("val", seed)
        lc, yc = load_logits_labels("calib", seed)
        T = fit_temperature(lv, yv)
        out[str(seed)] = {
            "T": T,
            "val_nll_before": nll(softmax(lv), yv), "val_nll_after": nll(softmax(lv, T), yv),
            "val_ece_before": ece(softmax(lv), yv), "val_ece_after": ece(softmax(lv, T), yv),
            "calib_nll_before": nll(softmax(lc), yc), "calib_nll_after": nll(softmax(lc, T), yc),
            "calib_ece_before": ece(softmax(lc), yc), "calib_ece_after": ece(softmax(lc, T), yc),
        }
        r = out[str(seed)]
        print(f"seed {seed}: T = {T:.3f} | NLL val {r['val_nll_before']:.3f} -> {r['val_nll_after']:.3f} "
              f"| ECE calib {r['calib_ece_before']:.3f} -> {r['calib_ece_after']:.3f}")

    path = ensure_dir("outputs/results") / "temp_scaling.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("écrit :", path)


if __name__ == "__main__":
    main()