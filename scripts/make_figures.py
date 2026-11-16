"""Figures du projet SAN-2, générées depuis outputs/results.

Figures actuelles :
    training_curves : loss et accuracy val par epoch, 3 seeds, epoch retenue marquée
    noise_pilot     : accuracy et NLL sur val en fonction de sigma, niveaux leger/moyen/fort

Usage :
    python scripts/make_figures.py
    python scripts/make_figures.py --noise_tag _grid2
Sorties : outputs/figures/*.png et *.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("écrit :", out_dir / f"{name}.png")


def fig_training(res_dir, out_dir):
    files = sorted(res_dir.glob("train_main_seed*.json"))
    if not files:
        print("train_main_seed*.json introuvables : figure training_curves ignorée")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        s, h = d["seed"], d["history"]
        ep = np.array([r["epoch"] for r in h])
        tr = np.array([r["train_loss"] for r in h])
        vl = np.array([r["val_loss"] for r in h])
        va = np.array([r["val_acc"] for r in h])
        c = COLORS.get(s)
        best = int(np.argmin(vl))                     # critère du protocole : val_loss
        ax[0].plot(ep, tr, "--", color=c, alpha=0.6)
        ax[0].plot(ep, vl, "-", color=c, label=f"seed {s}")
        ax[0].plot(ep[best], vl[best], "o", color=c)
        ax[1].plot(ep, va, "-", color=c, label=f"seed {s}")
        ax[1].plot(ep[best], va[best], "o", color=c)
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Loss (pointillé : train, plein : val)")
    ax[1].set(xlabel="epoch", ylabel="accuracy (val)", title="Accuracy val")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.suptitle("Entraînement ResNet-50 (point = epoch retenue, min val_loss)", y=1.02, fontsize=10)
    save(fig, out_dir, "training_curves")


def fig_noise(res_dir, out_dir, tag):
    csv = res_dir / f"noise_pilot{tag}.csv"
    lv = res_dir / f"noise_pilot_levels{tag}.json"
    if not csv.exists():
        print(f"{csv.name} introuvable : figure noise_pilot ignorée")
        return
    df = pd.read_csv(csv)
    levels = {}
    if lv.exists():
        with open(lv) as fh:
            levels = json.load(fh).get("sigma_levels", {})

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, col, ylabel, loc in [(ax[0], "acc", "accuracy (val)", "upper right"),
                                (ax[1], "nll", "NLL (val)", "upper left")]:
        for s, g in df.groupby("seed"):
            a.plot(g["sigma"], g[col], "-", color=COLORS.get(s), alpha=0.35, lw=1, label=f"seed {s}")
        m = df.groupby("sigma")[col].mean()
        sd = df.groupby("sigma")[col].std()
        a.plot(m.index, m.values, "k-", lw=2, label="moyenne")
        a.fill_between(m.index, m - sd, m + sd, color="k", alpha=0.12)
        for name, sig in levels.items():
            a.axvline(sig, ls=":", color="gray")
            # étiquettes en haut du graphe pour ne pas chevaucher les courbes
            a.text(sig, 0.98, name, transform=a.get_xaxis_transform(), rotation=90,
                   ha="right", va="top", fontsize=8)
        a.set(xlabel="sigma (image en [0,1], avant normalisation)", ylabel=ylabel)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Pilote du bruit gaussien sur val (moyenne ± écart-type sur 3 seeds)", y=1.02, fontsize=10)
    save(fig, out_dir, f"noise_pilot{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="outputs/results")
    ap.add_argument("--out", default="outputs/figures")
    ap.add_argument("--noise_tag", default="_grid2")
    args = ap.parse_args()

    res_dir = ROOT / args.results_dir
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_training(res_dir, out_dir)
    fig_noise(res_dir, out_dir, args.noise_tag)


if __name__ == "__main__":
    main()