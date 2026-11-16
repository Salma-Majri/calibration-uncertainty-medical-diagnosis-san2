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


def fig_calibration(res_dir, out_dir):
    ts_path = res_dir / "temp_scaling.json"
    if not ts_path.exists():
        print("temp_scaling.json introuvable : figure calibration ignorée")
        return

    with open(ts_path) as fh:
        ts_data = json.load(fh)

    records = []
    if isinstance(ts_data, dict):
        source = ts_data.get("per_seed", ts_data)

        for k, v in source.items():
            if k in ("target_drops", "grid", "sigma_levels", "per_seed"):
                continue

            # On accepte "0", "seed_0", "seed0", "seed 0", 0 (int)
            seed_str = str(k).lower().replace("seed", "").replace("_", "").strip()
            if not seed_str.lstrip("-").isdigit():
                continue
            s = int(seed_str)

            if isinstance(v, dict):
                T = v.get("T", v.get("temperature", v.get("temp")))
                nll_before = v.get("nll_before", v.get("val_nll_before"))
                nll_after = v.get("nll_after", v.get("val_nll_after"))
                ece_before = v.get("ece_before", v.get("calib_ece_before"))
                ece_after = v.get("ece_after", v.get("calib_ece_after"))

                if T is None:
                    print(f"[warn] seed {s} : aucune température trouvée, ligne ignorée")
                    continue

                records.append({
                    "seed": s,
                    "T": float(T),
                    "nll_before": nll_before,
                    "nll_after": nll_after,
                    "ece_before": ece_before,
                    "ece_after": ece_after,
                })
            elif isinstance(v, (int, float)):
                records.append({
                    "seed": s,
                    "T": float(v),
                    "nll_before": None,
                    "nll_after": None,
                    "ece_before": None,
                    "ece_after": None,
                })

    if not records:
        print("Format temp_scaling.json non reconnu : figure calibration ignorée")
        return

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)

    metric_cols = ["nll_before", "nll_after", "ece_before", "ece_after"]
    has_metrics = (
        all(c in df.columns for c in metric_cols)
        and df[metric_cols].notna().all().all()
    )

    def seed_color(s):
        for key in (s, str(s), f"seed_{s}", f"seed{s}"):
            if key in COLORS:
                return COLORS[key]
        return "gray"

    if has_metrics:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        seeds = df["seed"].astype(int)
        x = np.arange(len(seeds))
        w = 0.35
        labels = [f"seed {s}" for s in seeds]
        colors = [seed_color(s) for s in seeds]

        # Panel 1 : Température
        ax[0].bar(x, df["T"], color=colors, width=0.5)
        ax[0].axhline(1.0, ls="--", color="k", lw=1, label="T = 1.0")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("Température T")
        ax[0].set_title("Température optimale")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, axis="y")

        # Panel 2 : NLL avant/après
        ax[1].bar(x - w / 2, df["nll_before"], width=w, label="avant T", alpha=0.75)
        ax[1].bar(x + w / 2, df["nll_after"], width=w, label="après T", alpha=0.9)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("NLL")
        ax[1].set_title("NLL (val)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")

        # Panel 3 : ECE avant/après
        ax[2].bar(x - w / 2, df["ece_before"], width=w, label="avant T", alpha=0.75)
        ax[2].bar(x + w / 2, df["ece_after"], width=w, label="après T", alpha=0.9)
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(labels)
        ax[2].set_ylabel("ECE")
        ax[2].set_title("ECE (calib)")
        ax[2].legend(fontsize=8)
        ax[2].grid(alpha=0.3, axis="y")

        fig.suptitle("Calibration post-hoc par Temperature Scaling", fontsize=10)

    else:
        fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
        x = np.arange(len(df))
        labels = [f"seed {s}" for s in df["seed"].astype(int)]
        colors = [seed_color(s) for s in df["seed"].astype(int)]

        ax.bar(x, df["T"], color=colors, width=0.5)
        ax.axhline(1.0, ls="--", color="k", lw=1, label="T = 1.0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Température T")
        ax.set_title("Temperature Scaling par seed")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    save(fig, out_dir, "calibration_temp_scaling")

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
    fig_calibration(res_dir, out_dir)


if __name__ == "__main__":
    main()