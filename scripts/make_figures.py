"""Figures du projet SAN-2, générées depuis outputs/results.

Figures actuelles :
    training_curves             : loss et accuracy val par epoch, 3 seeds, epoch retenue marquée
    noise_pilot                 : accuracy et NLL sur val en fonction de sigma, niveaux leger/moyen/fort
    calibration_temp_scaling    : température optimale T, NLL (val) et ECE (calib) avant/après T
    conformal_shift             : couverture, taille d'ensemble et abstention sous bruit et OOD
    conformal_ood_adaptation    : impact des sources de calibration OOD (id / ood_few_shot / mixed)
    conformal_alpha_sensitivity : couverture empirique vs cible 1 - alpha et taille des ensembles

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

VARIANT_LABELS = {
    ("lac", "marginal"): "LAC marginal",
    ("lac", "class_conditional"): "LAC par classe",
    ("aps", "marginal"): "APS marginal",
    ("aps", "class_conditional"): "APS par classe",
}
VARIANT_STYLES = {
    ("lac", "marginal"): ("o-", "tab:blue"),
    ("lac", "class_conditional"): ("o--", "tab:cyan"),
    ("aps", "marginal"): ("s-", "tab:orange"),
    ("aps", "class_conditional"): ("s--", "tab:red"),
}


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
        best = int(np.argmin(vl))
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
    for a, col, ylabel in [(ax[0], "acc", "accuracy (val)"),
                           (ax[1], "nll", "NLL (val)")]:
        for s, g in df.groupby("seed"):
            a.plot(g["sigma"], g[col], "-", color=COLORS.get(s), alpha=0.35, lw=1, label=f"seed {s}")
        m = df.groupby("sigma")[col].mean()
        sd = df.groupby("sigma")[col].std()
        a.plot(m.index, m.values, "k-", lw=2, label="moyenne")
        a.fill_between(m.index, m - sd, m + sd, color="k", alpha=0.12)
        for name, sig in levels.items():
            a.axvline(sig, ls=":", color="gray")
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

        ax[0].bar(x, df["T"], color=colors, width=0.5)
        ax[0].axhline(1.0, ls="--", color="k", lw=1, label="T = 1.0")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("Température T")
        ax[0].set_title("Température optimale")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3, axis="y")

        ax[1].bar(x - w / 2, df["nll_before"], width=w, label="avant T", alpha=0.75)
        ax[1].bar(x + w / 2, df["nll_after"], width=w, label="après T", alpha=0.9)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels)
        ax[1].set_ylabel("NLL")
        ax[1].set_title("NLL (val)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")

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


def _load_conformal_summary(res_dir, filename="conformal_final_summary.csv"):
    path = res_dir / filename
    if not path.exists():
        print(f"{filename} introuvable : figure ignorée")
        return None
    return pd.read_csv(path)


def _pick_main_alpha(df):
    return 0.1 if 0.1 in df["alpha"].unique() else sorted(df["alpha"].unique())[0]


def fig_conformal_shift(res_dir, out_dir):
    """Couverture, taille d'ensemble et abstention sous décalage (calibration ID)."""
    df = _load_conformal_summary(res_dir, "conformal_final_summary.csv")
    if df is None:
        return

    alpha = _pick_main_alpha(df)
    sub = df[(df["alpha"] == alpha) & (df["source"] == "id")].copy()
    if sub.empty:
        return

    # Ordre canonique des régimes de test
    regimes = [
        ("test_id", "clean", "ID\n(propre)"),
        ("test_id_noise", "leger", "Bruit\nléger"),
        ("test_id_noise", "moyen", "Bruit\nmoyen"),
        ("test_id_noise", "fort", "Bruit\nfort"),
        ("test_ood", "clean", "OOD\n(externe)"),
    ]
    present = [(es, sl, lbl) for es, sl, lbl in regimes
               if not sub[(sub["eval_set"] == es) & (sub["shift_level"] == sl)].empty]
    if not present:
        return

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), constrained_layout=True)
    x = np.arange(len(present))

    for (score, cond), label in VARIANT_LABELS.items():
        style, color = VARIANT_STYLES[(score, cond)]
        ys_cov, sd_cov, ys_sz, sd_sz, ys_ab, sd_ab = [], [], [], [], [], []
        for es, sl, _ in present:
            row = sub[(sub["eval_set"] == es) & (sub["shift_level"] == sl)
                      & (sub["score"] == score) & (sub["conditioning"] == cond)]
            if row.empty:
                ys_cov.append(np.nan); sd_cov.append(0.0)
                ys_sz.append(np.nan); sd_sz.append(0.0)
                ys_ab.append(np.nan); sd_ab.append(0.0)
            else:
                r = row.iloc[0]
                ys_cov.append(r["coverage"]); sd_cov.append(r.get("coverage_std", 0.0))
                ys_sz.append(r["mean_set_size"]); sd_sz.append(r.get("mean_set_size_std", 0.0))
                ys_ab.append(r["abstention_rate"]); sd_ab.append(r.get("abstention_rate_std", 0.0))

        ys_cov, sd_cov = np.array(ys_cov), np.array(sd_cov)
        ys_sz, sd_sz = np.array(ys_sz), np.array(sd_sz)
        ys_ab, sd_ab = np.array(ys_ab), np.array(sd_ab)

        ax[0].plot(x, ys_cov, style, color=color, lw=1.8, label=label)
        ax[0].fill_between(x, ys_cov - sd_cov, ys_cov + sd_cov, color=color, alpha=0.12)

        ax[1].plot(x, ys_sz, style, color=color, lw=1.8, label=label)
        ax[1].fill_between(x, ys_sz - sd_sz, ys_sz + sd_sz, color=color, alpha=0.12)

        ax[2].plot(x, ys_ab, style, color=color, lw=1.8, label=label)
        ax[2].fill_between(x, ys_ab - sd_ab, ys_ab + sd_ab, color=color, alpha=0.12)

    ax[0].axhline(1.0 - alpha, ls=":", color="k", lw=1.2, label=f"cible 1-α = {1-alpha:.2f}")
    ax[0].set(ylabel="Couverture empirique", title="Couverture (calibrée sur ID)")
    ax[1].set(ylabel="Taille moyenne |C(x)|", title="Taille des ensembles")
    ax[2].set(ylabel="Taux d'abstention (|C(x)| ≠ 1)", title="Abstention")

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([lbl for _, _, lbl in present], fontsize=8.5)
        a.grid(alpha=0.3)

    ax[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle(f"Robustesse de la prédiction conforme sous décalage (α = {alpha}, moyenne ± écart-type)", fontsize=10)
    save(fig, out_dir, "conformal_shift")


def fig_conformal_ood_adaptation(res_dir, out_dir):
    """Comparaison des sources de calibration sur test_ood (id vs ood_few_shot vs mixed)."""
    df = _load_conformal_summary(res_dir, "conformal_final_summary.csv")
    if df is None:
        return

    alpha = _pick_main_alpha(df)
    sub = df[(df["alpha"] == alpha) & (df["eval_set"] == "test_ood")].copy()
    if sub.empty:
        return

    sources_order = [s for s in ("id", "ood_few_shot", "mixed") if s in sub["source"].unique()]
    source_labels = {"id": "ID seul", "ood_few_shot": "OOD few-shot", "mixed": "Mixte (ID + OOD)"}
    variants = [v for v in VARIANT_LABELS if not sub[(sub["score"] == v[0]) & (sub["conditioning"] == v[1])].empty]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    x = np.arange(len(variants))
    w = 0.25
    offsets = np.linspace(-w, w, len(sources_order))

    for i, src in enumerate(sources_order):
        covs, cov_stds, szs, sz_stds = [], [], [], []
        for score, cond in variants:
            row = sub[(sub["source"] == src) & (sub["score"] == score) & (sub["conditioning"] == cond)]
            if row.empty:
                covs.append(np.nan); cov_stds.append(0.0)
                szs.append(np.nan); sz_stds.append(0.0)
            else:
                r = row.iloc[0]
                covs.append(r["coverage"]); cov_stds.append(r.get("coverage_std", 0.0))
                szs.append(r["mean_set_size"]); sz_stds.append(r.get("mean_set_size_std", 0.0))

        ax[0].bar(x + offsets[i], covs, yerr=cov_stds, width=w, capsize=3,
                  label=source_labels.get(src, src), alpha=0.88)
        ax[1].bar(x + offsets[i], szs, yerr=sz_stds, width=w, capsize=3,
                  label=source_labels.get(src, src), alpha=0.88)

    ax[0].axhline(1.0 - alpha, ls="--", color="k", lw=1, label=f"cible {1-alpha:.2f}")
    ax[0].set(ylabel="Couverture sur test_ood", title="Couverture OOD par source de calibration")
    ax[1].set(ylabel="Taille moyenne |C(x)|", title="Taille moyenne des ensembles sur test_ood")

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=8.5)
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=8)

    fig.suptitle(f"Adaptation de la calibration sur test_ood (α = {alpha})", fontsize=10)
    save(fig, out_dir, "conformal_ood_adaptation")


def fig_conformal_alpha_sensitivity(res_dir, out_dir):
    """Couverture empirique vs cible 1 - alpha et compromis taille d'ensemble sur test_id."""
    df = _load_conformal_summary(res_dir, "conformal_final_summary.csv")
    if df is None:
        return

    sub = df[(df["eval_set"] == "test_id") & (df["source"] == "id")].copy()
    if sub.empty or sub["alpha"].nunique() < 2:
        return

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    alphas = sorted(sub["alpha"].unique())
    targets = [1.0 - a for a in alphas]

    ax[0].plot(targets, targets, "k--", lw=1.2, label="idéal (couverture = 1 - α)")

    for (score, cond), label in VARIANT_LABELS.items():
        g = sub[(sub["score"] == score) & (sub["conditioning"] == cond)].sort_values("alpha", ascending=False)
        if g.empty:
            continue
        style, color = VARIANT_STYLES[(score, cond)]
        t_vals = 1.0 - g["alpha"].values
        ax[0].errorbar(t_vals, g["coverage"].values, yerr=g.get("coverage_std", 0.0).values,
                       fmt=style, color=color, lw=1.6, capsize=3, label=label)
        ax[1].errorbar(t_vals, g["mean_set_size"].values, yerr=g.get("mean_set_size_std", 0.0).values,
                       fmt=style, color=color, lw=1.6, capsize=3, label=label)

    ax[0].set(xlabel="Couverture cible (1 - α)", ylabel="Couverture empirique (test_id)",
              title="Validité marginale sur test_id")
    ax[1].set(xlabel="Couverture cible (1 - α)", ylabel="Taille moyenne |C(x)|",
              title="Coût en taille d'ensemble")

    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

    fig.suptitle("Sensibilité au niveau de risque α sur test_id (moyenne ± écart-type sur 3 seeds)", fontsize=10)
    save(fig, out_dir, "conformal_alpha_sensitivity")


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
    fig_conformal_shift(res_dir, out_dir)
    fig_conformal_ood_adaptation(res_dir, out_dir)
    fig_conformal_alpha_sensitivity(res_dir, out_dir)


if __name__ == "__main__":
    main()