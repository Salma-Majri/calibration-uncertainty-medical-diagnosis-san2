"""Pilote du bruit gaussien d'acquisition, sur `val` uniquement.

Mesure la chute d'accuracy en fonction de sigma (image en [0,1], avant normalisation ImageNet),
pour les checkpoints des seeds du protocole, puis propose 3 niveaux (leger, moyen, fort)
dont la chute moyenne d'accuracy est la plus proche des cibles --target_drops.
sigma = 0 (aucun bruit) est exclu des candidats.
Usage :
    python scripts/noise_pilot.py                    # ecrit outputs/results/noise_pilot*.csv/json
    python scripts/noise_pilot.py --tag _grid2       # ecrit noise_pilot_grid2*.csv/json (sans ecraser)
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import ISICDataset, build_transform, split_labels  # noqa: E402
from train import build_model, make_loader                       # noqa: E402
from utils import (ensure_dir, get_device, load_config, load_hf_dataset,  # noqa: E402
                   load_split_indices, set_seed)

DEFAULT_GRID = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.030]


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--target_drops", type=float, nargs=3, default=[0.02, 0.05, 0.10],
                    help="chutes d'accuracy visees pour leger, moyen, fort")
    ap.add_argument("--tag", default="", help="suffixe des fichiers de sortie (ex. _grid2)")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg["seeds"]["training"]
    device = get_device()

    hf = load_hf_dataset(cfg)
    val_idx = load_split_indices(cfg)["val"]
    labels = split_labels(hf, val_idx).astype(np.int64)
    y = torch.tensor(labels)

    cfg_model = copy.deepcopy(cfg)
    cfg_model["model"]["pretrained"] = None
    rows = []

    for seed in seeds:
        ckpt = ROOT / f"checkpoints/resnet50_main_seed{seed}.pt"
        if not ckpt.exists():
            sys.exit(f"Checkpoint absent : {ckpt}")
        model = build_model(cfg_model)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)

        for sigma in args.grid:
            set_seed(seed)
            ds = ISICDataset(hf, val_idx, build_transform(cfg, train=False, noise_sigma=sigma))
            loader = make_loader(ds, args.batch_size, False, seed, args.num_workers)
            logits = get_logits(model, loader, device)
            pred = logits.argmax(1).numpy()
            rows.append({
                "seed": seed, "sigma": sigma,
                "acc": float((pred == labels).mean()),
                "macro_f1": float(f1_score(labels, pred, average="macro")),
                "nll": float(F.cross_entropy(logits, y).item()),
            })
            print(rows[-1])

    res_dir = ensure_dir("outputs/results")
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / f"noise_pilot{args.tag}.csv", index=False)

    g = df.groupby("sigma")
    summary = pd.DataFrame({
        "acc_mean": g["acc"].mean(), "acc_std": g["acc"].std(),
        "macro_f1_mean": g["macro_f1"].mean(), "nll_mean": g["nll"].mean(),
    })
    base = summary.loc[0.0, "acc_mean"] if 0.0 in summary.index else summary["acc_mean"].iloc[0]
    summary["acc_drop"] = base - summary["acc_mean"]
    summary.to_csv(res_dir / f"noise_pilot_summary{args.tag}.csv")
    print("\n", summary.round(4).to_string())

    candidates = summary[summary.index > 0]      # sigma = 0 n'est pas un niveau de bruit
    levels = {}
    for name, target in zip(["leger", "moyen", "fort"], args.target_drops):
        levels[name] = float((candidates["acc_drop"] - target).abs().idxmin())
        if candidates["acc_drop"].max() < target:
            print(f"Attention : la chute maximale ({candidates['acc_drop'].max():.3f}) n'atteint pas {target}.")
    if len(set(levels.values())) < 3:
        print("Attention : plusieurs niveaux identiques, la grille est trop grossiere.")
    print("\nNiveaux proposes (cibles de chute d'accuracy", args.target_drops, ") :", levels)

    with open(res_dir / f"noise_pilot_levels{args.tag}.json", "w") as f:
        json.dump({"target_drops": args.target_drops, "grid": args.grid, "sigma_levels": levels}, f, indent=2)
    print("A recopier dans configs/protocol.yaml (commit dedie).")


if __name__ == "__main__":
    main()