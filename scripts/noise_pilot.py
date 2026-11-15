"""Pilote du bruit gaussien d'acquisition, sur `val` uniquement.

Mesure la chute d'accuracy en fonction de sigma (image en [0,1], avant normalisation ImageNet),
pour les checkpoints des seeds du protocole, puis propose 3 niveaux (leger, moyen, fort)
dont la chute moyenne d'accuracy est la plus proche des cibles --target_drops.


Usage :
    python scripts/noise_pilot.py
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
from train import build_model, make_loader  # noqa: E402
from utils import (ensure_dir, get_device, load_config, load_hf_dataset,  # noqa: E402
                   load_split_indices, set_seed)

DEFAULT_GRID = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
DEFAULT_TARGET_DROPS = [0.02, 0.05, 0.10]


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out)


def load_checkpoint(ckpt_path: Path, device: torch.device):
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


def select_sigma_levels(acc_drop_series: pd.Series, target_drops):
    """Choisit 3 sigmas distincts (leger, moyen, fort) proches des cibles."""
    levels = {}
    used = set()
    for name, target in zip(["leger", "moyen", "fort"], target_drops):
        diffs = (acc_drop_series - target).abs().sort_values()
        chosen = None
        for sigma_idx in diffs.index:
            sigma_val = float(sigma_idx)
            if sigma_val not in used:
                chosen = sigma_val
                used.add(sigma_val)
                break
        if chosen is None:
            chosen = float(diffs.index[0])
        levels[name] = chosen
        if acc_drop_series.max() < target:
            print(
                f"Attention : la chute maximale ({acc_drop_series.max():.4f}) "
                f"n'atteint pas la cible {target:.4f}."
            )
    return levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--grid", type=float, nargs="+", default=None)
    ap.add_argument(
        "--target_drops",
        type=float,
        nargs=3,
        default=None,
        help="chutes d'accuracy visees pour leger, moyen, fort",
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    shift_cfg = cfg.get("shift_simulation", {})
    noise_cfg = shift_cfg.get("acquisition_noise", {}) if isinstance(shift_cfg, dict) else {}

    grid = args.grid if args.grid is not None else noise_cfg.get("grid", DEFAULT_GRID)
    target_drops = (
        args.target_drops
        if args.target_drops is not None
        else noise_cfg.get("target_drops", DEFAULT_TARGET_DROPS)
    )

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
        model.load_state_dict(load_checkpoint(ckpt, device))
        model.to(device)

        for sigma in grid:
            set_seed(seed)
            ds = ISICDataset(hf, val_idx, build_transform(cfg, train=False, noise_sigma=sigma))
            loader = make_loader(ds, args.batch_size, False, seed, args.num_workers)
            logits = get_logits(model, loader, device)
            pred = logits.argmax(1).numpy()
            row = {
                "seed": seed,
                "sigma": float(sigma),
                "acc": float((pred == labels).mean()),
                "macro_f1": float(f1_score(labels, pred, average="macro")),
                "nll": float(F.cross_entropy(logits, y).item()),
            }
            rows.append(row)
            print(row)

    res_dir = Path(ensure_dir("outputs/results"))
    df = pd.DataFrame(rows)
    df.to_csv(res_dir / "noise_pilot.csv", index=False)

    g = df.groupby("sigma")
    summary = pd.DataFrame({
        "acc_mean": g["acc"].mean(),
        "acc_std": g["acc"].std(),
        "macro_f1_mean": g["macro_f1"].mean(),
        "nll_mean": g["nll"].mean(),
    })
    base = summary.loc[0.0, "acc_mean"] if 0.0 in summary.index else summary["acc_mean"].iloc[0]
    summary["acc_drop"] = base - summary["acc_mean"]
    summary.to_csv(res_dir / "noise_pilot_summary.csv")
    print("\n", summary.round(4).to_string())

    levels = select_sigma_levels(summary["acc_drop"], target_drops)
    print("\nNiveaux proposes (cibles de chute d'accuracy", target_drops, ") :", levels)

    with open(res_dir / "noise_pilot_levels.json", "w") as f:
        json.dump(
            {
                "target_drops": list(target_drops),
                "grid": list(grid),
                "sigma_levels": levels,
            },
            f,
            indent=2,
        )
    print("A recopier dans configs/protocol.yaml (commit dedie).")


if __name__ == "__main__":
    main()