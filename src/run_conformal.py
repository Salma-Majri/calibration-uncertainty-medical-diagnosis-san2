"""Lance la prediction conforme sur les logits extraits.

Deux modes :
    smoke : calibration sur calib, evaluation sur val (aucun jeu de test lu). Sert a verifier
            que le pipeline tourne et que la couverture est proche de 1 - alpha.
    final : evaluation sur test_id, test_id bruite (3 niveaux de protocol.yaml) et test_ood.

Calibration :
    test_id (propre ou bruite) : calib (ID)
    test_ood                   : calib (ID), ood_calib_pool (few-shot) ou calib + ood_calib_pool (mixte)


Usage :
    python src/run_conformal.py --mode smoke
    python src/run_conformal.py --mode final
    python src/run_conformal.py --mode final --no_ts          # ablation sans temperature scaling
    python src/run_conformal.py --mode final --n_ood 50 100 361   # ablation taille du pool OOD

Sorties (outputs/results/) : conformal_{mode}{tag}.csv (une ligne par configuration et par seed),
conformal_{mode}{tag}_summary.csv (moyenne +- ecart-type sur les seeds), model_metrics_{mode}{tag}.csv,
conformal_{mode}{tag}_meta.json.
"""
import argparse
import itertools
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from calibration import load_logits_labels
from conformal import build_calibration_set, calibrate, predict_sets
from metrics import evaluate_probs, evaluate_sets, softmax
from utils import ROOT, ensure_dir, load_config

K = 8


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:
        return {"commit": None, "dirty": None}


def eval_sets(mode, cfg, sources):
    """Jeux d'evaluation : (nom, split, tag de fichier logits, niveau de bruit, sources de calibration)."""
    if mode == "smoke":
        return [("val", "val", "", "clean", ["id"])]
    out = [("test_id", "test_id", "", "clean", ["id"])]
    sig = cfg["shift_simulation"]["acquisition_noise"]["sigma_levels"]
    for name, s in sig.items():
        out.append(("test_id_noise", "test_id", f"_sigma{s:g}", name, ["id"]))
    out.append(("test_ood", "test_ood", "", "clean", [s for s in sources]))
    return out


def flat(res):
    d = {k: v for k, v in res.items() if k not in ("coverage_per_class", "n_per_class")}
    for c, v in enumerate(res["coverage_per_class"]):
        d[f"cov_c{c}"] = np.nan if v is None else v
    return d


def summarize(df):
    keys = ["eval_set", "shift_level", "source", "n_ood", "T_mode", "score", "conditioning", "alpha"]
    metrics = [c for c in df.columns if c in
               {"coverage", "mean_set_size", "singleton_rate", "empty_rate", "abstention_rate",
                "selective_risk"} or c.startswith("cov_c")]
    per_seed = df.groupby(keys + ["seed"], dropna=False)[metrics].mean().reset_index()   # moyenne sur les tirages
    g = per_seed.groupby(keys, dropna=False)[metrics]
    m, s = g.mean(), g.std()
    s.columns = [c + "_std" for c in s.columns]
    return pd.concat([m, s], axis=1).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "final"], required=True)
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--no_ts", action="store_true", help="T = 1 (ablation sans temperature scaling)")
    ap.add_argument("--n_ood", type=int, nargs="+", default=None,
                    help="tailles du pool OOD utilisees pour calibrer (defaut : tout le pool)")
    ap.add_argument("--n_draws", type=int, default=10,
                    help="tirages aleatoires quand n_ood < taille du pool")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_dirty", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cc = cfg["conformal"]
    alphas = sorted({cc["alpha_main"], *cc.get("alpha_sensitivity", [])})
    scores, conds = cc["scores"], cc["conditioning"]
    min_samples = cc.get("class_conditional_min_samples", 20)
    sources = cc["calibration_sources"]
    tag = "_noTS" if args.no_ts else ""
    out_dir = ensure_dir("outputs/results")
    out_csv = out_dir / f"conformal_{args.mode}{tag}{'' if args.n_ood is None else '_nood'}.csv"

    gs = git_state()
    if args.mode == "final":
        if out_csv.exists() and not args.force:
            sys.exit(f"{out_csv.name} existe deja : le mode final ne se lance qu'une fois (--force pour ecraser).")
        if gs["dirty"] and not args.allow_dirty:
            sys.exit("Arbre git modifie : commit d'abord le code et le protocole, puis relance (--allow_dirty pour forcer).")

    with open(out_dir / "temp_scaling.json") as f:
        ts = json.load(f)
    plan = eval_sets(args.mode, cfg, sources)
    rows, model_rows, Ts = [], [], {}

    for seed in cfg["seeds"]["training"]:
        T = 1.0 if args.no_ts else ts[str(seed)]["T"]
        Ts[seed] = T
        lc, yc = load_logits_labels("calib", seed)
        pc = softmax(lc, T)
        pool = None
        if any(s != "id" for *_, srcs in plan for s in srcs):
            lo, yo = load_logits_labels("ood_calib_pool", seed)
            pool = (softmax(lo, T), yo)

        for name, split, ftag, level, srcs in plan:
            try:
                lt, yt = load_logits_labels(split, seed, ftag)
            except FileNotFoundError as e:
                sys.exit(f"Logits absents : {e.filename}\nLance d'abord extract_logits.py "
                         f"(--splits {split}{' --noise_sigma ' + ftag[6:] if ftag else ''}).")
            pt = softmax(lt, T)
            model_rows.append({"seed": seed, "eval_set": name, "shift_level": level,
                               "T": T, **evaluate_probs(pt, yt)})

            for src in srcs:
                if src == "id":
                    cal_opts = [(0, 0)]
                else:
                    full = len(pool[1])
                    sizes = [full] if args.n_ood is None else args.n_ood
                    cal_opts = [(n, d) for n in sizes for d in range(1 if n >= full else args.n_draws)]
                for n_ood, draw in cal_opts:
                    if src == "id":
                        P, Y = build_calibration_set("id", pc, yc)
                    else:
                        P, Y = build_calibration_set(src, pc, yc, pool[0], pool[1],
                                                     n_ood=n_ood, seed=seed * 1000 + draw)
                    for score, cond, alpha in itertools.product(scores, conds, alphas):
                        cal = calibrate(P, Y, alpha, score, cond, min_samples)
                        res = evaluate_sets(predict_sets(pt, cal), yt, K)
                        rows.append({"seed": seed, "eval_set": name, "shift_level": level,
                                     "source": src, "n_ood": n_ood, "draw": draw,
                                     "T_mode": "none" if args.no_ts else "ts",
                                     "score": score, "conditioning": cond, "alpha": alpha,
                                     "n_cal": cal.n_cal, "q_global": cal.q_global,
                                     "n_fallback": int(cal.fallback.sum()), **flat(res)})
        print(f"seed {seed} : T = {T:.3f}, {len(rows)} lignes")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    summ = summarize(df)
    summ.to_csv(out_csv.with_name(out_csv.stem + "_summary.csv"), index=False)
    pd.DataFrame(model_rows).to_csv(out_dir / f"model_metrics_{args.mode}{tag}.csv", index=False)
    with open(out_csv.with_name(out_csv.stem + "_meta.json"), "w") as f:
        json.dump({"mode": args.mode, "args": vars(args), "alphas": alphas, "T": Ts, **gs}, f, indent=2)

    main_a = cc["alpha_main"]
    show = summ[(summ.alpha == main_a) & (summ.score == "lac") & (summ.conditioning == "marginal")]
    cols = ["eval_set", "shift_level", "source", "n_ood", "coverage", "coverage_std",
            "abstention_rate", "mean_set_size"]
    print(f"\nLAC marginal, alpha = {main_a} (moyenne sur seeds) :")
    print(show[cols].round(3).to_string(index=False))
    print("\necrit :", out_csv)


if __name__ == "__main__":
    main()