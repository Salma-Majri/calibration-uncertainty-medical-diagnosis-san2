"""Telecharge FedISIC2019 a la revision figee dans protocol.yaml et verifie qu'il correspond au protocole.

Usage :
    python data/download.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_config, load_hf_dataset, load_split_indices  # noqa: E402


def main():
    cfg = load_config()
    d = cfg["dataset"]

    print(f"Dataset  : {d['name']}")
    print(f"Revision : {d['revision']}")
    hf = load_hf_dataset(cfg)

    n = len(hf)
    centers = Counter(hf["center"])
    centers = {int(k): v for k, v in sorted(centers.items())}
    ok = True

    if n != d["n_images"]:
        print(f"ERREUR : {n} images, protocole attend {d['n_images']}")
        ok = False
    if centers != {int(k): v for k, v in d["center_counts"].items()}:
        print(f"ERREUR : centres {centers}, protocole attend {d['center_counts']}")
        ok = False

    splits_path = ROOT / cfg["splits"]["file"]
    if splits_path.exists():
        splits = load_split_indices(cfg)
        total = sum(len(v) for v in splits.values())
        if total != n:
            print(f"ERREUR : splits couvrent {total} images sur {n}")
            ok = False
        for k, v in splits.items():
            exp = cfg["splits"]["realized_sizes"].get(k)
            if exp is not None and len(v) != exp:
                print(f"ERREUR : split {k} = {len(v)}, protocole attend {exp}")
                ok = False
    else:
        print(f"Splits absents ({splits_path}) : lancer python src/splits.py")

    print(f"Images : {n} | Centres : {centers}")
    print("OK : dataset conforme au protocole." if ok else "ECHEC : dataset non conforme.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()