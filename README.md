# SAN-2 : Incertitude calibrée et abstention en aide au diagnostic

Projet de fin de module, IA avancée (Master SDIA, ENSET Mohammedia, 2026-2027).

## Question de recherche

La prédiction conforme tient-elle sa garantie de couverture sur une tâche de diagnostic
lorsque la distribution de test change (nouveau centre d'acquisition, sous-population rare),
et quel taux d'abstention faut-il payer pour la rétablir ?

## Données

- **FedISIC2019** (`flwrlabs/fed-isic2019` sur Hugging Face), 18 597 images dermoscopiques, 8 classes, 6 centres.
- Seul le split HF `train` est utilisé. Le split HF `test` n'est pas utilisé.
- Les images ne sont pas redistribuées dans ce dépôt (licence ISIC) : elles sont téléchargées par script.
- Centre OOD principal : **3** (non vu à l'entraînement).

## Méthodes comparées

- Prédiction conforme fractionnée (scores LAC et APS, marginale et par classe)
- MC-Dropout
- Ensembles (3 modèles)

Calibration conforme sur le centre source (ID), sur un petit échantillon du centre OOD, ou mixte.

## Protocole

Le protocole est figé avant les résultats dans [`configs/protocol.yaml`](configs/protocol.yaml)
(splits, seeds, hyperparamètres, niveaux alpha, règles d'usage des jeux de données, limites connues).

| Split | Origine | Taille | Usage |
|---|---|---|---|
| train | centres 0, 1, 2, 4, 5 | 10 913 | entraînement |
| val | centres sources | 1 679 | early stopping, réglages |
| calib | centres sources | 2 518 | calibration conforme ID |
| test_id | centres sources | 1 680 | évaluation ID (une seule fois) |
| ood_calib_pool | centre 3 | 361 | calibration conforme OOD |
| test_ood | centre 3 | 1 446 | évaluation OOD (une seule fois) |

Seeds d'entraînement : 0, 1, 2. Seed des splits : 0.

## Structure du dépôt

```
configs/protocol.yaml     protocole figé
data/download.py          téléchargement et vérification des données (revision figée)
data/splits/              indices des splits (JSON versionnés)
src/splits.py             génération des splits
src/utils.py              config, seeds, chargement du dataset et des splits
src/dataset.py            Dataset PyTorch (bruit gaussien d'acquisition optionnel)
src/train.py              entraînement
src/extract_logits.py     extraction des logits par split (CPU/GPU)
src/metrics.py            accuracy, F1, NLL, ECE, couverture, taille des ensembles, abstention, risque sélectif
src/calibration.py        temperature scaling (ajusté sur val)
src/conformal.py          prédiction conforme (LAC, APS, marginale, par classe, sources de calibration)
src/run_conformal.py      lance la prédiction conforme sur les logits (modes smoke et final)
src/mc_dropout.py         MC-Dropout (à venir)
src/ensembles.py          ensembles (à venir)
scripts/noise_pilot.py    pilote du bruit gaussien sur val (choix des niveaux sigma)
scripts/make_figures.py   figures (courbes d'entraînement, pilote du bruit)
scripts/run_main.sh       relance l'expérience principale (à venir)
scripts/run_ablations.sh  ablations (à venir)
notebooks/error_analysis.ipynb   (à venir)
outputs/                  logits, résultats, figures (non versionnés)
paper/                    article et figures
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduire

Étapes sur GPU (Colab T4) :

```bash
python data/download.py                                   # vérifie le dataset (revision figée)
python src/splits.py --main_ood_center 3 --seed 0 --out data/splits/splits_ood3_seed0.json
python src/train.py --seed 0                              # idem pour --seed 1 et --seed 2
python src/extract_logits.py --seed 0                     # idem pour --seed 1 et --seed 2
python scripts/noise_pilot.py --tag _grid2                # pilote du bruit sur val
# logits avec bruit sur test_id, pour chaque seed et chaque sigma de protocol.yaml
python src/extract_logits.py --seed 0 --splits test_id --noise_sigma 0.008   # puis 0.01, 0.015 ; seeds 1 et 2
```

L'entraînement a été exécuté sur Colab (GPU T4), environ 30 à 40 minutes par seed.
`outputs/` et `checkpoints/` ne sont pas versionnés : ils se régénèrent avec les commandes ci-dessus.

Étapes sans GPU (à partir des logits dans `outputs/logits/`) :

```bash
python src/calibration.py                                 # temperature scaling, un T par seed, ajusté sur val
python src/run_conformal.py --mode smoke                  # calib -> val : vérifie le pipeline, aucun jeu de test lu
python src/run_conformal.py --mode final                  # test_id, test_id bruité, test_ood (une seule fois, arbre git propre)
python src/run_conformal.py --mode final --no_ts          # ablation sans temperature scaling
python src/run_conformal.py --mode final --n_ood 50 100 361   # ablation taille du pool OOD
python scripts/make_figures.py --noise_tag _grid2
```

Le mode `final` refuse de s'exécuter si le fichier de sortie existe déjà ou si l'arbre git est modifié.

Expérience principale (commande unique, à venir) :

```bash
bash scripts/run_main.sh
```

## État d'avancement

- [x] Structure du projet
- [x] Splits (`src/splits.py`)
- [x] Protocole figé (`configs/protocol.yaml`), sigma_levels déjà fixés (0.008 / 0.010 / 0.015)
- [x] Dataset et entraînement (`src/dataset.py`, `src/train.py`, seeds 0, 1, 2)
- [x] Extraction des logits (`src/extract_logits.py`)
- [x] Pilote du bruit (sigma) sur val
- [x] Métriques et temperature scaling
- [x] Prédiction conforme (LAC, APS, marginale, par classe)
- [ ] MC-Dropout
- [ ] Ensembles
- [ ] Ablations
- [ ] Analyse des erreurs et figures
- [ ] Article

## Limites connues

Effectifs par classe (classes 0 à 7) :

| Split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| calib | 458 | 1265 | 364 | 94 | 232 | 25 | 29 | 51 |
| test_id | 306 | 844 | 243 | 62 | 155 | 16 | 20 | 34 |
| ood_calib_pool | 55 | 130 | 44 | 17 | 81 | 4 | 1 | 29 |
| test_ood | 221 | 519 | 179 | 69 | 325 | 16 | 2 | 115 |

- Classes rares (5 et 6) : les couvertures par classe sur test_id et test_ood sont très bruitées.
  Sur test_ood, la classe 6 n'a que 2 images : sa couverture n'est pas interprétable.
- `ood_calib_pool` (361 images) : les classes 3 (17), 5 (4) et 6 (1) ont moins de 20 exemples.
  La calibration par classe y est impossible pour ces classes : repli sur le quantile global
  (seuil `class_conditional_min_samples = 20`). Le repli s'étend à d'autres classes quand le pool
  est sous-échantillonné (ablation à 50 et 100 images).
- Score APS déterministe (non randomisé) : couverture garantie mais légèrement conservatrice.
- Calibration mixte : mélange simple de calib et d'un échantillon OOD, sans pondération.
- Les métriques par classe pour les classes 5 et 6 sont indicatives.

## Équipe

- Salma Majri
- Fatima zahra Massane

Encadrant : Pr. Soufiane HAMIDA.