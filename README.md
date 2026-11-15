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
src/dataset.py            Dataset PyTorch
src/train.py              entraînement
src/extract_logits.py     extraction des logits par split
src/conformal.py          prédiction conforme
src/mc_dropout.py         MC-Dropout
src/ensembles.py          ensembles
src/metrics.py            couverture, taille des ensembles, ECE, risque sélectif
scripts/run_main.sh       relance l'expérience principale
scripts/run_ablations.sh  ablations
notebooks/error_analysis.ipynb
outputs/                  logits, résultats, figures (non versionnés)
paper/                    article
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduire

```bash
python data/download.py                                   # vérifie le dataset (revision figée)
python src/splits.py --main_ood_center 3 --seed 0 --out data/splits/splits_ood3_seed0.json
python src/train.py --seed 0                              # idem pour --seed 1 et --seed 2
python src/extract_logits.py --seed 0                     # idem pour --seed 1 et --seed 2
```

L'entraînement a été exécuté sur Colab (GPU T4), environ 30 à 40 minutes par seed.
`outputs/` et `checkpoints/` ne sont pas versionnés : ils se régénèrent avec les commandes ci-dessus.

Expérience principale (commande unique, à venir) :

```bash
bash scripts/run_main.sh
```

## État d'avancement

- [x] Structure du projet
- [x] Splits (`src/splits.py`)
- [x] Protocole figé (`configs/protocol.yaml`), `sigma_levels` à fixer après le pilote sur val
- [x] Dataset et entraînement (`src/dataset.py`, `src/train.py`, seeds 0, 1, 2)
- [x] Extraction des logits (`src/extract_logits.py`)
- [ ] Pilote du bruit (sigma) sur val
- [ ] Métriques et temperature scaling
- [ ] Prédiction conforme (LAC, APS, marginale, par classe)
- [ ] MC-Dropout
- [ ] Ensembles
- [ ] Ablations
- [ ] Analyse des erreurs et figures
- [ ] Article

## Limites connues

- Classes rares (5 et 6) : très peu d'échantillons dans calib, test_id et test_ood.
- `ood_calib_pool` (361 images) : la classe 6 n'a qu'une image, la classe 5 en a 4. La calibration par classe n'y est pas possible, repli sur le quantile global.
- Les métriques par classe pour les classes 5 et 6 sur `test_ood` sont indicatives.

## Équipe

- Salma Majri
- Fatima zahra Massane

Encadrant : Pr. Soufiane HAMIDA.