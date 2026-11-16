"""Métriques SAN-2 : probabilités (accuracy, macro-F1, NLL, ECE) et ensembles de prédiction
(couverture, taille, abstention, risque sélectif, courbe risque-couverture).

Convention : un ensemble de prédiction est une matrice booléenne `sets` de forme [N, K]
(True si la classe est dans l'ensemble). Selon protocol.yaml, on s'abstient quand la taille
de l'ensemble est différente de 1 ; la prédiction est alors l'unique classe de l'ensemble.

"""
import numpy as np
from scipy.special import softmax as _softmax
from sklearn.metrics import f1_score


def softmax(logits, T=1.0):
    return _softmax(np.asarray(logits, dtype=np.float64) / T, axis=1)


# ----------------------------------------------------------------------
# Métriques sur les probabilités
# ----------------------------------------------------------------------
def accuracy(probs, y):
    return float((probs.argmax(1) == y).mean())


def macro_f1(probs, y):
    return float(f1_score(y, probs.argmax(1), average="macro", zero_division=0))


def nll(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def ece(probs, y, n_bins=15):
    """Expected Calibration Error, confiance = probabilité max, bins de largeur égale."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf, edges[1:-1])            # 0 .. n_bins-1
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def evaluate_probs(probs, y):
    return {"acc": accuracy(probs, y), "macro_f1": macro_f1(probs, y),
            "nll": nll(probs, y), "ece": ece(probs, y)}


# ----------------------------------------------------------------------
# Métriques sur les ensembles de prédiction
# ----------------------------------------------------------------------
def set_sizes(sets):
    return sets.sum(1)


def coverage(sets, y):
    return float(sets[np.arange(len(y)), y].mean())


def mean_set_size(sets):
    return float(set_sizes(sets).mean())


def singleton_rate(sets):
    return float((set_sizes(sets) == 1).mean())


def empty_rate(sets):
    return float((set_sizes(sets) == 0).mean())


def abstention_rate(sets):
    """Abstention = taille de l'ensemble différente de 1 (protocol.yaml)."""
    return float((set_sizes(sets) != 1).mean())


def selective_risk(sets, y):
    """Taux d'erreur parmi les exemples où l'on ne s'abstient pas (ensemble de taille 1)."""
    keep = set_sizes(sets) == 1
    if not keep.any():
        return float("nan")
    pred = sets[keep].argmax(1)
    return float((pred != y[keep]).mean())


def coverage_per_class(sets, y, n_classes):
    """Couverture par classe vraie. Retourne (couverture, effectifs) ; nan si effectif nul."""
    hit = sets[np.arange(len(y)), y]
    cov = np.full(n_classes, np.nan)
    counts = np.bincount(y, minlength=n_classes)
    for c in range(n_classes):
        if counts[c] > 0:
            cov[c] = hit[y == c].mean()
    return cov, counts


def evaluate_sets(sets, y, n_classes):
    cov_c, counts = coverage_per_class(sets, y, n_classes)
    return {
        "coverage": coverage(sets, y),
        "mean_set_size": mean_set_size(sets),
        "singleton_rate": singleton_rate(sets),
        "empty_rate": empty_rate(sets),
        "abstention_rate": abstention_rate(sets),
        "selective_risk": selective_risk(sets, y),
        "coverage_per_class": [None if np.isnan(v) else float(v) for v in cov_c],
        "n_per_class": counts.tolist(),
    }


# ----------------------------------------------------------------------
# Classification sélective (abstention par seuil de confiance)
# ----------------------------------------------------------------------
def risk_coverage_curve(probs, y, n_points=50):
    """Courbe risque-couverture : on garde les exemples les plus confiants en premier.

    Retourne (couverture, risque, aurc) ; `aurc` = risque moyen sur toutes les couvertures.
    """
    conf = probs.max(1)
    wrong = (probs.argmax(1) != y).astype(float)[np.argsort(-conf, kind="stable")]
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(wrong) / k
    cov = k / len(y)
    idx = np.linspace(0, len(y) - 1, n_points).astype(int)
    return cov[idx], risk[idx], float(risk.mean())