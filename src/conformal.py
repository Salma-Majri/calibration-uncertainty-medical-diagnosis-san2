"""Prédiction conforme fractionnée (split conformal) à partir de probabilités softmax.

Scores de non-conformité (plus grand = moins conforme) :
    lac : s(x, y) = 1 - p_y(x)
    aps : s(x, y) = somme des probabilités des classes au moins aussi probables que y
          (version déterministe, non randomisée : légèrement conservatrice)

Conditionnement :
    marginal          : un seuil unique pour toutes les classes
    class_conditional : un seuil par classe vraie ; si une classe a moins de `min_samples`
                        exemples de calibration, on utilise le seuil global (repli, documenté
                        dans protocol.yaml).

Ensemble de prédiction : C(x) = { k : s(x, k) <= q_k }. Abstention (protocol.yaml) :
taille de l'ensemble différente de 1, ensemble vide compris.

Garantie (échangeabilité calibration/test) : 1 - alpha <= P(y in C(x)) <= 1 - alpha + 1/(n+1)
pour le conditionnement marginal. Elle ne dépend pas de la qualité du modèle (seulement
de la taille des ensembles). Sous décalage de distribution, l'échangeabilité est rompue
et la garantie ne tient plus : c'est l'objet de l'étude.

"""
from dataclasses import dataclass, field

import numpy as np

SCORES = ("lac", "aps")
CONDITIONINGS = ("marginal", "class_conditional")
SOURCES = ("id", "ood_few_shot", "mixed")


# ----------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------
def score_matrix(probs, score="lac"):
    """Score de chaque classe candidate pour chaque exemple : matrice [N, K]."""
    probs = np.asarray(probs, dtype=np.float64)
    if score == "lac":
        return 1.0 - probs
    if score == "aps":
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)                 # inclut la classe elle-même
        out = np.empty_like(cum)
        np.put_along_axis(out, order, cum, axis=1)
        return out
    raise ValueError(f"score inconnu : {score} (attendu : {SCORES})")


def true_class_scores(probs, y, score="lac"):
    return score_matrix(probs, score)[np.arange(len(y)), y]


# ----------------------------------------------------------------------
# Quantile conforme
# ----------------------------------------------------------------------
def conformal_quantile(scores, alpha):
    """Quantile d'ordre ceil((n+1)(1-alpha))/n des scores. +inf si n est trop petit
    (l'ensemble contient alors toutes les classes) ou si n = 0."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(s[k - 1])


def coverage_bounds(n, alpha):
    """Bornes théoriques de la couverture marginale pour n exemples de calibration."""
    return 1.0 - alpha, min(1.0, 1.0 - alpha + 1.0 / (n + 1))


# ----------------------------------------------------------------------
# Calibration et prédiction
# ----------------------------------------------------------------------
@dataclass
class Calibrated:
    score: str
    conditioning: str
    alpha: float
    qhat: np.ndarray                 # [K] seuil par classe (identique pour toutes si marginal)
    q_global: float
    n_cal: int
    n_per_class: np.ndarray          # [K]
    fallback: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # [K] True = seuil global

    def summary(self):
        return {"score": self.score, "conditioning": self.conditioning, "alpha": self.alpha,
                "q_global": self.q_global, "qhat": self.qhat.tolist(), "n_cal": self.n_cal,
                "n_per_class": self.n_per_class.tolist(), "fallback": self.fallback.tolist()}


def calibrate(probs, y, alpha, score="lac", conditioning="marginal", min_samples=20):
    """Calcule les seuils conformes sur un jeu de calibration (probs [N, K], y [N])."""
    if score not in SCORES:
        raise ValueError(f"score inconnu : {score}")
    if conditioning not in CONDITIONINGS:
        raise ValueError(f"conditionnement inconnu : {conditioning}")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha doit etre dans ]0, 1[")
    probs = np.asarray(probs)
    y = np.asarray(y).astype(np.int64)
    K = probs.shape[1]
    s = true_class_scores(probs, y, score)
    q_global = conformal_quantile(s, alpha)
    counts = np.bincount(y, minlength=K)

    qhat = np.full(K, q_global)
    fallback = np.ones(K, dtype=bool)
    if conditioning == "class_conditional":
        for c in range(K):
            if counts[c] >= min_samples:
                qhat[c] = conformal_quantile(s[y == c], alpha)
                fallback[c] = False
    else:
        fallback[:] = False          # le conditionnement marginal n'utilise pas de repli
    return Calibrated(score, conditioning, float(alpha), qhat, q_global, len(y), counts, fallback)


def predict_sets(probs, cal):
    """Ensembles de prédiction booléens [N, K] pour les exemples de test."""
    S = score_matrix(probs, cal.score)
    return S <= cal.qhat[None, :]


def conformal_sets(probs_cal, y_cal, probs_test, alpha, score="lac",
                   conditioning="marginal", min_samples=20):
    """Raccourci : calibre puis prédit. Retourne (sets, Calibrated)."""
    cal = calibrate(probs_cal, y_cal, alpha, score, conditioning, min_samples)
    return predict_sets(probs_test, cal), cal


# ----------------------------------------------------------------------
# Sources de calibration (protocol.yaml : id / ood_few_shot / mixed)
# ----------------------------------------------------------------------
def build_calibration_set(source, id_probs, id_y, ood_probs=None, ood_y=None,
                          n_ood=None, seed=0):
    """Assemble le jeu de calibration.

    id           : calib des centres sources uniquement
    ood_few_shot : n_ood exemples tirés au hasard dans ood_calib_pool (tout si n_ood=None)
    mixed        : calib (ID) + n_ood exemples du pool OOD, mélange simple sans pondération
    Le tirage est reproductible via `seed`. test_ood n'est jamais utilisé ici.
    """
    if source not in SOURCES:
        raise ValueError(f"source inconnue : {source} (attendu : {SOURCES})")
    if source == "id":
        return np.asarray(id_probs), np.asarray(id_y)
    if ood_probs is None or ood_y is None:
        raise ValueError("ood_probs et ood_y sont requis pour ood_few_shot et mixed")
    ood_probs, ood_y = np.asarray(ood_probs), np.asarray(ood_y)
    n = len(ood_y) if n_ood is None else int(n_ood)
    if not 0 < n <= len(ood_y):
        raise ValueError(f"n_ood doit etre dans [1, {len(ood_y)}], recu {n}")
    idx = np.random.default_rng(seed).choice(len(ood_y), size=n, replace=False)
    if source == "ood_few_shot":
        return ood_probs[idx], ood_y[idx]
    return (np.concatenate([np.asarray(id_probs), ood_probs[idx]]),
            np.concatenate([np.asarray(id_y), ood_y[idx]]))