"""Nonnegative component coefficients plus an unconstrained smooth background.

Polynomial coordinates are spectrum index, not an assumed uniform Raman axis.
Eliminating the background by orthogonal projection gives an exact NNLS problem.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import f1_score


class BackgroundNNLS:
    def __init__(self, reference, degree=-1):
        self.a = np.asarray(reference, dtype=np.float64).T
        if self.a.ndim != 2 or not np.isfinite(self.a).all() or degree not in (-1, 0, 1, 2, 3):
            raise ValueError("Finite (classes,width) reference and degree -1..3 required")
        self.degree = degree
        if degree >= 0:
            u = np.linspace(-1., 1., self.a.shape[0])
            basis = np.polynomial.legendre.legvander(u, degree)
            self.background, _ = np.linalg.qr(basis, mode="reduced")
        else:
            self.background = np.zeros((self.a.shape[0], 0))
        projected = self.a - self.background @ (self.background.T @ self.a)
        self.q, self.r = np.linalg.qr(projected, mode="reduced")
        if np.linalg.matrix_rank(self.r) != self.a.shape[1]:
            raise ValueError("Projected references rank deficient; background absorbs identifiable components")

    def predict(self, spectra):
        x = np.asarray(spectra, dtype=np.float64).reshape(len(spectra), -1)
        if x.shape[1] != self.a.shape[0] or not len(x) or not np.isfinite(x).all():
            raise ValueError("Nonempty finite spectra with matching width required")
        # q is orthogonal to background; no large projection matrix is formed.
        projected = x - (x @ self.background) @ self.background.T
        coefficients = np.stack([nnls(self.r, v)[0] for v in projected @ self.q])
        background_coeff = (x - coefficients @ self.a.T) @ self.background
        return coefficients, background_coeff

    def reconstruct(self, coefficients, background_coeff):
        return coefficients @ self.a.T + background_coeff @ self.background.T


def labels_top2(scores):
    labels = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(labels, np.argsort(scores, axis=1)[:, -2:], True, axis=1)
    return labels


def evaluate_labels(truth, labels):
    truth, labels = np.asarray(truth, bool), np.asarray(labels, bool)
    if truth.shape != labels.shape or truth.ndim != 2 or not len(truth):
        raise ValueError("Aligned, nonempty label matrices required")
    true_count, count = truth.sum(1), labels.sum(1)
    return dict(samples=len(truth), exact_set_accuracy=float(np.all(truth == labels,1).mean()),
                count_accuracy=float((count == true_count).mean()),
                macro_f1=float(f1_score(truth, labels, average="macro", zero_division=0)),
                undercount=int((count < true_count).sum()), overcount=int((count > true_count).sum()))


def validation_threshold(truth, scores):
    scores = np.asarray(scores, np.float64)
    truth = np.asarray(truth, bool)
    if scores.shape != truth.shape or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid validation data")
    values = np.unique(scores)
    thresholds = np.r_[np.nextafter(values[0], -np.inf),
                       values[:-1]+(values[1:]-values[:-1])/2, np.nextafter(values[-1], np.inf)]
    correct = np.array([np.all((scores >= t) == truth, axis=1).sum() for t in thresholds])
    selected = float(thresholds[np.flatnonzero(correct == correct.max())[-1]])
    return dict(threshold=selected, validation=evaluate_labels(truth, scores >= selected))


def select_background(reference, validation_x, truth, degrees=(-1, 0, 1, 2, 3)):
    """Top-2 validation accuracy; ties prefer the less complex degree."""
    if not degrees or len(set(degrees)) != len(degrees):
        raise ValueError("Nonempty unique candidate degrees required")
    candidates, predictions = [], {}
    for degree in sorted(degrees):
        model = BackgroundNNLS(reference, degree)
        scores, _ = model.predict(validation_x)
        correct = int(np.all(labels_top2(scores) == truth, axis=1).sum())
        candidates.append(dict(degree=degree, correct=correct, samples=len(truth)))
        predictions[degree] = scores
    selected = max(candidates, key=lambda row: (row["correct"], -row["degree"]))
    return selected["degree"], candidates, predictions
