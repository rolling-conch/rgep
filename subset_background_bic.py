"""Exhaustive nonnegative support selection with smooth background and BIC.

All 2**K supports, including zero and singleton, are eligible. Boundary NNLS
solutions are represented by their smaller active supports, so enumerating
feasible interior least-squares fits is sufficient for the global selection.
"""
from itertools import combinations
import numpy as np


class SubsetBackgroundBIC:
    def __init__(self, reference, degree=3):
        a = np.asarray(reference, dtype=np.float64).T
        if a.ndim != 2 or not np.isfinite(a).all() or degree not in range(-1, 4):
            raise ValueError("Finite references and degree -1..3 required")
        self.width, self.classes = a.shape
        if not 1 <= self.classes <= 10 or self.width <= self.classes + degree + 1:
            raise ValueError("Unsupported dimensions")
        self.degree = degree
        self.a = a
        b = np.polynomial.legendre.legvander(np.linspace(-1, 1, self.width), degree) \
            if degree >= 0 else np.empty((self.width, 0))
        self.background = np.linalg.qr(b, mode="reduced")[0]
        projected = a - self.background @ (self.background.T @ a)
        self.q, self.r = np.linalg.qr(projected, mode="reduced")
        if np.linalg.matrix_rank(self.r) != self.classes:
            raise ValueError("Background-projected reference library is rank deficient")
        self.supports = [s for k in range(self.classes + 1)
                         for s in combinations(range(self.classes), k)]
        self.labels = np.zeros((len(self.supports), self.classes), dtype=bool)
        self.inverses = []
        for i, s in enumerate(self.supports):
            self.labels[i, list(s)] = True
            self.inverses.append(np.linalg.pinv(self.r[:, s]) if s else None)
        self.dimensions = self.labels.sum(1) + self.background.shape[1]

    def residuals(self, spectra):
        x = np.asarray(spectra, dtype=np.float64).reshape(len(spectra), -1)
        if x.shape[1] != self.width or not len(x) or not np.isfinite(x).all():
            raise ValueError("Invalid spectra")
        norm2 = np.sum(x*x, axis=1)
        if np.any(norm2 <= 0):
            raise ValueError("Zero spectrum")
        y = x @ self.q
        bg = x @ self.background
        orthogonal = np.maximum(norm2 - np.sum(bg*bg, 1) - np.sum(y*y, 1), 0)
        rss = np.full((len(x), len(self.supports)), np.inf)
        rss[:, 0] = orthogonal + np.sum(y*y, 1)
        tolerance = 1e-10 * np.sqrt(norm2)
        for index, support in enumerate(self.supports[1:], 1):
            c = y @ self.inverses[index].T
            valid = np.all(c > tolerance[:, None], axis=1)
            residual = y - c @ self.r[:, support].T
            rss[valid, index] = orthogonal[valid] + np.sum(residual[valid]**2, axis=1)
        return rss, norm2

    def select(self, rss, norm2, penalty=1.0):
        if not np.isfinite(penalty) or penalty <= 0:
            raise ValueError("Positive finite penalty required")
        # Unknown Gaussian variance contributes a constant common to all supports.
        safe = np.maximum(rss, np.asarray(norm2)[:, None] * 1e-14)
        criterion = self.width * np.log(safe / self.width)
        criterion += penalty * self.dimensions[None, :] * np.log(self.width)
        # Supports sorted by cardinality then class index: deterministic ties.
        choice = np.argmin(criterion, axis=1)
        return dict(labels=self.labels[choice], support_index=choice,
                    rss=rss[np.arange(len(rss)), choice],
                    criterion=criterion[np.arange(len(rss)), choice])

    def predict(self, spectra, penalty=1.0):
        rss, norm2 = self.residuals(spectra)
        result = self.select(rss, norm2, penalty)
        x = np.asarray(spectra, dtype=np.float64).reshape(len(spectra), -1)
        y = x @ self.q
        coefficients = np.zeros((len(x), self.classes))
        for index in np.unique(result["support_index"]):
            support = self.supports[index]
            if not support:
                continue
            rows = np.flatnonzero(result["support_index"] == index)
            coefficients[np.ix_(rows, support)] = y[rows] @ self.inverses[index].T
        result["coefficients"] = coefficients
        return result
