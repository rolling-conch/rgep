"""Nonnegative learned dictionaries, train-reference-only component naming."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from reference_background_nnls import BackgroundNNLS, labels_top2, evaluate_labels


def named_library(components, references):
    h, refs = np.asarray(components, np.float64), np.asarray(references, np.float64)
    if h.shape != refs.shape or h.ndim != 2 or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError('Seven finite nonnegative basis spectra matching training references required')
    scale = np.linalg.norm(h, axis=1)
    if np.any(scale < 1e-12):
        raise ValueError('Degenerate zero NMF component')
    h = h/scale[:,None]
    refs = refs/np.maximum(np.linalg.norm(refs,axis=1,keepdims=True),1e-12)
    cosine = h@refs.T
    component, category = linear_sum_assignment(-cosine)
    named = np.empty_like(h)
    named[category] = h[component]
    return named, dict(component_indices=component.tolist(), class_indices=category.tolist(),
                       cosine_matrix=cosine.tolist(), matched_cosines=cosine[component,category].tolist(),
                       basis_row_norm_before_scaling=scale.tolist(),
                       caveat='One-to-one assignment using training references only; not guaranteed chemical identifiability')


def dictionary_predictions(library, x):
    # Frozen Frobenius NMF inference is an NNLS subproblem. Use the same exact
    # solver as the reference-library baseline rather than different tolerances.
    solver = BackgroundNNLS(library)
    coefficients, bg = solver.predict(x)
    reconstruction = solver.reconstruct(coefficients,bg)
    error = float(np.mean((np.asarray(x).reshape(len(x),-1)-reconstruction)**2))
    return coefficients, error


def metric_rules(truth, scores, threshold):
    return dict(top2=evaluate_labels(truth,labels_top2(scores)),
                threshold=evaluate_labels(truth,scores>=threshold))


def candidate_key(truth, scores, reconstruction_mse):
    if not np.isfinite(scores).all() or not np.isfinite(reconstruction_mse):
        raise ValueError('Nonfinite validation output')
    return int(np.all(labels_top2(scores)==truth,axis=1).sum()), -float(reconstruction_mse)
