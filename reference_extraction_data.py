"""Seven-query labels and inference; test labels never select model queries."""
import numpy as np

def query_targets(data, ids, queries):
    pairs = data['pair'][ids]
    matches = pairs == queries[:, None]
    spectrum = np.sum(data['y'][ids] * matches[:, None, :], axis=-1, keepdims=True)
    return dict(spectrum=spectrum.astype('float32'), presence=matches.any(axis=1)[:, None].astype('float32'))

def training_references(pure, targets, queries, rng):
    refs = []
    for y, k in zip(targets[..., 0], queries):
        pool = pure[int(k)]
        if np.linalg.norm(y) > 1e-8:
            # Exclude identical/near-identical target source, including scaled duplicates.
            valid = np.max(np.abs(pool - y / np.linalg.norm(y)), axis=1) > 1e-6
            pool = pool[valid]
        if not len(pool):
            raise ValueError('No independent training reference available for target class')
        refs.append(pool[rng.integers(len(pool))])
    return np.asarray(refs, dtype='float32')[..., None]

def query_all(model, x, references, batch_size=16):
    spectra, presence = [], []
    for reference in references:
        ys, ps = [], []
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size]
            rb = np.broadcast_to(reference[None, :, None], xb.shape).copy()
            out = model(dict(mixture=xb, reference=rb), training=False)
            ys.append(np.asarray(out['spectrum'])[..., 0]); ps.append(np.asarray(out['presence'])[:, 0])
        spectra.append(np.concatenate(ys)); presence.append(np.concatenate(ps))
    spectra, presence = np.stack(spectra, axis=-1), np.stack(presence, axis=-1)
    if not np.isfinite(spectra).all() or not np.isfinite(presence).all():
        raise FloatingPointError('Non-finite extraction output')
    return spectra, presence

def score_presence(truth, probabilities):
    from sklearn.metrics import f1_score
    top2 = np.zeros_like(truth)
    np.put_along_axis(top2, np.argsort(probabilities, axis=1)[:, -2:], 1, axis=1)
    threshold = probabilities >= .5
    return dict(samples=len(truth), top2_pair_accuracy=float(np.all(top2 == truth, axis=1).mean()),
        fixed05_subset_accuracy=float(np.all(threshold == truth, axis=1).mean()),
        fixed05_macro_f1=float(f1_score(truth, threshold, average='macro', zero_division=0))), top2


def health_metrics(data, spectra, probabilities):
    from sklearn.metrics import roc_auc_score
    truth = np.zeros((len(spectra), 7), dtype='int32')
    np.put_along_axis(truth, data['pair'], 1, axis=1)
    target = np.zeros_like(spectra)
    for i in range(2):
        target[np.arange(len(target)), :, data['pair'][:, i]] = data['y'][..., i]
    metrics, _ = score_presence(truth, probabilities)
    mask = truth.astype(bool)
    energy = np.sum(spectra**2, axis=1)
    error = np.sum((spectra-target)**2, axis=1)
    true_energy = np.sum(target**2, axis=1)
    metrics.update(presence_auc=float(roc_auc_score(truth.ravel(), probabilities.ravel())),
        positive_presence_mean=float(probabilities[mask].mean()), negative_presence_mean=float(probabilities[~mask].mean()),
        positive_nonzero_fraction=float((energy[mask] > 1e-8).mean()),
        positive_relative_mse=float(np.mean(error[mask]/np.maximum(true_energy[mask],1e-8))),
        absent_query_mean_energy=float(energy[~mask].mean()), positive_query_mean_energy=float(energy[mask].mean()))
    metrics['health_passed'] = bool(metrics['positive_nonzero_fraction'] >= .99 and
        metrics['presence_auc'] >= .75 and metrics['positive_presence_mean']-metrics['negative_presence_mean'] >= .1 and
        metrics['top2_pair_accuracy'] >= .25 and metrics['positive_relative_mse'] < .9 and
        metrics['absent_query_mean_energy'] < metrics['positive_query_mean_energy'])
    return metrics
