"""Frozen cardinality stress-test data; optical weights, not mass fractions."""
from itertools import combinations
import numpy as np
from rssnet_data import normalize_input


def simulate_cardinality(pure, per_combination, seed, components):
    """Held-out 1/3-component extension; never changes the legacy binary RNG.

    Uniform constrained simplex: w=.1+(1-.1*K)*Dirichlet(1,...,1).
    K=2 therefore has exactly the same marginal U(.1,.9) weight law.
    Source rows are sampled only from the caller's partition. Optical weights
    are not mass fractions. All classes/combinations have equal sample counts.
    """
    if components not in (1, 3) or per_combination < 1:
        raise ValueError("Positive per_combination and components=1 or 3 required")
    if len(pure) < components:
        raise ValueError("Insufficient source classes")
    rng = np.random.default_rng(seed)
    records = {k: [] for k in ("x", "truth", "classes", "weights", "source_rows", "snr_db", "background_a")}
    for classes in combinations(range(len(pure)), components):
        ids = np.stack([rng.integers(len(pure[k]), size=per_combination) for k in classes], 1)
        weights = (np.ones((per_combination, 1)) if components == 1 else
                   .1 + (1 - .1 * components) * rng.dirichlet(np.ones(components), per_combination))
        weights = weights.astype(np.float32)
        mixture = sum(pure[k][ids[:, j]] * weights[:, j, None] for j, k in enumerate(classes))
        white = rng.normal(size=mixture.shape)
        brown = np.cumsum(rng.normal(size=mixture.shape), axis=1)
        brown -= brown.mean(axis=1, keepdims=True)
        brown /= np.maximum(brown.std(axis=1, keepdims=True), 1e-8)
        nuisance = white + brown
        nuisance /= np.maximum(np.linalg.norm(nuisance, axis=1, keepdims=True), 1e-8)
        snr = rng.uniform(10, 30, (per_combination, 1))
        noisy = mixture + nuisance * np.linalg.norm(mixture, axis=1, keepdims=True) * 10 ** (-snr / 20)
        amplitude = rng.uniform(0, .08, (per_combination, 1))
        u = np.linspace(0, 1, mixture.shape[1])[None, :]
        noisy += amplitude * noisy.max(axis=1, keepdims=True) * (u - .5) ** 2
        x, _ = normalize_input(np.maximum(noisy, 0))
        truth = np.zeros((per_combination, len(pure)), bool)
        truth[:, list(classes)] = True
        values = dict(x=x, truth=truth, classes=np.tile(classes, (per_combination, 1)),
                      weights=weights, source_rows=ids, snr_db=snr[:, 0], background_a=amplitude[:, 0])
        for key, value in values.items():
            records[key].append(value)
    return {key: np.concatenate(value) for key, value in records.items()}
