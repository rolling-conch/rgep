"""Load audited P samples with stored endpoints; no new mixture synthesis."""
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from fixed_splits import file_sha256
from rssnet_data import normalize_input
from decoupled_data import DEFAULT_CATEGORIES


def load_original(source, manifest, audit_dir):
    root = Path(audit_dir)
    report = json.loads((root / 'report.json').read_text())
    if not report['ready_for_P']:
        raise ValueError('Source audit failed')
    for path, key in [(source, 'source_sha256'), (manifest, 'manifest_sha256'),
                      (root / 'P_train.npz', 'P_train_sha256')]:
        if file_sha256(path) != report[key]:
            raise ValueError('Audit fingerprint mismatch: ' + str(path))
    selected = json.loads((root / 'P_selected_rows.json').read_text())
    mapping = {(r['pair'], r['row']): r for r in json.loads((root / 'mapping.json').read_text())}
    if len(selected) != 10752 or len({(r['pair'], r['row']) for r in selected}) != 10752:
        raise ValueError('Unexpected or repeated P rows')
    d = loadmat(source, simplify_cells=True)
    xs, ys, pairs = [], [], []
    for r in selected:
        k, i = r['pair'], r['row']
        if not mapping[k, i]['train_eligible']:
            raise ValueError('Non-training source')
        a, b = k[:2], k[2:]
        xs.append(d[k][i])
        ys.append(np.stack([d[k+'_'+a][i], d[k+'_'+b][i]], axis=-1))
        pairs.append([DEFAULT_CATEGORIES.index(a), DEFAULT_CATEGORIES.index(b)])
    x, scale = normalize_input(np.asarray(xs))
    y = np.asarray(ys, dtype=np.float32) / scale[..., None]
    with np.load(root / 'P_train.npz') as p:
        np.testing.assert_array_equal(x, p['x'])
        np.testing.assert_array_equal(pairs, p['pair'])
    np.testing.assert_allclose(y.sum(axis=-1), x[..., 0], atol=1e-6)
    return dict(x=x, y=y, pair=np.asarray(pairs, dtype=np.int64))
