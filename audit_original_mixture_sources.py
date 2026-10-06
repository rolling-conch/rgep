"""Conservatively map stored endpoints to pure spectra, then select train-only P."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from fixed_splits import file_sha256
from decoupled_data import DEFAULT_CATEGORIES
from rssnet_data import normalize_input, pure_partitions, write_json


def candidates(endpoints, pure, tolerance=1e-6):
    a = np.array(endpoints, dtype=np.float64, copy=True)
    b = np.array(pure, dtype=np.float64, copy=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    result, errors = [], []
    for start in range(0, len(a), 64):
        distance = np.max(np.abs(a[start:start+64, None] - b[None]), axis=-1)
        result.extend([np.flatnonzero(row <= tolerance).tolist() for row in distance])
        errors.extend(distance.min(axis=1).tolist())
    return result, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='dataset/dataset_train_mix_33600_20251116.mat')
    p.add_argument('--manifest', default='dataset/splits/rssnet_pure_19580920.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    pure_partitions(args.source, args.manifest, 19580920)  # validates hashes and isolation
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    roles = {(r['category'], r['row']): r['partition'] for r in manifest['rows']}
    source = loadmat(args.source, simplify_cells=True)
    pool = source['Data_pure_norm_Struct']
    details, selected_x, selected_pairs, selected_rows = [], [], [], []
    report = dict(source_sha256=file_sha256(args.source), manifest_sha256=file_sha256(args.manifest),
                  tolerance_unit_max_abs=1e-6, per_pair={}, selected_per_pair=512,
                  rule='all tolerance-matching candidates of BOTH endpoints must belong to train; ambiguous cross-role excluded',
                  limitation='Numerical provenance, not physical sample/batch independence; matched-size P, not full original recipe')
    rng = np.random.default_rng(19580920)
    for a, b in combinations(range(7), 2):
        ca, cb = DEFAULT_CATEGORIES[a], DEFAULT_CATEGORIES[b]
        key = ca + cb
        if key not in source:
            key = cb + ca
        x = np.asarray(source[key])
        ea, eb = source[key+'_'+ca], source[key+'_'+cb]
        ma, da = candidates(ea, pool[ca])
        mb, db = candidates(eb, pool[cb])
        allowed = []
        for i, (ia, ib) in enumerate(zip(ma, mb)):
            eligible = bool(ia and ib) and all(roles[ca,j]=='train' for j in ia) and all(roles[cb,j]=='train' for j in ib)
            if eligible:
                allowed.append(i)
            details.append(dict(pair=key, row=i, source_a=ia, source_b=ib,
                                max_error_a=da[i], max_error_b=db[i], train_eligible=eligible))
        report['per_pair'][key] = dict(total=len(x), unmapped=sum(not ia or not ib for ia,ib in zip(ma,mb)),
            ambiguous=sum(len(ia)>1 or len(ib)>1 for ia,ib in zip(ma,mb)), eligible=len(allowed),
            max_nearest_error=max(max(da),max(db)),
            sum_relative_error=float(np.linalg.norm(x-ea-eb)/np.linalg.norm(x)))
        if len(allowed) >= 512:
            ids = rng.choice(allowed, 512, replace=False)
            selected_x.append(normalize_input(x[ids])[0])
            selected_pairs.extend([(a,b)]*512)
            selected_rows.extend([dict(pair=key,row=int(i)) for i in ids])
    write_json(out/'mapping.json', details)
    report['all_rows_mapped'] = all(v['unmapped']==0 for v in report['per_pair'].values())
    report['ready_for_P'] = report['all_rows_mapped'] and len(selected_x)==21
    if report['ready_for_P']:
        np.savez_compressed(out/'P_train.npz', x=np.concatenate(selected_x), pair=np.asarray(selected_pairs,dtype=np.int64))
        report['P_train_sha256'] = file_sha256(out/'P_train.npz')
        write_json(out/'P_selected_rows.json', selected_rows)
    write_json(out/'report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not report['ready_for_P']:
        raise ValueError('Mapping/coverage insufficient: do not launch P')


if __name__=='__main__':
    main()
