"""Validation-selected global thresholds without Top-2 or count correction.

Uses frozen aligned RGEP validation/real prediction caches. Run in the original
Linux container to reproduce synthetic input array fingerprints exactly.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
ALIGN = 'run/ReferenceCheckpointAlignment_20260924_seed19580920'
DEFAULT_RUN = ALIGN + '/joint/training/ReferenceExtractor_20260923_170259_526394'
DEFAULT_BASE = 'run/RSSNet_OriginalP_20260921_seed19580920/Matched_rssnet_20260921_065731_269111'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        x = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(str(x.dtype).encode())
        h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()

def labels_from_pairs(pairs, classes=7):
    y = np.zeros((len(pairs), classes), np.int32)
    np.put_along_axis(y, pairs, 1, axis=1)
    return y

def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, bool), np.asarray(prediction, bool)
    if truth.shape != prediction.shape or truth.ndim != 2 or not len(truth):
        raise ValueError('Nonempty, aligned multilabel arrays required')
    true_count, predicted_count = truth.sum(1), prediction.sum(1)
    exact = np.all(truth == prediction, axis=1)
    correct_count = true_count == predicted_count
    return dict(samples=len(truth), exact_set_correct=int(exact.sum()),
        exact_set_accuracy=float(exact.mean()), count_correct=int(correct_count.sum()),
        count_accuracy=float(correct_count.mean()),
        macro_f1=float(f1_score(truth, prediction, average='macro', zero_division=0)),
        predicted_count_histogram=np.bincount(predicted_count, minlength=truth.shape[1]+1).tolist(),
        true_count_histogram=np.bincount(true_count, minlength=truth.shape[1]+1).tolist(),
        undercount=int((predicted_count < true_count).sum()),
        overcount=int((predicted_count > true_count).sum()),
        correct_count_wrong_classes=int((correct_count & ~exact).sum()))

def select_threshold(truth, scores):
    """Evaluate every distinct decision region; exact-set max, tie higher t.

    Float64 comparisons preserve midpoints between float32 cached probabilities.
    Extreme candidates also permit predicting all classes or no classes.
    """
    scores = np.asarray(scores, np.float64)
    truth = np.asarray(truth, bool)
    if scores.shape != truth.shape or not np.isfinite(scores).all() or not len(scores):
        raise ValueError('Invalid validation scores')
    values = np.unique(scores)
    midpoints = values[:-1] + (values[1:] - values[:-1]) / 2
    candidates = np.r_[np.nextafter(values[0], -np.inf), midpoints,
                       np.nextafter(values[-1], np.inf)]
    correct = np.asarray([np.all((scores >= t) == truth, axis=1).sum() for t in candidates])
    best_index = int(np.flatnonzero(correct == correct.max())[-1])
    threshold = float(candidates[best_index])
    return dict(threshold=threshold, validation=metrics(truth, scores >= threshold),
                candidate_count=len(candidates), tied_best_regions=int((correct == correct.max()).sum())), candidates, correct

class LibraryNNLS:
    def __init__(self, refs):
        self.a = np.asarray(refs, np.float64).T
        if self.a.ndim != 2 or not np.isfinite(self.a).all():
            raise ValueError('Invalid library')
        self.q, self.r = np.linalg.qr(self.a, mode='reduced')
        if np.linalg.matrix_rank(self.r) != self.a.shape[1]:
            raise ValueError('Rank deficient library')

    def predict(self, inputs):
        x = np.asarray(inputs, np.float64).reshape(len(inputs), -1)
        if x.shape[1] != self.a.shape[0] or not np.isfinite(x).all():
            raise ValueError('Invalid spectra')
        result = np.stack([nnls(self.r, target)[0] for target in x @ self.q])
        for i in np.linspace(0, len(x)-1, min(10, len(x)), dtype=int):
            np.testing.assert_allclose(result[i], nnls(self.a, x[i])[0], atol=1e-8, rtol=1e-7)
        return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=DEFAULT_RUN)
    parser.add_argument('--real-dir', default=ALIGN+'/joint/real_conditions_1_9')
    parser.add_argument('--baseline', default=DEFAULT_BASE)
    parser.add_argument('--output', default='run/UnknownCount_ThresholdAligned_20260924_seed19580920')
    args = parser.parse_args()
    run, real, base, out = [ROOT/Path(v) for v in (args.run_dir,args.real_dir,args.baseline,args.output)]
    if out.exists():
        raise FileExistsError('Refusing to overwrite results: '+str(out))
    cfg, state = read(run/'config.json'), read(run/'status.json')
    if state['state'] != 'completed' or read(real/'status.json')['state'] != 'completed':
        raise ValueError('Training and real inference must be completed')
    weight_hash = sha(run/'checkpoint/best.weights.h5')
    if weight_hash != state['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    with np.load(run/'reference.npz') as f:
        refs, names = f['reference'], f['names']
    with np.load(base/'reference.npz') as f:
        np.testing.assert_array_equal(refs, f['reference'])
        np.testing.assert_array_equal(names, f['names'])
    from rssnet_data import pure_partitions, simulate, normalize_input
    pure, regenerated_refs, _ = pure_partitions(ROOT/cfg['args']['source'], ROOT/cfg['args']['manifest'])
    np.testing.assert_array_equal(refs, regenerated_refs)
    validation = simulate(pure['validation'], 64, 19581020)
    if array_digest(validation) != cfg['validation_array_sha256']:
        raise ValueError('Validation fingerprint differs. Use the original Linux environment; do not bypass this check.')
    truth = labels_from_pairs(validation['pair'])
    val_path = run/'result/selected_validation_predictions.npz'
    with np.load(val_path) as f:
        np.testing.assert_array_equal(truth, f['truth'])
        rg_val = f['probabilities'].astype(np.float64)
        selection = read(run/'result/checkpoint_selection.json')
        if metrics(truth, f['labels'])['exact_set_correct'] != selection['correct_samples']:
            raise ValueError('Cached selected validation predictions do not match selection record')
        np.testing.assert_allclose(metrics(truth, rg_val >= .5)['exact_set_accuracy'], selection['fixed05_subset_accuracy'])
    solver = LibraryNNLS(refs)
    start = time.perf_counter()
    nnls_val = solver.predict(validation['x'])
    # Selection happens here, before reading any real test labels/predictions.
    selections, curves = {}, {}
    for method, scores in [('NNLS', nnls_val), ('RGEP', rg_val)]:
        selected, candidates, correct = select_threshold(truth, scores)
        selections[method] = selected
        curves[method+'_thresholds'] = candidates
        curves[method+'_validation_exact_correct'] = correct
        print(method, selected, flush=True)
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(run=str(run.relative_to(ROOT)), selected_epoch=state['best_epoch'],
        decision='One global scalar threshold per method; score >= threshold; no Top-2, no count correction, no forced nonempty output.',
        selection='Maximum full synthetic validation exact-set accuracy over all distinct score decision regions. Midpoint thresholds; ties choose higher threshold. All/none prediction regions included.',
        nnls_score='Raw nonnegative coefficients with unit-L2 mixture and unit-L2 training median references; no coefficient sum normalization, no sum-to-one constraint, not mass fractions.',
        rgep_score='Cached presence scores from the checkpoint-aligned joint model.',
        evaluation='Same 1344 synthetic validation mixtures; thresholds locked before evaluating 1344 synthetic test and 56700 reconstructed real spectra.',
        limitation='All validation and real truth sets contain two classes. No hard count prior at prediction; not validation of variable-cardinality or open-set mixtures.',
        selections=selections, validation_array_sha256=array_digest(validation), model_sha256=weight_hash,
        reference_sha256=sha(run/'reference.npz'), validation_predictions_sha256=sha(val_path),
        script_sha256=sha(__file__), threshold_selection_seconds=time.perf_counter()-start)
    write(out/'protocol.json', protocol)
    write(out/'status.json',dict(state='evaluating',thresholds_locked=True))
    np.savez_compressed(out/'validation_scores.npz', truth=truth, alpha=validation['alpha'],
                        NNLS=nnls_val, RGEP=rg_val, **curves)
    provenance = dict(inputs={}, names=names.tolist())
    synthetic_path = run/'result/synthetic_predictions.npz'
    with np.load(synthetic_path) as f:
        synthetic_truth = labels_from_pairs(f['pair'])
        if array_digest({k:f[k] for k in ('x','y','pair','alpha')}) != cfg['test_array_sha256']:
            raise ValueError('Synthetic test cache fingerprint differs')
        syn_scores = dict(NNLS=solver.predict(f['x']), RGEP=f['probabilities'].astype(np.float64))
        synthetic = {k: metrics(synthetic_truth, v >= selections[k]['threshold']) for k,v in syn_scores.items()}
        np.savez_compressed(out/'synthetic_test_predictions.npz', truth=synthetic_truth, alpha=f['alpha'],
                            **syn_scores, **{k+'_labels':v >= selections[k]['threshold'] for k,v in syn_scores.items()})
    provenance['inputs'][str(synthetic_path.relative_to(ROOT))] = sha(synthetic_path)
    rows, all_truth, all_pred = [], [], {'NNLS':[], 'RGEP':[]}
    for condition in range(1,10):
        rp, bp = real/f'condition{condition}/predictions.npz', base/f'result/condition{condition}/predictions.npz'
        prov = read(real/f'condition{condition}/provenance.json')
        if prov['model_sha256'] != weight_hash or prov['reference_sha256'] != protocol['reference_sha256'] or prov['source_predictions_sha256'] != sha(bp):
            raise ValueError('Real inference provenance mismatch')
        with np.load(rp) as rf, np.load(bp) as bf:
            np.testing.assert_array_equal(rf['sample_index'], bf['sample_index'])
            np.testing.assert_array_equal(rf['pairs'], bf['pairs'])
            np.testing.assert_array_equal(rf['names'], names)
            y = rf['truth']
            pair_indices = np.asarray([[names.tolist().index(p[:2]),names.tolist().index(p[2:])] for p in bf['pairs']])
            np.testing.assert_array_equal(y, labels_from_pairs(pair_indices))
            x = normalize_input(bf['input'])[0]
            scores = dict(NNLS=solver.predict(x), RGEP=rf['probabilities'].astype(np.float64))
            labels = {k: v >= selections[k]['threshold'] for k,v in scores.items()}
            row = dict(condition=condition, models={k:metrics(y,v) for k,v in labels.items()})
            rows.append(row); all_truth.append(y)
            for k in labels: all_pred[k].append(labels[k])
            np.savez_compressed(out/f'condition{condition}_predictions.npz', truth=y,
                sample_index=rf['sample_index'], pairs=rf['pairs'], **scores,
                **{k+'_labels':v for k,v in labels.items()})
        for path in (rp,bp): provenance['inputs'][str(path.relative_to(ROOT))] = sha(path)
        print('Real condition',condition,row['models'],flush=True)
    y = np.concatenate(all_truth)
    pooled = {k:metrics(y,np.concatenate(v)) for k,v in all_pred.items()}
    results = dict(protocol=protocol, validation={k:v['validation'] for k,v in selections.items()},
                   synthetic_test=synthetic, real_conditions=rows, real_pooled=pooled)
    write(out/'results.json',results); write(out/'provenance.json',provenance)
    lines=['# 取消Top-2：NNLS与RGEP验证集阈值对照','',
        f'RGEP使用统一选模实验的第{state["best_epoch"]}轮联合检查点。两种方法参考库、验证集、实测输入一致。', '',
        '完整标签集合准确率要求所有类别完全正确；数量准确率只要求预测类别数等于真实数，类别可能全错。Macro-F1按七类计算后平均。', '',
        '## 阈值选择','', '以完整合成验证集标签集合准确率最高选择一个共享阈值，枚举分数改变判定的全部区间，采用区间中点，平局取较高阈值。NNLS使用未作和归一化的非负拟合系数，RGEP使用存在性分数。阈值数值不具有相同物理含义。','',
        '| 方法 | 选定阈值 | 验证集完整集合 % | 数量正确 % | Macro-F1 |','|---|---:|---:|---:|---:|']
    for k,v in selections.items():
        m=v['validation']; lines.append(f'| {k} | {v["threshold"]:.10g} | {100*m["exact_set_accuracy"]:.2f} | {100*m["count_accuracy"]:.2f} | {m["macro_f1"]:.4f} |')
    lines+=['','## 实测结果（阈值锁定，不作数量修正）','',
        '| 条件 | NNLS完整集合 % | RGEP完整集合 % | NNLS数量正确 % | RGEP数量正确 % | NNLS Macro-F1 | RGEP Macro-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows+[dict(condition='九条件合并',models=pooled)]:
        a,b=[row['models'][k] for k in ('NNLS','RGEP')]
        lines.append(f'| {row["condition"]} | {100*a["exact_set_accuracy"]:.2f} | {100*b["exact_set_accuracy"]:.2f} | {100*a["count_accuracy"]:.2f} | {100*b["count_accuracy"]:.2f} | {a["macro_f1"]:.4f} | {b["macro_f1"]:.4f} |')
    lines+=['','## 合成测试集（同样阈值）','', '| 方法 | N | 完整集合 % | 数量正确 % | Macro-F1 |','|---|---:|---:|---:|---:|']
    for k,m in synthetic.items(): lines.append(f'| {k} | {m["samples"]} | {100*m["exact_set_accuracy"]:.2f} | {100*m["count_accuracy"]:.2f} | {m["macro_f1"]:.4f} |')
    lines+=['','## 解释边界','',
        '- 推理不指定组分数量，不强制至少一类，也不对预测数量作二元修正。',
        '- 验证和实测真值仍全部是两类；数量准确率在这里等于输出两类的比例，不能证明对一、三或更多组分泛化。',
        '- 同一合成验证集也用于RGEP检查点选择；阈值验证结果是调参结果，应以独立合成测试和实测结果评价。',
        '- 全部候选仍限制在七类训练参考库内，不涉及未知材料发现；系数不是质量比，分数不等同于已校准概率。',
        '- 本次只检验已完成的一个联合模型检查点，旧99.12%模型的阈值不可混用。',
        '- 结果为单种子、反复使用的回顾性实测基准；不根据本次实测结果改阈值、重新选择检查点或缩小条件范围。','',
        '数据与可复现记录见protocol.json、results.json、provenance.json及各条件NPZ。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'status.json',dict(state='completed',conditions=list(range(1,10)),samples=len(y),thresholds_locked=True))
    print('POOLED',json.dumps(pooled,ensure_ascii=False),flush=True)

if __name__ == '__main__': main()
