"""Train-only batch/streaming NMF versus frozen Raman identification baselines."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import json
import shutil
import time
import warnings
from pathlib import Path
import numpy as np
import sklearn
from sklearn.decomposition import NMF, MiniBatchNMF
from threadpoolctl import threadpool_limits
from nmf_baseline import named_library,dictionary_predictions,metric_rules,candidate_key
from reference_background_nnls import validation_threshold
from train_csam_rgep_aligned import prepare,assert_epoch,truth_labels,read
from fixed_splits import file_sha256
from matched_separators import array_digest
from rssnet_data import write_json
from evaluate_reference_raw_nnls import load_raw


def assess(h, refs, validation, truth):
    library,mapping=named_library(h,refs)
    scores,mse=dictionary_predictions(library,validation['x'])
    key=candidate_key(truth,scores,mse)
    return library,mapping,scores,mse,key


def report(run, results, selection, csam):
    lines=['# NMF对照：固定训练字典、独立样本预测','',
           'NMF学习时不使用实测谱；潜在组分类别只由训练参考库命名。系数按单位L2字典计算，不是概率或质量分数。',
           '批量NMF使用第1轮10,752条谱；增量NMF使用全部50轮同一合成数据流。优化步数不等于神经网络更新次数。',
           '标准NMF的7个基向量不保证各自对应一种材料；一对一参考匹配是额外的已知类别库假设。','',
           '| 方法 | 原始谱Top-2/% | 去噪谱Top-2/% | 原始谱阈值/% | 去噪谱阈值/% |',
           '|---|---:|---:|---:|---:|']
    def row(name,r,d):
        lines.append(f"| {name} | {100*r['top2']['exact_set_accuracy']:.2f} | {100*d['top2']['exact_set_accuracy']:.2f} | "
                     f"{100*r['threshold']['exact_set_accuracy']:.2f} | {100*d['threshold']['exact_set_accuracy']:.2f} |")
    for method in selection:
        row(method,results['raw']['pooled'][method],results['paper-denoised']['pooled'][method])
    for method in ('NNLS','NNLS_background','RGEP'):
        r=results['raw']['baselines']; d=results['paper-denoised']['baselines']
        row(method,{k:r[method+'_'+k] for k in ('top2','threshold')},
                   {k:d[method+'_'+k] for k in ('top2','threshold')})
    row('CSAM_aligned',csam['raw']['pooled'],csam['paper-denoised']['pooled'])
    for source in ('raw','paper-denoised'):
        lines += ['', '## '+source+'逐条件', '',
                  '| Dataset | 批量NMF Top-2/% | 增量NMF Top-2/% | 批量NMF阈值/% | 增量NMF阈值/% |',
                  '|---|---:|---:|---:|---:|']
        for result in results[source]['conditions']:
            m=result['metrics']
            values=[m[k][rule]['exact_set_accuracy']*100 for rule in ('top2','threshold') for k in selection]
            lines.append('| '+str(result['condition'])+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines += ['', '## 验证集锁定参数', '']
    for method,s in selection.items():
        lines.append(f"- {method}：所选候选 {s['selected_id']}；阈值 {s['threshold']['threshold']:.8g}；"
                     f"验证Top-2正确 {s['validation_correct']}/1344；最小匹配余弦 {min(s['mapping']['matched_cosines']):.4f}。")
    lines += ['', '## 范围', '',
              '- 主表是类别集合识别，不是实测端元或丰度真值误差。',
              '- 验证集用于选初始化/轮次及阈值，合成测试和实测预测只在这些选择冻结后计算。',
              '- Top-2使用恰好两类先验；阈值不强制数量，但全部实测真值仍是双组分。',
              '- 只有一个训练数据种子，批量NMF三种初始化不是三个独立数据种子。',
              '- 不根据实测结果重新初始化或调参；重构MSE仅表示拟合输入，不代表真实组分恢复准确。']
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--raw-evaluation',default='run/ReferenceRawNNLS_online_seed19580920_20260925')
    p.add_argument('--denoised-evaluation',default='run/ReferenceDenoisedNNLS_online_seed19580920_20260925')
    p.add_argument('--csam-run',default='run/CSAM_RGEP_OnlineAligned_seed19580920_20260925')
    p.add_argument('--smoke',action='store_true',help='Tiny training/validation only; never read real/test predictions')
    a=p.parse_args(); run=Path(a.output); run.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); write_json(run/'status.json',dict(state='preparing',smoke=a.smoke))
    try:
        cfg,state,generator,refs,held,audit=prepare(Path(a.anchor))
        seed=cfg['args']['seed']; epochs=2 if a.smoke else cfg['args']['epochs']
        train=generator.generate(0); assert_epoch(train,0,audit,seed)
        validation=held['validation']; y=truth_labels(validation)
        if a.smoke:
            ids=np.arange(0,len(train['x']),512)
            train={k:v[ids] for k,v in train.items()}
            ids=np.arange(0,len(y),64)
            validation={k:v[ids] for k,v in validation.items()};y=y[ids]
        x=np.ascontiguousarray(train['x'][...,0])
        if np.any(x<0): raise ValueError('NMF input must be nonnegative')
        protocol=dict(anchor=a.anchor,smoke=a.smoke,training_seed=seed,n_components=7,
            batch_candidates=[dict(init='nndsvda',seed=seed),dict(init='random',seed=seed+1),dict(init='random',seed=seed+2)],
            batch_max_iter=15 if a.smoke else 1000,batch_tol=1e-4,
            online=dict(init='nndsvda',epochs=epochs,batch_size=512,forget_factor=1.0,seed=seed),
            selection='maximum full synthetic validation Top-2; tie lower input-reconstruction MSE; exact tie earliest candidate',
            inference='unit-L2 learned spectra; fixed-library exact NNLS coefficients; NO dictionary adaptation at test time',
            naming='Hungarian one-to-one cosine assignment against training median references; no mixture labels used to fit H',
            threshold='maximize same synthetic validation exact-set; tie higher threshold; no test tuning',
            source_manifest_sha256=cfg['source_manifest_sha256'],reference_sha256=file_sha256(Path(a.anchor)/'reference.npz'),
            validation_sha256=cfg['validation_array_sha256'],test_sha256=cfg['test_array_sha256'],
            anchor_checkpoint_sha256=state['checkpoint_sha256'],sklearn_version=sklearn.__version__,
            script_sha256=file_sha256(__file__),helper_sha256=file_sha256('nmf_baseline.py'))
        write_json(run/'protocol.json',protocol)
        for file in [__file__,'nmf_baseline.py','reference_online_data.py','reference_background_nnls.py','rssnet_data.py']:
            shutil.copy2(file,run/Path(file).name)
        best_key=None; selected={}; libraries={}; histories={'NMF_batch':[],'NMF_online':[]}
        for index,candidate in enumerate(protocol['batch_candidates']):
            write_json(run/'status.json',dict(state='fitting_batch',candidate=index+1,smoke=a.smoke))
            tick=time.perf_counter()
            model=NMF(n_components=7,init=candidate['init'],random_state=candidate['seed'],solver='cd',
                      beta_loss='frobenius',tol=1e-4,max_iter=protocol['batch_max_iter'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit_transform(x)
            library,mapping,scores,mse,key=assess(model.components_,refs,validation,y)
            row=dict(candidate=index+1,**candidate,iterations=model.n_iter_,seconds=time.perf_counter()-tick,
                     validation_correct=key[0],validation_reconstruction_mse=mse,warnings=[str(w.message) for w in caught])
            histories['NMF_batch'].append(row)
            np.savez_compressed(run/f'batch_candidate{index+1}.npz',components=model.components_,library=library)
            if best_key is None or key>best_key:
                best_key=key; libraries['NMF_batch']=library
                selected['NMF_batch']=dict(selected_id=index+1,validation_correct=key[0],mapping=mapping,
                                           threshold=validation_threshold(y,scores),fit_record=row)
            write_json(run/'fit_history.json',histories);print('BATCH',row,flush=True)
        online=MiniBatchNMF(n_components=7,init='nndsvda',random_state=seed,batch_size=512,
                            beta_loss='frobenius',forget_factor=1.0,tol=1e-4)
        best_key=None; steps=0
        for epoch in range(epochs):
            tick=time.perf_counter();data=generator.generate(epoch);assert_epoch(data,epoch,audit,seed)
            data_hash=array_digest(data)
            xx=np.ascontiguousarray(data['x'][...,0])
            if a.smoke: xx=xx[::512]
            order=np.random.default_rng(seed+epoch).permutation(len(xx))
            write_json(run/'status.json',dict(state='fitting_online',epoch=epoch+1,epochs=epochs,smoke=a.smoke))
            for pos in range(0,len(xx),512):
                online.partial_fit(xx[order[pos:pos+512]]);steps+=1
            library,mapping,scores,mse,key=assess(online.components_,refs,validation,y)
            row=dict(epoch=epoch+1,training_array_sha256=data_hash,mixtures=len(xx),steps=steps,
                     seconds=time.perf_counter()-tick,validation_correct=key[0],validation_reconstruction_mse=mse)
            histories['NMF_online'].append(row)
            if best_key is None or key>best_key:
                best_key=key;libraries['NMF_online']=library.copy()
                selected['NMF_online']=dict(selected_id=epoch+1,validation_correct=key[0],mapping=mapping,fit_record=row)
                np.savez_compressed(run/'online_best.npz',components=online.components_,library=library)
            write_json(run/'fit_history.json',histories);print('ONLINE',row,flush=True)
        for method,library in libraries.items():
            scores,_=dictionary_predictions(library,validation['x'])
            selected[method]['threshold']=validation_threshold(y,scores)
            np.savez_compressed(run/f'{method}_frozen.npz',library=library,validation_truth=y,validation_scores=scores)
        write_json(run/'selection.json',selected)
        write_json(run/'budget.json',dict(fit_and_selection_seconds=time.perf_counter()-start,
            batch_training_mixtures=len(x),online_training_mixtures=sum(r['mixtures'] for r in histories['NMF_online']),
            online_steps=steps,online_epochs=epochs,threads=4,smoke=a.smoke,
            caveat='NMF iteration/partial_fit steps are not equivalent to neural optimizer updates; fit includes validation'))
        if a.smoke:
            write_json(run/'status.json',dict(state='smoke_completed'));return
        # No held-out predictions before all choices above are frozen.
        synthetic={};test_y=truth_labels(held['test'])
        for method,library in libraries.items():
            scores,mse=dictionary_predictions(library,held['test']['x'])
            synthetic[method]=dict(metrics=metric_rules(test_y,scores,selected[method]['threshold']['threshold']),input_reconstruction_mse=mse)
            np.savez_compressed(run/f'{method}_synthetic.npz',truth=test_y,scores=scores,alpha=held['test']['alpha'])
        write_json(run/'synthetic_metrics.json',synthetic)
        results={}
        for source,path in [('raw',a.raw_evaluation),('paper-denoised',a.denoised_evaluation)]:
            previous=Path(path);prior=read(previous/'results.json')
            if (read(previous/'status.json')['state']!='completed' or prior['protocol']['model_sha256']!=state['checkpoint_sha256']
                    or prior['protocol']['reference_sha256']!=protocol['reference_sha256']):
                raise ValueError('Comparison anchor mismatch')
            dest=run/source;dest.mkdir();ys=[];all_scores={m:[] for m in libraries};rows=[]
            for i in range(1,10):
                write_json(run/'status.json',dict(state='evaluating_real',source=source,condition=i))
                file=(Path('dataset/dataset_mixture_raw')/f'dataset{i}_norm.mat' if source=='raw'
                      else Path('dataset/dataset_mixture_denoised')/f'dataset{i}_denoised.mat')
                xx,yy,pairs=load_raw(file,suffix='' if source=='raw' else '_norm')
                r=prior['conditions'][i-1]
                if file_sha256(file)!=r['input_sha256'] or array_digest(dict(x=xx))!=r['normalized_input_sha256']:
                    raise ValueError('Real input changed')
                with np.load(previous/f'condition{i}_predictions.npz') as f:
                    np.testing.assert_array_equal(yy,f['truth']);np.testing.assert_array_equal(pairs,f['pairs'])
                    np.testing.assert_array_equal(np.arange(len(yy)),f['sample_index'])
                scores_dict={};metrics={};reconstruction={}
                for method,library in libraries.items():
                    scores,mse=dictionary_predictions(library,xx);scores_dict[method]=scores
                    metrics[method]=metric_rules(yy,scores,selected[method]['threshold']['threshold'])
                    reconstruction[method]=mse;all_scores[method].append(scores)
                np.savez_compressed(dest/f'condition{i}_predictions.npz',truth=yy,pairs=pairs,sample_index=np.arange(len(yy)),**scores_dict)
                rows.append(dict(condition=i,metrics=metrics,input_reconstruction_mse=reconstruction))
                ys.append(yy);write_json(dest/'partial_results.json',rows);print('REAL',source,i,metrics,flush=True)
            truth=np.concatenate(ys)
            results[source]=dict(conditions=rows,pooled={m:metric_rules(truth,np.concatenate(s),selected[m]['threshold']['threshold'])
                                  for m,s in all_scores.items()},baselines=prior['pooled'])
            write_json(dest/'results.json',results[source])
        csam_path=Path(a.csam_run)
        if read(csam_path/'status.json')['state']!='completed':raise ValueError('CSAM comparison incomplete')
        csam=read(csam_path/'result/real_comparison.json')
        report(run,results,selected,csam)
        write_json(run/'status.json',dict(state='completed',samples_per_source=56700,total_seconds=time.perf_counter()-start))
        print('COMPLETED:',run,flush=True)
    except Exception as exc:
        write_json(run/'status.json',dict(state='failed',error=repr(exc)));raise


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
