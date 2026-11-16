"""Prueba adaptativa acotada: train/cal/select/test separados; CPU batch1 real."""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.adaptive import observe_prefix,choose_actions,action_cost_ms
from src.advanced import SmallCNN,load_checkpoint,predict,preprocess
from src.classification_data import load_classification_split
from src.conventional import load_model
from experiments.exp02_conventional import _json,_csv,_sha,classification_metrics
from experiments.exp03_advanced import provenance
from experiments.exp09_channel_diagnostic import atomic_npz

POLICIES=('always_svm256','always_svm512','always_cnn','confidence','flatness','selector')


def calibration_metrics(prob,correct):
    prob=np.clip(prob,1e-8,1-1e-8)
    ece=0.
    for lower,upper in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(prob>=lower)&(prob<upper if upper<1 else prob<=upper)
        if mask.any():
            ece+=mask.mean()*abs(prob[mask].mean()-correct[mask].mean())
    return dict(brier=float(np.mean((prob-correct)**2)),nll=float(-np.mean(correct*np.log(prob)+(1-correct)*np.log(1-prob))),ece10=float(ece))


def bootstrap_difference(y,snr,correct_a,correct_b,time_a,time_b,weight,seed=2026110615):
    rng=np.random.default_rng(seed); reps=10000
    values=np.column_stack((correct_a.astype(float)-correct_b,time_a-time_b))
    sums=np.zeros((reps,2))
    for label in np.unique(y):
        for level in np.unique(snr):
            cell=values[(y==label)&(snr==level)]
            counts=rng.multinomial(len(cell),np.full(len(cell),1/len(cell)),size=reps)
            sums+=counts@cell
    means=sums/len(y)
    return dict(accuracy_delta=float(values[:,0].mean()),accuracy_ci95=np.percentile(means[:,0],[2.5,97.5]).tolist(),
        total_ms_delta=float(values[:,1].mean()),total_ms_ci95=np.percentile(means[:,1],[2.5,97.5]).tolist(),
        utility_delta=float(np.mean(values[:,0]-weight*values[:,1])),
        utility_ci95=np.percentile(means[:,0]-weight*means[:,1],[2.5,97.5]).tolist(),replicates=reps,seed=seed)


def main():
    out=Path('results/adaptive'); root=Path('data/adaptive_dataset')
    config=json.loads(Path('configs/adaptive_experiment.json').read_text())
    reserve=json.loads((out/'reservation.json').read_text())
    assert config==reserve['config'] and provenance(root)==reserve['dataset_hashes']
    assert all(_sha(p)==h for p,h in reserve['model_hashes'].items())
    torch.set_num_threads(1)
    svm=[load_model(f'results/fase5/conventional_{n}.joblib')['pipeline'] for n in (256,512)]
    cnn=SmallCNN().eval(); cnn.load_state_dict(load_checkpoint(config['cnn'])['model'])
    weight=config['lambda_per_ms']

    def state(message):
        (out/'ESTADO_REANUDACION.md').write_text('# Estado adaptativo\n\n'+message+
            '\n\nReanudar: `.\\.venv\\Scripts\\python.exe -m experiments.exp10_adaptive`. Reutiliza cachés y selector congelado. No reentrena bases.\n',encoding='utf-8')

    def cache(split):
        path=out/f'{split}_counterfactuals.npz'
        d=load_classification_split(root,split,512,allow_shift=True)
        if path.exists():
            with np.load(path) as f:
                return d,{k:f[k].copy() for k in f.files}
        observation=observe_prefix(d['iq'][:,:256],svm[0])
        p=np.column_stack((observation['initial'],svm[1].predict(d['iq']),
                           predict(cnn,torch.from_numpy(preprocess(d['iq'][:,:256])),'cpu').argmax(1)))
        values=dict(**observation,predictions=p,truth=d['targets'])
        atomic_npz(path,**values)
        state('Caché offline guardada: '+split+'. Objetivos contrafactuales solo para ajuste/evaluación; nunca entrada del selector.')
        return d,values

    def cnn_one(iq):
        with torch.inference_mode():
            return int(cnn(torch.from_numpy(preprocess(iq[None,:256]))).argmax(1).item())

    def base_call(action,iq):
        return int(svm[action].predict(iq[None,:256 if action==0 else 512])[0]) if action<2 else cnn_one(iq)

    def route(kind,iq,artifact):
        if kind in POLICIES[:3]:
            action=POLICIES.index(kind)
            return action,base_call(action,iq)
        observation=observe_prefix(iq[None,:256],svm[0])
        action=int(choose_actions(observation,kind,artifact)[0])
        prediction=int(observation['initial'][0]) if action==0 else base_call(action,iq)
        return action,prediction

    with threadpool_limits(limits=1):
        if not (out/'policy.joblib').exists():
            train,ct=cache('train'); val,cv=cache('val')
            seen=defaultdict(int); cal=[]; select=[]
            for i,row in enumerate(val['evaluation_rows']):
                key=(row['modulation'],row['snr_nominal_db'])
                (cal if seen[key]<25 else select).append(i); seen[key]+=1
            cal=np.array(cal); select=np.array(select)
            assert len(cal)==len(select)==1200 and not set(cal)&set(select)
            _json(out/'development_partitions.json',dict(train_parents=[r['parent_waveform_id'] for r in train['evaluation_rows']],
                cal_parents=[val['evaluation_rows'][i]['parent_waveform_id'] for i in cal],
                selection_parents=[val['evaluation_rows'][i]['parent_waveform_id'] for i in select]))
            # Costos previos al test: medición de bases en val_cal, batch1.
            latencies=[]
            for action in range(3):
                for i in cal[:8]: base_call(action,val['iq'][i])
                for repeat in range(3):
                    for i in cal[:96]:
                        start=time.perf_counter_ns(); base_call(action,val['iq'][i])
                        latencies.append(dict(action=action,repeat=repeat,row=int(i),ms=(time.perf_counter_ns()-start)/1e6))
            base_ms=[float(np.mean([r['ms'] for r in latencies if r['action']==a])) for a in range(3)]
            _csv(out/'base_latency_development.csv',latencies)
            calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,random_state=config['selector_seed']))
            calibrator.fit(cv['margin'][cal],(cv['initial'][cal]==cv['truth'][cal]).astype(int))
            targets=(ct['predictions']==ct['truth'][:,None]).astype(float)
            acquisition,compute=action_cost_ms(np.arange(3),base_ms)
            targets-=weight*(acquisition+compute)[None,:]
            selector=RandomForestRegressor(n_estimators=config['rf_trees'],max_depth=config['rf_max_depth'],
                min_samples_leaf=config['rf_min_samples_leaf'],random_state=config['selector_seed'],n_jobs=1)
            selector.fit(ct['gate'],targets)
            artifact=dict(calibrator=calibrator,selector=selector,confidence_threshold=0.,flatness_threshold=0.,base_ms=base_ms)
            obs={k:v[select] for k,v in cv.items() if k in ('gate','initial','margin','flatness')}
            thresholds=config['confidence_thresholds']
            flat=list(np.quantile(ct['flatness'],config['flatness_quantiles']))
            flat=[float(min(flat)-1),*map(float,flat),float(max(flat)+1)]
            candidates=[]
            for kind,grid in [('confidence',thresholds),('flatness',flat)]:
                best=-np.inf; best_threshold=None
                for threshold in grid:
                    artifact[kind+'_threshold']=threshold
                    actions=choose_actions(obs,kind,artifact)
                    pred=cv['predictions'][select,actions]
                    ac,cp=action_cost_ms(actions,base_ms)
                    utility=float(np.mean(pred==cv['truth'][select])-weight*np.mean(ac+cp))
                    candidates.append(dict(kind=kind,threshold=threshold,utility_without_common_gate=utility))
                    if utility>best: best,best_threshold=utility,threshold
                artifact[kind+'_threshold']=best_threshold
            _csv(out/'threshold_selection.csv',candidates)
            # Medición completa de rutas en selección, antes de abrir test.
            val_rows=[]
            for kind in POLICIES:
                for i in select[:8]: route(kind,val['iq'][i],artifact)
                times=[]
                for i in select[:96]:
                    start=time.perf_counter_ns(); action,p=route(kind,val['iq'][i],artifact)
                    times.append((time.perf_counter_ns()-start)/1e6)
                actions=np.full(len(select),POLICIES.index(kind)) if kind in POLICIES[:3] else choose_actions(obs,kind,artifact)
                pred=cv['predictions'][select,actions]
                acquisition=np.where(actions==1,64.,32.)
                val_rows.append(dict(policy=kind,accuracy=float(np.mean(pred==cv['truth'][select])),
                    compute_ms=float(np.mean(times)),acquisition_ms=float(acquisition.mean()),
                    utility=float(np.mean(pred==cv['truth'][select])-weight*(np.mean(times)+acquisition.mean()))))
            artifact['reference_fixed']=max(val_rows[:3],key=lambda r:r['utility'])['policy']
            artifact['selected_policy']=max(val_rows,key=lambda r:r['utility'])['policy']
            artifact['config']=config
            temp=out/'policy.tmp'; joblib.dump(artifact,temp); os.replace(temp,out/'policy.joblib')
            _json(out/'validation_summary.json',val_rows)
            _json(out/'selection_frozen.json',dict(policy_sha256=_sha(out/'policy.joblib'),reference_fixed=artifact['reference_fixed'],
                selected_policy=artifact['selected_policy'],confidence_threshold=artifact['confidence_threshold'],
                flatness_threshold=artifact['flatness_threshold'],base_ms=base_ms,lambda_per_ms=weight,
                test_opened=False,decision_inputs='65_features_plus6_svm_scores_from256_only'))
            state('Selector/calibrador/umbrales congelados. Próximo paso test nuevo, sin ajustes posteriores.')
        artifact=joblib.load(out/'policy.joblib')
        frozen=json.loads((out/'selection_frozen.json').read_text())
        assert _sha(out/'policy.joblib')==frozen['policy_sha256']
        metrics={}; allrows=[]; comparisons={}; calibration={}
        for split in ('test','test_shift'):
            d,c=cache(split)
            snr=np.array([float(r['snr_nominal_db']) for r in d['evaluation_rows']])
            observation={k:c[k] for k in ('gate','initial','margin','flatness')}
            calibration[split]=calibration_metrics(artifact['calibrator'].predict_proba(c['margin'])[:,1],(c['initial']==c['truth']).astype(float))
            timing_path=out/f'{split}_routes.npz'
            if timing_path.exists():
                with np.load(timing_path) as f: measured={k:f[k].copy() for k in f.files}
            else:
                actions=np.zeros((len(d['iq']),len(POLICIES)),dtype=int); predictions=actions.copy(); times=np.zeros_like(actions,dtype=float)
                for kind in POLICIES:
                    for iq in d['iq'][:8]: route(kind,iq,artifact)
                rng=np.random.default_rng(config['selector_seed'])
                for i,iq in enumerate(d['iq']):
                    for j in rng.permutation(len(POLICIES)):
                        start=time.perf_counter_ns(); action,pred=route(POLICIES[j],iq,artifact)
                        times[i,j]=(time.perf_counter_ns()-start)/1e6
                        actions[i,j]=action; predictions[i,j]=pred
                    if (i+1)%600==0: print(f'{split}: rutas {i+1}/{len(d["iq"])}',flush=True)
                measured=dict(actions=actions,predictions=predictions,compute_ms=times)
                atomic_npz(timing_path,**measured)
                state('Rutas completas guardadas: '+split+'. Se conservarán al reanudar.')
            metrics[split]={}
            for j,kind in enumerate(POLICIES):
                actions=measured['actions'][:,j]; pred=measured['predictions'][:,j]
                expected=np.full(len(actions),j) if j<3 else choose_actions(observation,kind,artifact)
                assert np.array_equal(actions,expected)
                assert np.array_equal(pred,c['predictions'][np.arange(len(actions)),actions])
                cp=measured['compute_ms'][:,j]; ac=np.where(actions==1,64.,32.)
                metrics[split][kind]=dict(**classification_metrics(c['truth'],pred),compute_mean_ms=float(cp.mean()),
                    compute_median_ms=float(np.median(cp)),compute_p95_ms=float(np.percentile(cp,95)),acquisition_mean_ms=float(ac.mean()),
                    total_mean_ms=float(np.mean(cp+ac)),utility=float(np.mean(pred==c['truth'])-weight*np.mean(cp+ac)),
                    actions={str(a):int((actions==a).sum()) for a in range(3)})
                for i,row in enumerate(d['evaluation_rows']):
                    allrows.append(dict(split=split,policy=kind,parent=row['parent_waveform_id'],target=int(c['truth'][i]),prediction=int(pred[i]),
                        action=int(actions[i]),snr_db=snr[i],compute_ms=float(cp[i]),acquisition_ms=float(ac[i]),total_ms=float(cp[i]+ac[i])))
            ref=POLICIES.index(artifact['reference_fixed'])
            comparisons[split]={}
            for name in ('confidence','flatness','selector'):
                j=POLICIES.index(name)
                comparisons[split][name+'_minus_'+artifact['reference_fixed']]=bootstrap_difference(c['truth'],snr,
                    measured['predictions'][:,j]==c['truth'],measured['predictions'][:,ref]==c['truth'],
                    measured['compute_ms'][:,j]+np.where(measured['actions'][:,j]==1,64.,32.),
                    measured['compute_ms'][:,ref]+np.where(measured['actions'][:,ref]==1,64.,32.),weight)
        _json(out/'metrics.json',metrics); _json(out/'paired_comparisons.json',comparisons); _json(out/'calibration.json',calibration)
        _csv(out/'predictions_routes.csv',allrows)
        groups=[]
        for split in metrics:
            for kind in POLICIES:
                for snr in config_from_data(root)['snr_db']:
                    rows=[r for r in allrows if r['split']==split and r['policy']==kind and r['snr_db']==snr]
                    groups.append(dict(split=split,policy=kind,snr_db=snr,n=len(rows),accuracy=float(np.mean([r['target']==r['prediction'] for r in rows])),
                        total_ms=float(np.mean([r['total_ms'] for r in rows])),a0=sum(r['action']==0 for r in rows),a1=sum(r['action']==1 for r in rows),a2=sum(r['action']==2 for r in rows)))
        _csv(out/'by_snr.csv',groups)
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,split in zip(axes,metrics):
            for kind,m in metrics[split].items():
                ax.scatter(m['total_mean_ms'],1-m['accuracy']); ax.annotate(kind,(m['total_mean_ms'],1-m['accuracy']),fontsize=8)
            ax.set(xlabel='Adquisición virtual + CPU medido (ms)',ylabel='Error',title=split); ax.grid(alpha=.3)
        fig.savefig(out/'risk_cost.png',dpi=150); plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        for ax,split in zip(axes,metrics):
            bottom=np.zeros(len(POLICIES))
            for a in range(3):
                values=np.array([metrics[split][k]['actions'][str(a)]/2400 for k in POLICIES])
                ax.bar(POLICIES,values,bottom=bottom,label=f'a{a}'); bottom+=values
            ax.tick_params(axis='x',rotation=30); ax.set(ylabel='Fracción de acciones',title=split); ax.legend()
        fig.savefig(out/'action_distribution.png',dpi=150); plt.close(fig)
        assert all(_sha(p)==h for p,h in reserve['model_hashes'].items())
        _json(out/'completed.json',dict(completed=True,base_models_unchanged=True,test_consumed=True,policy_sha256=_sha(out/'policy.joblib')))
        state('Experimento completado. Bases/selector congelados, test y shift consumidos; métricas/IC/figuras guardados. No ajustar por test.')
        print(json.dumps(metrics,indent=2),flush=True)


def config_from_data(root):
    return json.loads((root/'config.json').read_text())


if __name__=='__main__':
    main()
