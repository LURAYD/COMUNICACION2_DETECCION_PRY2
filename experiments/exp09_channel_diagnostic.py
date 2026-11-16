"""Diagnóstico exploratorio reanudable por escenario; modelos congelados."""
import argparse
import itertools
import json
import os
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.advanced import SmallCNN,load_checkpoint,predict,preprocess
from src.conventional import CLASSES,load_model
from experiments.channel_scenarios import scenario_parent
from experiments.exp02_conventional import _sha,_json,_csv,classification_metrics
from experiments.exp06_final_holdout import paired_bootstrap


def atomic_npz(path,**arrays):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez(f,**arrays); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/channel_diagnostic')
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('configs/channel_diagnostic.json').read_text())
    base=json.loads(Path(config['base_config']).read_text())
    assert config['common_channel_length']==14
    paths=dict(cnn_original='results/fase6/best.pt',cnn_phase='results/fase6_phase_aug/best.pt',
               svm256='results/fase5/conventional_256.joblib',svm512='results/fase5/conventional_512.joblib')
    frozen=dict(config=config,base_config=base,model_hashes={k:_sha(v) for k,v in paths.items()},
                paired_noise='same_standardized_noise_scaled_per_signal_full_frame_power',
                crop='rrc_length_minus_one_plus_13_plus_common_timing',scope='exploratory_not_final_test')
    protocol=out/'protocol_frozen.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())==frozen
    else:
        _json(protocol,frozen)
    models={}
    torch.set_num_threads(1)
    for name,path in paths.items():
        if name.startswith('cnn'):
            model=SmallCNN().cuda().eval(); model.load_state_dict(load_checkpoint(path)['model'])
        else:
            model=load_model(path)['pipeline']
        models[name]=model
    plan=[(name,snr) for name in CLASSES for snr in config['snr_db'] for _ in range(config['parents_per_class_snr'])]
    truth=np.array([CLASSES.index(n) for n,s in plan]); snrs=np.array([s for n,s in plan])
    saved={}
    for scenario in config['scenarios']:
        path=out/f'{scenario}_predictions.npz'
        if path.exists():
            with np.load(path) as d:
                saved[scenario]={k:d[k].copy() for k in d.files}
            assert np.array_equal(saved[scenario]['truth'],truth)
            continue
        iq=[]; metadata=[]
        for i,(name,snr) in enumerate(plan):
            x,info,_=scenario_parent(base,config['seed'],i,name,snr,scenario)
            iq.append(x); metadata.append(info)
        iq=np.array(iq)
        # Persistencia observable separada de verdad de diagnóstico.
        atomic_npz(out/f'{scenario}_observations.npz',iq256=iq[:,:256],iq512=iq)
        _json(out/f'{scenario}_oracle.json',metadata)
        predictions={}
        with threadpool_limits(limits=1):
            for name,model in models.items():
                if name.startswith('cnn'):
                    predictions[name]=predict(model,torch.from_numpy(preprocess(iq[:,:256])),'cuda').argmax(1)
                else:
                    predictions[name]=model.predict(iq[:,:int(name[3:])])
        saved[scenario]=dict(truth=truth,snr=snrs,**predictions)
        atomic_npz(path,**saved[scenario])
        (out/'ESTADO_REANUDACION.md').write_text('# Diagnóstico en curso\n\nEscenarios completados: '+', '.join(saved)+
            '\n\nReanudar: `.\\.venv\\Scripts\\python.exe -m experiments.exp09_channel_diagnostic` . Se reutilizan predicciones atómicas por escenario; no se entrena.\n',encoding='utf-8')
        print('Escenario guardado: '+scenario,flush=True)
    metric_rows=[]; complements=[]; oracle=[]; effects=[]
    names=list(models)
    for scenario,d in saved.items():
        for level in [None,*config['snr_db']]:
            mask=np.ones(len(truth),dtype=bool) if level is None else snrs==level
            correct={name:d[name][mask]==truth[mask] for name in names}
            for name in names:
                m=classification_metrics(truth[mask],d[name][mask])
                metric_rows.append(dict(scenario=scenario,snr_db='all' if level is None else level,model=name,
                                        **{k:v for k,v in m.items() if k!='per_class'}))
            for a,b in itertools.combinations(names,2):
                ca,cb=correct[a],correct[b]
                complements.append(dict(scenario=scenario,snr_db='all' if level is None else level,a=a,b=b,n=int(mask.sum()),
                    both_correct=int((ca&cb).sum()),only_a=int((ca&~cb).sum()),only_b=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),
                    oracle_pair_accuracy=float((ca|cb).mean())))
            oracle.append(dict(scenario=scenario,snr_db='all' if level is None else level,n=int(mask.sum()),
                 oracle_three_original=float((correct['svm256']|correct['svm512']|correct['cnn_original']).mean()),
                 oracle_three_phase=float((correct['svm256']|correct['svm512']|correct['cnn_phase']).mean()),
                 oracle_four=float(np.logical_or.reduce(list(correct.values())).mean()),
                 best_fixed_accuracy=max(float(c.mean()) for c in correct.values())))
        reference='beta_035' if scenario.startswith('beta_') else 'id'
        for name in names:
            result=paired_bootstrap(truth,snrs,d[name],saved[reference][name],seed=2026110612,replicates=10000)
            effects.append(dict(scenario=scenario,reference=reference,model=name,**result))
    _csv(out/'metrics.csv',metric_rows); _csv(out/'complementarity.csv',complements); _csv(out/'oracle_upper_bounds.csv',oracle)
    _json(out/'paired_scenario_effects.json',effects)
    interaction={name:float(np.mean((saved['both_shift'][name]==truth).astype(float)-(saved['cfo_shift'][name]==truth)
                        -(saved['multipath_shift'][name]==truth)+(saved['id'][name]==truth))) for name in names}
    _json(out/'factorial_interaction_accuracy.json',interaction)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    groups=[['id','cfo_zero','cfo_shift','multipath_shift','both_shift'],['beta_025','beta_035','beta_050','beta_065']]
    for ax,group in zip(axes,groups):
        for name in names:
            vals=[next(r['accuracy'] for r in metric_rows if r['scenario']==s and r['snr_db']=='all' and r['model']==name) for s in group]
            ax.plot(range(len(group)),vals,'o-',label=name)
        ax.set(xticks=range(len(group)),xticklabels=group,ylabel='Exactitud agregada',ylim=(0,1))
        ax.tick_params(axis='x',rotation=25); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.savefig(out/'controlled_effects.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    selected=[r for r in oracle if r['snr_db']=='all']
    for key,label in [('best_fixed_accuracy','Mejor fijo (descriptivo)'),('oracle_three_original','Oracle a0/a1/CNN original'),('oracle_four','Oracle cuatro')]:
        ax.plot([r['scenario'] for r in selected],[r[key] for r in selected],'o-',label=label)
    ax.set(ylabel='Exactitud',ylim=(0,1),title='Límites oracle: usan etiquetas, no son políticas implementables')
    ax.tick_params(axis='x',rotation=30); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(out/'oracle_complementarity.png',dpi=150); plt.close(fig)
    assert frozen['model_hashes']=={k:_sha(v) for k,v in paths.items()}
    _json(out/'completed.json',dict(completed=True,parents=len(plan),scenarios=len(saved),models_unchanged=True,
                                  observation_hashes={p.name:_sha(p) for p in out.glob('*observations.npz')}))
    print(json.dumps([r for r in metric_rows if r['snr_db']=='all'],indent=2))


if __name__=='__main__':
    main()
