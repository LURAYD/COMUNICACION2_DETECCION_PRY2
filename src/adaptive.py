"""Selector observable: solo prefijo256 y cálculos efectivamente ejecutados."""
import numpy as np
from sklearn.pipeline import Pipeline
from src.features import extract_feature_matrix,FEATURE_NAMES

FLATNESS_INDEX=FEATURE_NAMES.index('spectral_flatness')


def observe_prefix(iq,svm256):
    x=np.asarray(iq)
    if x.ndim!=2 or x.shape[1]!=256 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('El selector solo admite matrices IQ de256 muestras finitas.')
    features=extract_feature_matrix(x)
    classifier=Pipeline(svm256.steps[1:])
    labels=classifier.predict(features)
    scores=classifier.decision_function(features)
    ordered=np.sort(scores,axis=1)
    return dict(gate=np.concatenate([features,scores],axis=1),initial=labels,
                margin=(ordered[:,-1]-ordered[:,-2])[:,None],flatness=features[:,FLATNESS_INDEX])


def choose_actions(observation,kind,artifact):
    if kind=='confidence':
        confidence=artifact['calibrator'].predict_proba(observation['margin'])[:,1]
        return np.where(confidence<artifact['confidence_threshold'],1,0)
    if kind=='flatness':
        return np.where(observation['flatness']>artifact['flatness_threshold'],2,1)
    if kind=='selector':
        return artifact['selector'].predict(observation['gate']).argmax(axis=1)
    raise ValueError('Política desconocida.')


def action_cost_ms(actions,base_ms,gate_ms=0.,routed=True):
    actions=np.asarray(actions,dtype=int)
    if np.any((actions<0)|(actions>2)):
        raise ValueError('Acción inválida.')
    acquisition=np.where(actions==1,64.,32.)
    if routed:
        compute=base_ms[0]+gate_ms+np.where(actions==1,base_ms[1],np.where(actions==2,base_ms[2],0.))
    else:
        compute=np.asarray(base_ms)[actions]
    return acquisition,compute
