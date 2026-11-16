import json
from pathlib import Path
import numpy as np
from experiments.channel_scenarios import scenario_parent


def test_controlled_pairs_share_parent_and_noise():
    base=json.loads(Path('configs/dataset_e2.json').read_text())
    a,ia,da=scenario_parent(base,19,0,'QPSK',10,'id')
    b,ib,db=scenario_parent(base,19,0,'QPSK',10,'cfo_shift')
    c,ic,dc=scenario_parent(base,19,0,'QPSK',10,'multipath_shift')
    assert ia['crop_start']==ib['crop_start']==ic['crop_start']
    assert ia['full_length']==ib['full_length']==ic['full_length']
    np.testing.assert_array_equal(da['bits'],dc['bits'])
    np.testing.assert_allclose(da['standardized_noise'],dc['standardized_noise'],atol=1e-14)
    np.testing.assert_allclose(abs(da['signal']),abs(db['signal']),atol=1e-14)
    np.testing.assert_allclose(da['noise'],db['noise'],atol=1e-14)
    assert ia['beta']==ib['beta']==ic['beta']
    assert ia['cfo_hz']==ic['cfo_hz']
    assert abs(ia['cfo_hz'])<=100 and 200<=abs(ib['cfo_hz'])<=300
    assert not np.array_equal(a,b) and not np.array_equal(a,c)
    np.testing.assert_array_equal(a,scenario_parent(base,19,0,'QPSK',10,'id')[0])


def test_beta_only_and_independent_parents():
    base=json.loads(Path('configs/dataset_e2.json').read_text())
    a,ia,da=scenario_parent(base,19,0,'16QAM',0,'beta_035')
    b,ib,db=scenario_parent(base,19,0,'16QAM',0,'beta_065')
    for key in ('parent','cfo_hz','bank','channel_index','amplitude','phase_rad','crop_start'):
        assert ia[key]==ib[key]
    assert ia['beta']==.35 and ib['beta']==.65
    np.testing.assert_array_equal(da['bits'],db['bits'])
    assert not np.array_equal(a,scenario_parent(base,19,1,'16QAM',0,'beta_035')[0])
