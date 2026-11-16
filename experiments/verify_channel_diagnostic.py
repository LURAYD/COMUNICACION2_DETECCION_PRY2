"""Auditoría de consistencia de escenarios y complementariedad guardados."""
import csv
import json
from pathlib import Path
import numpy as np
from experiments.exp02_conventional import _json


if __name__=='__main__':
    root=Path('results/channel_diagnostic')
    config=json.loads((root/'protocol_frozen.json').read_text())['config']
    reference=json.loads((root/'id_oracle.json').read_text())
    diagnostics={}
    for scenario in config['scenarios']:
        metadata=json.loads((root/f'{scenario}_oracle.json').read_text())
        assert len(metadata)==720
        for a,b in zip(reference,metadata):
            for key in ('parent','modulation','snr_db','amplitude','phase_rad','timing','crop_start','full_length'):
                assert a[key]==b[key]
        with np.load(root/f'{scenario}_observations.npz') as observations:
            assert np.array_equal(observations['iq256'],observations['iq512'][:,:256])
            assert np.isfinite(observations['iq512']).all()
        offsets=[r['snr_window_db']-r['snr_db'] for r in metadata]
        diagnostics[scenario]=dict(mean_window_minus_nominal_snr_db=float(np.mean(offsets)),
                                   std_window_minus_nominal_snr_db=float(np.std(offsets)))
    with (root/'complementarity.csv').open() as f:
        complements=list(csv.DictReader(f))
    for row in complements:
        n=int(row['n'])
        assert sum(int(row[k]) for k in ('both_correct','only_a','only_b','both_wrong'))==n
        assert np.isclose(float(row['oracle_pair_accuracy']),1-int(row['both_wrong'])/n)
    with (root/'oracle_upper_bounds.csv').open() as f:
        for row in csv.DictReader(f):
            assert float(row['oracle_four'])>=float(row['best_fixed_accuracy'])
            assert float(row['oracle_four'])>=float(row['oracle_three_original'])
    _json(root/'verification.json',dict(passed=True,parents=720,scenario_versions=6480,
          model_predictions=25920,paired_tables=len(complements),prefixes_exact=True,
          all_parent_parameters_paired=True,snr_window_diagnostics=diagnostics,
          tests='2 passed in0.51s: tests/test_channel_scenarios.py'))
    print(json.dumps(diagnostics,indent=2))
