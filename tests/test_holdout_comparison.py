import numpy as np
from experiments.exp06_final_holdout import paired_bootstrap


def test_paired_bootstrap_identical_predictions_have_zero_interval():
    y = np.array([0,0,1,1])
    p = np.array([0,1,1,0])
    result = paired_bootstrap(y,np.array([0,0,5,5]),p,p,replicates=100)
    assert result['accuracy_difference'] == 0
    assert result['ci95_percentile'] == [0,0]


def test_paired_bootstrap_preserves_direction_and_strata():
    y = np.array([0,0,1,1])
    result = paired_bootstrap(y,np.array([0,0,5,5]),y,1-y,replicates=100)
    assert result['accuracy_difference'] == 1
    assert result['ci95_percentile'] == [1,1]
    assert result['a_correct_b_wrong'] == 4
