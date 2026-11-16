import numpy as np
import pytest
from src.adaptive import observe_prefix,action_cost_ms,choose_actions


def test_selector_rejects_future_samples_before_model_call():
    with pytest.raises(ValueError):
        observe_prefix(np.ones((1,512),complex),None)
    with pytest.raises(ValueError):
        observe_prefix(np.ones(256,complex),None)


def test_accumulated_cost_and_fixed_cnn_are_distinct():
    a,c=action_cost_ms([0,1,2],[2.,3.,.5],gate_ms=1.)
    np.testing.assert_array_equal(a,[32.,64.,32.])
    np.testing.assert_array_equal(c,[3.,6.,3.5])
    a,c=action_cost_ms([2],[2.,3.,.5],routed=False)
    assert c[0]==.5 and a[0]==32


def test_observable_rule_needs_no_oracle_or_future_outputs():
    observation={'flatness':np.array([.1,.9])}
    np.testing.assert_array_equal(choose_actions(observation,'flatness',{'flatness_threshold':.5}),[1,2])
