import threshold as t
import numpy as np

def test_apply_threshold():
    
    x = np.array([0, 0.1, 0.85, 0, 0, 0, 0.05, 0])
    
    assert t.apply_threshold(x, 0.80) == 2
    assert t.apply_threshold(x, 0.90) == 1

    