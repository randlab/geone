import numpy as np
import unittest
from geone.cv_metrics import brier_score, zero_one_score, balanced_brier_score

class fake_estimator:
    def __init__(self):
        self.classes_ = np.array([0, 1, 3])
    def predict_proba(self, X):
        return X

class estimator_for_previous_testing(fake_estimator):
    def __init__(self):
        super().__init__()
        self.previous_X_ = np.array([[1,0,0]])
        self.previous_y_ = np.array([[0,0,1]])

class TestMetrics(unittest.TestCase):
    def test_all_scores(self):
        X1 = np.array([[1,0,0]])
        X2 = np.array([[0,1,0]])
        X3 = np.array([[0.5,0.2,0.3]])
        X4 = np.array([[0.2, 0.3, 0.5]])
        X5= np.array([[0.4, 0.4, 0.2]])
        assert brier_score(fake_estimator(), X1, [0]) == 0
        assert brier_score(fake_estimator(), X1, [1]) == -2
        assert brier_score(fake_estimator(), X2, [0]) == -2
        assert brier_score(fake_estimator(), X2, [1]) == 0
        assert brier_score(fake_estimator(), X3, [0]) == -0.38
        
        assert brier_score(fake_estimator(), np.vstack((X1,X3)), [0,0]) == -0.19
        assert brier_score(fake_estimator(), np.vstack((X1,X3)), [1,0]) == -1.19
        assert brier_score(fake_estimator(), np.vstack((X1,X2,X3)), [1,1,0]) == -2.38/3

        assert balanced_brier_score(fake_estimator(), np.vstack((X1,X2,X3)), [1,1,0]) == -1.38/2

        assert zero_one_score(fake_estimator(), X4, [3]) == 1  
        assert zero_one_score(fake_estimator(), X3, [0]) == 1  
        assert zero_one_score(fake_estimator(), X5, [1]) == 0.5
        assert zero_one_score(fake_estimator(), np.vstack((X3, X4, X5)), [1, 3, 0]) == 0.5

        assert zero_one_score(estimator_for_previous_testing(), X1, [3]) == 1
        assert brier_score(estimator_for_previous_testing(), X1, [3]) == 0
