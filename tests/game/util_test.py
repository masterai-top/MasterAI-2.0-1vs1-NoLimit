import numpy as np
from game.utils import normalize_probabilities, normalize_probabilities_safe


class TestProbs:
    def test_normalize_probabilities_safe(self):
        probs = np.array([2.93185e-81, 3.00956e-81, 3.17805e-81, 8.80785e-81])
        normalize_probabilities_safe(probs, probs)
        assert abs(probs.sum() - 1) < 1e-10
