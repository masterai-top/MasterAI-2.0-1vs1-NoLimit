import random
from typing import List, Union

# # raise a RuntimeWarning thrown by np
# # only for debug
# import numpy as np
# np.seterr(all="raise")

ALMOST_ZERO = 1e-20


def normalize_probabilities(
    unnormed_probs: List[float],
    p_probs: List[float] = None
) -> Union[List[float], None]:
    def _norm_prob(p_probs):
        sum_ = unnormed_probs.sum()
        assert sum_ >= ALMOST_ZERO, \
            "error: probs_sum = {}, probs = {}".format(sum_, p_probs)
        p_probs[...] = unnormed_probs / sum_
    if p_probs is None:
        p_probs = unnormed_probs.copy()
        _norm_prob(p_probs)
        return p_probs
    else:
        _norm_prob(p_probs)


def normalize_probabilities_safe(
    unnormed_probs: List[float],
    eps: float = 1e-5
) -> List[float]:
    probs = unnormed_probs.copy()
    probs += eps
    sum_ = probs.sum()
    assert sum_ > ALMOST_ZERO, \
        "error: probs_sum = {}, probs = {}".format(sum_, probs)
    probs /= sum_
    return probs


def sampling(unnormed_probs: List[float]) -> int:
    probs = unnormed_probs.copy()
    normalize_probabilities(probs, probs)
    r = random.random()
    cum_prob = 0
    for i, prob in enumerate(probs):
        cum_prob += prob
        if r < cum_prob:
            return i
