from random import random


def sample_action(probs):
    rand = random()
    cum_prob = 0
    for idx, prob in enumerate(probs):
        cum_prob += prob
        if rand < cum_prob:
            return idx
