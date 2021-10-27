import time
import sys
import os

sys.path.append(os.getcwd())
from robot.rebel.solver import Game, PARAMS, compute_strategy


def test_solver():
    env = PARAMS.env
    game = Game(
        num_hole_cards=env.num_hole_cards,
        num_deck_cards=env.num_deck_cards,
        stack_size=env.stack_size,
        max_raise_times=env.max_raise_times
    )
    state = game.get_initial_state()
    start = time.time()
    strategy = compute_strategy(game=game, state=state)
    end = time.time()
    print("strategy: ", strategy)
    print("shape: ", strategy.shape)
    print("elapsed time: %.5fs" % (end - start))
    return end - start


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S'
    )
    elapsed_time = 0
    n_sims = 10
    for _ in range(n_sims):
        elapsed_time += test_solver()
    elapsed_time /= n_sims
    print("average elapsed time per inference: %.5fs" % elapsed_time)
