from typing import Text, Tuple

from game.net import INet
from game.stats import eval_net
from game.subgame_solving import ISubgameSolver, SubgameSolvingParams, \
    TreeStrategy, build_solver, compute_exploitability, print_strategy
from game.real_net import create_torch_net
from game.recursive_solving import RecursiveSolvingParams, \
    compute_strategy_recursive, compute_strategy_recursive_to_leaf
from game.texas_holdem.texas_holdem_hunl import Game
from game.tree import unroll_tree
from .data_loop import CVNetBufferConnector, DataThreadLoop
from .thread_loop import ThreadLoop
from .prioritized_replay import ValuePrioritizedReplay
from .model_locker import ModelLocker


def create_cfr_thread(
    model_locker: ModelLocker,
    replay_buffer: ValuePrioritizedReplay,
    cfg: RecursiveSolvingParams,
    seed: int
) -> ThreadLoop:
    connector = CVNetBufferConnector(
        model_locker=model_locker, replay_buffer=replay_buffer
    )
    return DataThreadLoop(connector=connector, cfg=cfg, seed=seed)


def _compute_exploitability(
    params: RecursiveSolvingParams, model_path: Text
) -> float:
    game = Game(
        num_hole_cards=params.num_hole_cards,
        num_deck_cards=params.num_deck_cards,
        stack_size=params.stack_size,
        small_blind=params.small_blind,
        big_blind=params.big_blind,
        max_raise_times=params.max_raise_times,
        raise_ratios=params.raise_ratios
    )
    net: INet = create_torch_net(path=model_path)
    tree_strategy: TreeStrategy = compute_strategy_recursive(
        game=game, subgame_params=params.subgame_params, net=net
    )
    print_strategy(
        game=game,
        tree=unroll_tree(game=game),
        strategy=tree_strategy
    )
    return compute_exploitability(game=game, strategy=tree_strategy)


def compute_stats_with_net(
    params: RecursiveSolvingParams, model_path: Text
) -> Tuple[float, float, float]:
    game = Game(
        num_hole_cards=params.num_hole_cards,
        num_deck_cards=params.num_deck_cards,
        stack_size=params.stack_size,
        small_blind=params.small_blind,
        big_blind=params.big_blind,
        max_raise_times=params.max_raise_times,
        raise_ratios=params.raise_ratios
    )
    net: INet = create_torch_net(path=model_path)
    net_strategy = compute_strategy_recursive_to_leaf(
        game=game, subgame_params=params.subgame_params, net=net
    )
    print_strategy(
        game=game,
        tree=unroll_tree(game=game),
        strategy=net_strategy
    )
    exploitability = compute_exploitability(game=game, strategy=net_strategy)
    full_params: SubgameSolvingParams = params.subgame_params
    full_params.max_depth = int(1e6)
    fp: ISubgameSolver = build_solver(game=game, params=full_params)
    fp.multistep()
    full_strategy: TreeStrategy = fp.get_strategy()
    mse_net_traverse = eval_net(
        game=game,
        net_strategy=net_strategy,
        full_strategy=full_strategy,
        mdp_depth=params.subgame_params.max_depth,
        fp_iters=params.subgame_params.num_iters,
        net=net,
        traverse_by_net=True,
        verbose=True
    )
    mse_full_traverse = eval_net(
        game=game,
        net_strategy=net_strategy,
        full_strategy=full_strategy,
        mdp_depth=params.subgame_params.max_depth,
        fp_iters=params.subgame_params.num_iters,
        net=net,
        traverse_by_net=False,
        verbose=True
    )
    return exploitability, mse_net_traverse, mse_full_traverse
