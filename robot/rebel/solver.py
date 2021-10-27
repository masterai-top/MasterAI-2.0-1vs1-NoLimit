from functools import wraps
import threading
from game.net import INet
from omegaconf import OmegaConf
from typing import Text
from game.real_net import TorchNet
from game.recursive_solving import SubgameSolverBuilder
from game.subgame_solving import ISubgameSolver, Pair, TreeStrategy, \
    build_solver, get_initial_beliefs
from game.texas_holdem.texas_holdem_hunl import Game, PartialPublicState


CONFIG_PATH = "./conf/deploy/rebel.yaml"
PARAMS = OmegaConf.load(CONFIG_PATH)


def singleton(cls: object):
    """the singleton with a lock to make sure safe in multi threads"""
    _instance = {}
    _mutex = threading.Lock()

    @wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            with _mutex:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]
    return inner


@singleton
class Net(TorchNet):
    """a net class as a singleton
    """
    pass


def create_net(path: Text, device: Text = "cpu"):
    """creat an eval-only connector to the net as a singleton
    """
    return Net(path=path, device=device)


def _compute_strategy(
    game: Game,
    state: PartialPublicState,
    beliefs: Pair,
    solver_builder: SubgameSolverBuilder
) -> None:
    """compute strategies for the root
    """
    if game.is_terminal(state):
        return
    solver: ISubgameSolver = solver_builder(game, state, beliefs)
    solver.multistep()
    return solver.get_strategy()[0]


def _compute_strategy_with_solver(
    game: Game, state: PartialPublicState, solver_builder: SubgameSolverBuilder
) -> TreeStrategy:
    beliefs: Pair = get_initial_beliefs(game=game)
    return _compute_strategy(
        game=game,
        state=state,
        beliefs=beliefs,
        solver_builder=solver_builder,
    )


def compute_strategy(game: Game, state: PartialPublicState) -> TreeStrategy:
    """compute strategy by recursively solving subgames
    use only the strategy at root of the game for the full tree,
    and proceed to its children
    """
    subgame_params = PARAMS.env.subgame_params
    net: INet = create_net(path=PARAMS.model.path, device=PARAMS.model.device)

    def solver_builder(
        game: Game, state: PartialPublicState, beliefs: Pair
    ) -> ISubgameSolver:
        return build_solver(
            game=game,
            params=subgame_params,
            root=state,
            beliefs=beliefs,
            net=net
        )
    return _compute_strategy_with_solver(
        game=game, state=state, solver_builder=solver_builder
    )
