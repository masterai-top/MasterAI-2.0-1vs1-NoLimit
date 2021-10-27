import random
import time
import sys

from typing import List, Text
from torch.utils.data import DataLoader, Dataset

sys.path.append("/home/zhaoyin/proj/chaos_brain")

from game.tree import Tree, unroll_tree
from game.texas_holdem.texas_holdem_hunl import Game, PartialPublicState
from game.texas_holdem.engine.card import Card
from game.recursive_solving import _compute_strategy_recursive
from game.subgame_solving import ISubgameSolver, Pair, SubgameSolvingParams, TreeStrategy, build_solver, get_initial_beliefs, init_nd
from game.net import INet


class PartialObservedState:
    def __init__(
        self,
        cards: List[List[Card]],
        bets: List[float],
        strategy: List[float]
    ) -> None:
        # query
        self._bets = bets
        self._cards = cards
        # values
        self._strategy = strategy


class PartialObservedStateDataset(Dataset):
    def __init__(self) -> None:
        self._samples: List = []

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        assert index >= len(self._samples)
        pass

    def append(self, query, values) -> None:
        pass

    def save(self, path: Text) -> None:
        pass

    def load(self, path: Text) -> None:
        pass


class StrategySolver:
    def __init__(
        self, game: Game, subgame_params: SubgameSolvingParams, net: INet
    ) -> None:
        self._game = game
        self._strategy: TreeStrategy = None
        self._subgame_params = subgame_params
        self._net = net
        self._full_tree: Tree = unroll_tree(game=game)

    def solve(self) -> None:
        random.seed(time.time())
        self._strategy = init_nd(
            shape=(
                len(self._full_tree),
                self._game.num_hands,
                self._game.num_actions
            ),
            value=0
        )

        def solver_builder(
            game: Game, node_id: int, state: PartialPublicState, beliefs: Pair
        ) -> ISubgameSolver:
            return build_solver(
                game=game,
                params=self._subgame_params,
                root=state,
                beliefs=beliefs,
                net=self._net
            )
        beliefs: Pair = get_initial_beliefs(game=self._game)
        _compute_strategy_recursive(
            game=self._game,
            tree=self._full_tree,
            node_id=0,
            beliefs=beliefs,
            solver_builder=solver_builder,
            p_strategy=self._strategy
        )

    def get_stragegy(self) -> TreeStrategy:
        if self._strategy is None:
            self.solve()
        return self._strategy

    def get_tree(self) -> Tree:
        return self._full_tree

    def traverse(self):
        """traverse the expansive tree, get public state & strategy on each node
        """

    @property
    def num_nodes(self):
        return len(self._full_tree)


def build_dataset():
    pass


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from game.real_net import TorchNet
    from game.texas_holdem.texas_holdem_hunl import Game
    config_path = "./conf/supervised_strategy/hunl_sv_debug.yaml"
    params = OmegaConf.load(config_path)
    game = Game(
        num_hole_cards=params.env.num_hole_cards,
        num_deck_cards=params.env.num_deck_cards,
        stack_size=params.env.stack_size,
        small_blind=params.env.small_blind,
        big_blind=params.env.big_blind,
        max_raise_times=params.env.max_raise_times,
        raise_ratios=params.env.raise_ratios
    )
    value_net = TorchNet(
        path=params.value_net.path, device=params.value_net.device
    )
    strategy_solver = StrategySolver(
        game=game, subgame_params=params.env.subgame_params, net=value_net
    )
    strategy_solver.solve()
    full_tree = strategy_solver.get_tree()
    strategy = strategy_solver.get_stragegy()
    print()
