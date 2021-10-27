import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
from time import time
from typing import Any, List, Optional, Text, TextIO, Tuple, Union

from .net import INet
from .texas_holdem.texas_holdem_hunl import Game, PartialPublicState
from .tree import ChildrenActionIt, ChildrenIt, Tree, unroll_tree
from .utils import normalize_probabilities, normalize_probabilities_safe
from conf.dtypes import TORCH_FLOAT
from .texas_holdem.engine.card import Card
from .texas_holdem.engine.hand_eval import HandEvaluator

REACH_SMOOTHING_EPS = 1e-10
REGRET_SMOOTHING_EPS = 1e-10
assert Game.NUM_PLAYERS == 2, "error, only head to head games are supported"
Pair = Tuple[Any, Any]
TreeStrategy = List[List[List[float]]]


class ISubgameSolver(ABC):
    """solve to the end of the current street
    a subgame starts at the beginning of one street and ends at the start
    of the next street
    """

    @abstractmethod
    def get_hand_values(self, player_id: int) -> List[float]:
        """get values for each hand at the top of the game"""
        pass

    @abstractmethod
    def print_strategy(self, path: Text):
        pass

    @abstractmethod
    def step(self, tranverser: int):
        pass

    @abstractmethod
    def multistep(self):
        """make params.num_iter steps"""
        pass

    @abstractmethod
    def get_strategy(self) -> TreeStrategy:
        """matrix of shape [node, hand, action]:
        responses for every hand and node
        """
        pass

    def get_sampling_strategy(self) -> TreeStrategy:
        """strategy to use to choose next node in MDP"""
        return self.get_strategy()

    def get_belief_propogation_strategy(self) -> TreeStrategy:
        """strategy to use to compute beliefs in a leaf node to create
        a new subgame in the node
        """
        return self.get_sampling_strategy()

    @abstractmethod
    def update_value_network(self):
        """send current value estimation at the root node to the network
        """
        pass

    @abstractmethod
    def get_tree(self) -> Tree:
        pass


class SubgameSolvingParams:
    """
    """

    def __init__(self) -> None:
        # common FP-CFR params
        self._num_iters: int = 10
        self._max_depth: int = 2
        self._linear_update: bool = False
        # whetehr to use FP or CFR
        self._use_cfr: bool = True
        # FP only params
        self._optimistic: bool = False
        # CFR only
        self._dcfr: bool = False
        self._dcfr_alpha: float = 0
        self._dcfr_beta: float = 0
        self._dcfr_gamma: float = 0

    @property
    def num_iters(self) -> int:
        return self._num_iters

    @num_iters.setter
    def num_iters(self, x: int) -> None:
        self._num_iters = x

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, x: int) -> None:
        self._max_depth = x

    @property
    def linear_update(self) -> bool:
        return self._linear_update

    @linear_update.setter
    def linear_update(self, x: bool) -> None:
        self._linear_update = x

    @property
    def use_cfr(self) -> bool:
        return self._use_cfr

    @use_cfr.setter
    def use_cfr(self, x: bool) -> None:
        self._use_cfr = x

    @property
    def dcfr(self) -> bool:
        return self._dcfr

    @dcfr.setter
    def dcfr(self, x: bool) -> None:
        self._dcfr = x

    @property
    def dcfr_alpha(self) -> float:
        return self._dcfr_alpha

    @dcfr_alpha.setter
    def dcfr_alpha(self, x: float) -> None:
        self._dcfr_alpha = x

    @property
    def dcfr_beta(self) -> float:
        return self._dcfr_beta

    @dcfr_beta.setter
    def dcfr_beta(self, x: float) -> None:
        self._dcfr_beta = x

    @property
    def dcfr_gamma(self) -> float:
        return self._dcfr_gamma

    @dcfr_gamma.setter
    def dcfr_gamma(self, x: float) -> None:
        self._dcfr_gamma = x


class TreeStrategyStats:
    def __init__(
        self,
        tree: Tree,
        reach_probabilities: Pair,
        values: Pair,
        node_values: Pair,
        node_reach: List[float]
    ) -> None:
        """
        args:

        tree:
        reach_probabilities[p][node][hand]: the probabiliy to get
            hand `hand` and use to play blueprint to reach node `node`
        values[p][node][hand]: expected value that player `p` can get
            - the games starts at node node
            - p has hand `hand`
            - op hands are defined as noramlized(reach_probabilities[node][1 - p])
        node_values[p][node]: expected value that player `p` can get
            - the games starts at node node
            - p hands are defined as noramlized(reach_probabilities[node][1]).
            - op hands are defined as noramlized(reach_probabilities[node][1 - p])
        node_reach: the probability to reach a public node if both players
            play by blueprint
        """
        self._tree = tree
        self._reach_probs = reach_probabilities
        self._values = values
        self._node_values = node_values
        self._node_reach = node_reach

    @property
    def node_reach(self) -> List[float]:
        return self._node_reach

    @property
    def node_values(self) -> Pair:
        return self._node_values

    @property
    def reach_probabilities(self) -> Pair:
        return self._reach_probs


class PartialTreeTraverser:
    """helper base class for tree traversing (head to head)
    """

    def __init__(
        self,
        game: Game,
        tree: Tree,
        value_net: Optional[INet] = None
    ) -> None:
        self._game = game
        self._tree = tree
        self._value_net = value_net
        # size of the inputs of the value network
        self._query_size = get_query_size(game=game)
        # size of the outputs of the value network
        self._output_size = game.num_hands
        # list of pseude leaf nodes, i.e., nodes where value net eval is needed
        self._pseudo_leaves_indices: List[int] = []
        # list of leaf nodes
        self._terminal_indices: List[int] = []
        self._leaf_values: List[float] = []
        if value_net is None:
            for node in tree:
                if not game.is_terminal(node.state) and (not node.num_children):
                    raise RuntimeError(
                        "error: found a node " +
                        game.state_to_string(node.state) +
                        " that is a non-final leaf. " +
                        "either provide value net or increase max_depth"
                    )
        else:
            # initialzer buffers to query the neural network
            for node_id in range(len(tree)):
                node = tree[node_id]
                state = node.state
                if (not node.num_children) and (not game.is_terminal(state)):
                    self._pseudo_leaves_indices.append(node_id)
        for node_id in range(len(tree)):
            node = tree[node_id]
            if game.is_terminal(node.state):
                self._terminal_indices.append(node_id)
        # ---------------------------------------------------
        # initializing traverser_values & reach_probabilities
        # ---------------------------------------------------
        # values for each node and hand for one of the players
        # shape [num_nodes, num_hands]
        # leaf values could be populated with precompute_leaf_values
        # it's up to subclasess to pupulate the rest
        self._traverser_values: List[List[float]] = init_nd(
            shape=(len(tree), game.num_hands), value=0
        )
        # probability to reach a specific node by a player with
        # specific under the average policy: [num_players, num_nodes, num_hands]
        # computed with precompute_reaches
        self._reach_probabilities: Pair = []
        self._reach_probabilities.append(
            init_nd(shape=(len(tree), game.num_hands), value=0)
        )
        self._reach_probabilities.append(
            init_nd(shape=(len(tree), game.num_hands), value=0)
        )

    def write_query(
        self,
        node_id: int,
        traverser: int,
        p_buffer: List[float]
    ) -> None:
        """write a single query to the buffer
        the query corresponds to the node as seen by tranverser
        """
        state = self._tree[node_id].state
        write_index = write_query_to(
            game=self._game,
            traverser=traverser,
            state=state,
            reaches1=self._reach_probabilities[self._game.PLAYER1][node_id],
            reaches2=self._reach_probabilities[self._game.PLAYER2][node_id],
            p_buffer=p_buffer
        )
        assert write_index == self._query_size

    def add_training_example(self, traverser: int, values: List[float]) -> None:
        query = []
        self.write_query(node_id=0, traverser=traverser, p_buffer=query)
        query_tensor = torch.tensor(
            data=query, dtype=TORCH_FLOAT).unsqueeze_(dim=0)
        value_tensor = torch.tensor(
            data=values, dtype=TORCH_FLOAT).unsqueeze_(dim=0)
        self._value_net.add_training_example(
            query=query_tensor, values=value_tensor
        )

    def precompute_reaches(
        self,
        strategy: TreeStrategy,
        initial_beliefs: Union[Pair, List[float]],
        player: int = None
    ) -> None:
        if player is None:
            compute_reach_probabilities(
                tree=self._tree,
                strategy=strategy,
                initial_beliefs=initial_beliefs[self._game.PLAYER1],
                player=self._game.PLAYER1,
                p_reach_probabilities=self._reach_probabilities[self._game.PLAYER1]
            )
            compute_reach_probabilities(
                tree=self._tree,
                strategy=strategy,
                initial_beliefs=initial_beliefs[self._game.PLAYER2],
                player=self._game.PLAYER2,
                p_reach_probabilities=self._reach_probabilities[self._game.PLAYER2]
            )
        else:
            compute_reach_probabilities(
                tree=self._tree,
                strategy=strategy,
                initial_beliefs=initial_beliefs,
                player=player,
                p_reach_probabilities=self._reach_probabilities[player]
            )

    def precompute_all_leaf_values(self, traverser: int) -> None:
        """compute values for leaf nodes
        for terminals exact value is used; for non-terminals value net is called
        reaches for both players must be precomputed
        """
        t_start = time()
        self.query_value_net(traverser=traverser)
        logging.debug(
            "[precompute_all_leaf_values]: value net - %.5f"
            % (time() - t_start)
        )
        t_start = time()
        self.populate_leaf_values()
        logging.debug(
            "[precompute_all_leaf_values]: pseudo leaves - %.5f"
            % (time() - t_start)
        )
        t_start = time()
        self.precompute_terminal_leaves_values(traverser=traverser)
        logging.debug(
            "[precompute_all_leaf_values]: terminal leaves - %.5f"
            % (time() - t_start)
        )
        t_start = time()

    def query_value_net(self, traverser: int) -> None:
        """query value net, weight by oponent reaches, and save result as
        leaf_values tensor
        """
        if not self._pseudo_leaves_indices:
            return
        assert self._value_net, "error: value_net is empty"
        # num_leaves = len(self._pseudo_leaves_indices)
        scalers = []
        net_query_buffer = []
        for node_id in self._pseudo_leaves_indices:
            net_query_buffer.append([])
            self.write_query(
                node_id=node_id,
                traverser=traverser,
                p_buffer=net_query_buffer[-1]
            )
            oppo_id = 1 - traverser
            scalers.append(
                self._reach_probabilities[oppo_id][node_id].sum()
            )
        # list -> tensor
        net_query_buffer = torch.tensor(
            data=net_query_buffer, dtype=TORCH_FLOAT
        )
        scalers = torch.tensor(data=scalers, dtype=TORCH_FLOAT)
        leaf_values = self._value_net.compute_values(
            query=net_query_buffer
        )
        assert leaf_values.size(0) == scalers.size(0)
        leaf_values *= scalers.unsqueeze(1)
        self._leaf_values = leaf_values.numpy()

    def populate_leaf_values(self) -> None:
        """copy results from leaf_values to corresponding nodes in
        traverser_values
        """
        if not self._pseudo_leaves_indices:
            return
        for row, node_id in enumerate(self._pseudo_leaves_indices):
            self._traverser_values[node_id] = self._leaf_values[row]

    def precompute_terminal_leaves_values(self, traverser: int) -> None:
        """populate traverser_values for terminal nodes
        """
        for node_id in self._terminal_indices:
            last_action = self._tree[node_id].state.last_action
            board_cards = self._tree[node_id].state.board_cards
            pot_size = self._tree[node_id].state.pot_size
            stack_size = self._game.stack_size
            oppo_id = 1 - traverser
            oppo_reach_probabilities = \
                self._reach_probabilities[oppo_id][node_id]
            self._traverser_values[node_id] = compute_expected_terminal_values(
                game=self._game,
                last_action=last_action,
                board_cards=board_cards,
                pot_size=pot_size / stack_size,
                # pot_size=pot_size,
                inverse=self._tree[node_id].state.player_id != traverser,
                oppo_reach_probabilities=oppo_reach_probabilities
            )


class BRSolver(PartialTreeTraverser):
    def __init__(self, game: Game, tree: Tree, value_net: INet = None) -> None:
        super(BRSolver, self).__init__(
            game=game, tree=tree, value_net=value_net
        )
        # indexed by [node, hand, action]
        self._br_strategy = init_nd(
            shape=(len(tree), game.num_hands, game.num_actions), value=0
        )

    def compute_br(
        self,
        traverser: int,
        opponent_strategy: TreeStrategy,
        initial_beliefs: Pair,
        p_values: List[float]
    ) -> TreeStrategy:
        """re-computes BR strategy for the traverser and returns its expected
        BR value and the best response strategy
        only values for nodes where traverser is acting are valid
        """
        self.precompute_reaches(
            strategy=opponent_strategy, initial_beliefs=initial_beliefs
        )
        self.precompute_all_leaf_values(traverser=traverser)
        for public_node_id in reversed(range(len(self._tree))):
            node = self._tree[public_node_id]
            if not node.num_children:
                # all leaf values are set by precompute_all_leaf_values
                continue
            state = node.state
            value = np.zeros_like(self._traverser_values[public_node_id])
            if state.player_id == traverser:
                best_action = np.zeros(shape=(self._game.num_hands, ))
                for child_node_id, action in ChildrenActionIt(
                    node=node, game=self._game
                ):
                    new_value = self._traverser_values[child_node_id]
                    if child_node_id == node.children_begin:
                        value[...] = new_value
                        best_action = action
                    is_better = new_value > value
                    value[is_better] = new_value
                    best_action[is_better] = action
                self._br_strategy[public_node_id] = 0
                self._br_strategy[public_node_id, :, best_action] = 1
            else:
                for child_node_id in ChildrenIt(node=node):
                    value += self._traverser_values[child_node_id]
            self._traverser_values[public_node_id] = value
        p_values[...] = self._traverser_values[0]
        return self._br_strategy


class FP(ISubgameSolver):
    def __init__(
        self,
        game: Game,
        tree_or_root: Union[Tree, PartialPublicState],
        value_net: INet,
        beliefs: Pair,
        params: SubgameSolvingParams
    ):
        raise NotImplementedError


class CFR(ISubgameSolver, PartialTreeTraverser):
    """
    """

    def __init__(
        self,
        game: Game,
        tree_or_root: Union[Tree, PartialPublicState],
        value_net: INet,
        beliefs: Pair,
        params: SubgameSolvingParams
    ) -> None:
        assert params.use_cfr, "error: flag `use_cfr` is false"
        assert (not params.linear_update) or (not params.dcfr)
        if isinstance(tree_or_root, PartialPublicState):
            tree = unroll_tree(
                game=game,
                root=tree_or_root,
                max_depth=params.max_depth,
                is_subgame=True
            )
        else:
            tree = tree_or_root
        super(CFR, self).__init__(
            game=game, tree=tree, value_net=value_net
        )
        self._params: SubgameSolvingParams = params
        # num step() done for the player
        self._num_steps: Pair = [0] * self._game.num_players
        # believes for both players, [num_players, num_hands]
        self._initial_beliefs: Pair = beliefs
        # strategies, indexed by [node, hand, action]
        # initial strategies are uniform over feasible actions
        self._average_strategy: TreeStrategy = get_uniform_strategy(
            game=game, tree=tree
        )
        self._last_strategy: TreeStrategy = self._average_strategy.copy()
        self._sum_strategy: TreeStrategy = get_uniform_reach_weigted_strategy(
            game=game, tree=tree, initial_beliefs=beliefs
        )
        self._regrets: TreeStrategy = init_nd(
            shape=(len(tree), game.num_hands, game.num_actions),
            value=0
        )
        self._root_values: Pair = [
            np.zeros(shape=(game.num_hands, ))
            for _ in range(self._game.num_players)
        ]
        self._root_values_means: Pair = [
            np.zeros(shape=(game.num_hands, ))
            for _ in range(self._game.num_players)
        ]
        # buffer to store reach probabilties for the last_strategies
        self._reach_probabilities_buffer: List[List[float]] = init_nd(
            shape=(len(tree), game.num_hands),
            value=0
        )

    def update_regrets(self, traverser: int) -> None:
        """adds regrets for the last_strategies to regrets
        sets traverser_values[node] to the EVs of last_strategies for traverser
        """
        t_start = time()
        self.precompute_reaches(
            strategy=self._last_strategy, initial_beliefs=self._initial_beliefs
        )
        logging.debug(
            "[update_regrets]: reaches - %.5f" % (time() - t_start)
        )
        t_start = time()
        self.precompute_all_leaf_values(traverser=traverser)
        logging.debug(
            "[update_regrets]: leaves - %.5f" % (time() - t_start)
        )
        t_start = time()
        for public_node_id in reversed(range(len(self._tree))):
            node = self._tree[public_node_id]
            if not node.num_children:
                # all leaf values are set by precompute_all_leaf_values
                continue
            state = node.state
            value = np.zeros_like(self._traverser_values[public_node_id])
            if state.player_id == traverser:
                for child_node_id, action in ChildrenActionIt(
                    node=node, game=self._game
                ):
                    action_value = self._traverser_values[child_node_id]
                    self._regrets[public_node_id, :, action] += action_value
                    value += action_value * \
                        self._last_strategy[public_node_id, :, action]
                for child_node_id, action in ChildrenActionIt(
                    node=node, game=self._game
                ):
                    self._regrets[public_node_id, :, action] -= value
            else:
                assert state.player_id == 1 - traverser
                for child_node_id in ChildrenIt(node):
                    action_value = self._traverser_values[child_node_id]
                    value += action_value
            self._traverser_values[public_node_id] = value
        logging.debug(
            "[update_regrets]: regrets - %.5f" % (time() - t_start)
        )

    def step(self, traverser: int) -> None:
        t_start = time()
        self.update_regrets(traverser=traverser)
        logging.debug("[step]: update regrets - %.5f" % (time() - t_start))
        self._root_values[traverser] = self._traverser_values[0]
        alpha = 2 / (self._num_steps[traverser] + 2) \
            if self._params.linear_update \
            else 1 / (self._num_steps[traverser] + 1)
        self._root_values_means[traverser] += self._root_values[traverser] - \
            self._root_values_means[traverser] * alpha
        # discount
        pos_discount = 1
        neg_discount = 1
        strat_discount = 1
        # traverser always have uniform strategy, hence + 1
        num_strategies = self._num_steps[traverser] + 1
        if self._params.linear_update:
            pos_discount = neg_discount = strat_discount = \
                num_strategies / (num_strategies + 1)
        elif self._params.dcfr:
            if self._params.dcfr_alpha >= 5:
                pos_discount = 1
            else:
                pow_ = pow(num_strategies, self._params.dcfr_alpha)
                pos_discount = pow_ / (pow_ + 1)
            if self._params.dcfr_beta <= -5:
                neg_discount = 0
            else:
                pow_ = pow(num_strategies, self._params.dcfr_beta)
                neg_discount = pow_ / (pow_ + 1)
            strat_discount = pow(
                num_strategies / (num_strategies + 1), self._params.dcfr_gamma
            )
        # update instaneous strategy
        t_start = time()
        for node_id in range(len(self._tree)):
            if (
                (not self._tree[node_id].num_children)
                or (self._tree[node_id].state.player_id != traverser)
            ):
                continue
            feasible_actions = self._game.get_feasible_actions(
                state=self._tree[node_id].state
            )
            for action, is_feasible in enumerate(feasible_actions):
                if is_feasible:
                    self._last_strategy[node_id, :, action] = np.maximum(
                        self._regrets[node_id, :, action],
                        REACH_SMOOTHING_EPS
                    )
            for hand in range(self._game.num_hands):
                normalize_probabilities(
                    unnormed_probs=self._last_strategy[node_id, hand],
                    p_probs=self._last_strategy[node_id, hand]
                )
        logging.debug(
            "[step]: update instaneous strategy - %.5f" % (time() - t_start)
        )
        t_start = time()
        compute_reach_probabilities(
            tree=self._tree,
            strategy=self._last_strategy,
            initial_beliefs=self._initial_beliefs[traverser],
            player=traverser,
            p_reach_probabilities=self._reach_probabilities_buffer
        )
        logging.debug(
            "[step]: compute reach probabilities - %.5f" % (time() - t_start)
        )
        # update accumulative regret & strategy
        t_start = time()
        for node_id in range(len(self._tree)):
            if (
                (not self._tree[node_id].num_children)
                or (self._tree[node_id].state.player_id != traverser)
            ):
                continue
            feasible_actions = self._game.get_feasible_actions(
                state=self._tree[node_id].state
            )
            for action, is_feasible in enumerate(feasible_actions):
                if is_feasible:
                    pos_ = self._regrets[node_id, :, action] > 0
                    neg_ = np.logical_not(pos_)
                    self._regrets[node_id, :, action][pos_] *= pos_discount
                    self._regrets[node_id, :, action][neg_] *= neg_discount
                    self._sum_strategy[node_id, :, action] *= strat_discount
                    self._sum_strategy[node_id, :, action] += \
                        self._reach_probabilities_buffer[node_id] \
                        * self._last_strategy[node_id, :, action]
            for hand in range(self._game.num_hands):
                normalize_probabilities(
                    unnormed_probs=self._sum_strategy[node_id, hand],
                    p_probs=self._average_strategy[node_id, hand]
                )
        logging.debug(
            "[step]: update accumulative regret & strategy - %.5f"
            % (time() - t_start)
        )
        self._num_steps[traverser] += 1

    def multistep(self) -> None:
        for iter_ in range(self._params.num_iters):
            self.step(traverser=iter_ % self._game.num_players)

    def update_value_network(self) -> None:
        assert (self._num_steps[self._game.PLAYER1] > 0) \
            and (self._num_steps[self._game.PLAYER2] > 0)
        self.add_training_example(
            traverser=self._game.PLAYER1,
            values=self.get_hand_values(player_id=self._game.PLAYER1)
        )
        self.add_training_example(
            traverser=self._game.PLAYER2,
            values=self.get_hand_values(player_id=self._game.PLAYER2)
        )

    def get_strategy(self) -> TreeStrategy:
        return self._average_strategy

    def get_sampling_strategy(self) -> TreeStrategy:
        return self._last_strategy

    def get_belief_propogation_strategy(self) -> TreeStrategy:
        return self._last_strategy

    def print_strategy(self, path: Text = None) -> None:
        print_strategy(
            game=self._game,
            tree=self._tree,
            strategy=self._average_strategy,
            fpath_or_fstream=path
        )

    def get_hand_values(self, player_id: int) -> List[float]:
        return self._root_values_means[player_id]

    def get_tree(self) -> Tree:
        return self._tree


def init_nd(
    shape, value: Any, dtype=None
) -> Union[List[List[Any]], List[List[List[Any]]]]:
    # default dtype is np.float64
    # dtype = np.float32 if dtype is None else dtype
    array = np.ones(shape=shape, dtype=dtype)
    array *= value
    return array


def get_query_size(game: Game) -> int:
    """
    agent index         : 1
    acting agent        : 1
    pot                 : 1
    board               : 5
    infostate beliefs   : 2 x C^2_52
    """
    return 8 + 2 * game.num_hands


def write_query_to(
    game: Game,
    traverser: int,
    state: PartialPublicState,
    reaches1: List[float],
    reaches2: List[float],
    p_buffer: List[float]
) -> int:
    assert not state.is_terminal
    write_index = 0
    # agent index
    p_buffer.append(float(state.player_id))
    write_index += 1
    # acting agent
    p_buffer.append(float(traverser))
    write_index += 1
    # pot size
    p_buffer.append(state.pot_size / game.stack_size)
    write_index += 1
    # feasible actions
    # feasible_actions = game.get_feasible_actions(state=state)
    # for is_feasible in feasible_actions:
    #     p_buffer.append(float(is_feasible))
    # write_index += game.num_actions
    # board cards
    board_cards = [-1.] * 5
    for i, card in enumerate(state.board_cards):
        # card_id (1 - 52) -> embedding id (0 - 51)
        board_cards[i] = float(card.id - 1)
    p_buffer += board_cards
    write_index += 5
    # infostate beliefs (2 x C_52^2)
    probs = normalize_probabilities_safe(
        unnormed_probs=reaches1, eps=REACH_SMOOTHING_EPS
    )
    p_buffer += probs.tolist()
    write_index += len(reaches1)
    probs += normalize_probabilities_safe(
        unnormed_probs=reaches2, eps=REACH_SMOOTHING_EPS
    )
    p_buffer += probs.tolist()
    write_index += len(reaches2)
    return write_index


def get_query(
    game: Game,
    traverser: int,
    state: PartialPublicState,
    reaches1: List[float],
    reaches2: List[float]
) -> List[float]:
    query: List[float] = [0.] * get_query_size(game=game)
    write_query_to(
        game=game,
        traverser=traverser,
        state=state,
        reaches1=reaches1,
        reaches2=reaches2,
        p_buffer=query
    )
    return query


def deserialize_query(
    game: Game, query: List[float]
) -> Tuple[int, PartialPublicState, List[float], List[float]]:
    idx = 0
    state: PartialPublicState = None
    raise NotImplementedError


def get_uniform_strategy(game: Game, tree: Tree) -> TreeStrategy:
    strategy: TreeStrategy = init_nd(
        shape=(len(tree), game.num_hands, game.num_actions),
        value=0
    )
    for node_id in range(len(tree)):
        # get valid actions at each node
        feasible_actions = game.get_feasible_actions(state=tree[node_id].state)
        # terminal nodes
        if sum(feasible_actions) == 0:
            continue
        strategy[node_id, :, feasible_actions] = 1 / sum(feasible_actions)
    return strategy


def get_uniform_reach_weigted_strategy(
    game: Game, tree: Tree, initial_beliefs: Pair
) -> TreeStrategy:
    strategy: TreeStrategy = get_uniform_strategy(game=game, tree=tree)
    reach_probabilities_buffer = init_nd(
        shape=(len(tree), game.num_hands), value=0
    )
    for traverser in range(game.num_players):
        compute_reach_probabilities(
            tree=tree,
            strategy=strategy,
            initial_beliefs=initial_beliefs[traverser],
            player=traverser,
            p_reach_probabilities=reach_probabilities_buffer
        )
        for node_id in range(len(tree)):
            if (
                not tree[node_id].num_children
                or tree[node_id].state.player_id != traverser
            ):
                continue
            feasible_actions = game.get_feasible_actions(
                state=tree[node_id].state
            )
            for action, is_feasible in enumerate(feasible_actions):
                if is_feasible:
                    strategy[node_id, :, action] *= \
                        reach_probabilities_buffer[node_id]
    return strategy


def print_strategy(
    game: Game,
    tree: Tree,
    strategy: TreeStrategy,
    fpath_or_fstream: Union[Text, TextIO] = None
) -> None:
    assert len(tree) == len(strategy)
    if isinstance(fpath_or_fstream, Text):
        f = open(fpath_or_fstream, mode="w")
    f = fpath_or_fstream
    print("printing strategies per node", file=f)
    print("-------->", file=f)
    hands = game.hands
    for node_id in range(len(strategy)):
        state = tree[node_id].state
        if not tree[node_id].num_children:
            continue
        print(
            "node_id = %d, \t, state = %s "
            % (node_id, game.state_to_string(state)),
            end="", file=f
        )
        for hand in range(len(strategy[node_id])):
            hole_cards = hands[hand]
            hole_cards = ", ".join([str(card) for card in hole_cards])
            print("| hand = %s: " % hole_cards, end="", file=f)
            for val in strategy[node_id][hand]:
                print("%.2f " % val, end="", file=f)
        print("", file=f)
    if f is not None:
        f.close()


def get_initial_beliefs(game: Game) -> Pair:
    beliefs: Pair = [
        np.ones(shape=(game.num_hands, )) / game.num_hands
        for _ in range(game.num_players)
    ]
    return beliefs


def build_solver(
    game: Game,
    params: SubgameSolvingParams,
    root: PartialPublicState = None,
    beliefs: Pair = None,
    net: INet = None
) -> ISubgameSolver:
    if root is None:
        root = game.get_initial_state()
        beliefs = get_initial_beliefs(game)
    if params.use_cfr:
        return CFR(
            game=game,
            tree_or_root=root,
            value_net=net,
            beliefs=beliefs,
            params=params
        )
    else:
        return FP(
            game=game,
            tree_or_root=root,
            value_net=net,
            beliefs=beliefs,
            params=params
        )


def compute_exploitability(game: Game, strategy: TreeStrategy) -> float:
    exploitabilites = compute_exploitability2(game=game, strategy=strategy)
    return sum(exploitabilites) / game.num_players


def compute_exploitability2(
    game: Game, strategy: TreeStrategy
) -> Tuple[float, float]:
    root: PartialPublicState = game.get_initial_state()
    tree: Tree = unroll_tree(
        game=game, root=root, max_depth=int(1e6)
    )
    beliefs: Pair = [
        np.ones(shape=(game.num_hands, )) / game.num_hands
        for _ in range(game.num_players)
    ]
    solver: BRSolver = BRSolver(
        game=game, tree=tree, value_net=None
    )
    values0: List[float] = []
    values1: List[float] = []
    solver.compute_br(
        traverser=game.PLAYER1,
        opponent_strategy=strategy,
        initial_beliefs=beliefs,
        p_values=values0
    )
    solver.compute_br(
        traverser=game.PLAYER2,
        opponent_strategy=strategy,
        initial_beliefs=beliefs,
        p_values=values1
    )
    return sum(values0) / len(values0), sum(values1) / len(values1)


def compute_strategy_stats(
    game: Game, strategy: TreeStrategy
) -> TreeStrategyStats:
    raise NotImplementedError


def compute_ev(
    game: Game, strategy1: TreeStrategy, strategy2: TreeStrategy
) -> List[float]:
    raise NotImplementedError


def compute_ev2(
    game: Game, strategy1: TreeStrategy, strategy2: TreeStrategy
) -> Pair:
    raise NotImplementedError


def compute_immediate_regrets(
    game: Game, strategies: List[TreeStrategy]
) -> List[List[float]]:
    raise NotImplementedError


def compute_reach_probabilities(
    tree: Tree,
    strategy: TreeStrategy,
    initial_beliefs: List[float],
    player: int,
    p_reach_probabilities: List[List[float]]
) -> None:
    """ for each node `x` and hand `h` computes
    P(root->x, h | beliefs) := pi^{player}(root->x|h) * P(h).
    """
    num_nodes = len(tree)
    for node_id in range(num_nodes):
        if node_id == 0:
            p_reach_probabilities[node_id] = initial_beliefs
        else:
            node = tree[node_id]
            state = node.state
            last_action_player_id = tree[node.parent_id].state.player_id
            last_action = state.last_action
            if player == last_action_player_id:
                p_reach_probabilities[node_id] = \
                    p_reach_probabilities[node.parent_id] \
                    * strategy[node.parent_id, :, last_action]
            else:
                p_reach_probabilities[node_id] = \
                    p_reach_probabilities[node.parent_id]


def compute_expected_terminal_values(
    game: Game,
    last_action: int,
    board_cards: List[Card],
    pot_size: float,
    inverse: bool,
    oppo_reach_probabilities: List[float]
) -> List[float]:
    """
    pot_size = pot_size / init_stack_size
    """
    # need to convert the probabilities to the payoff of the traverser
    # note, the probabilities are true probabilities iff op_beliefs sum to 1
    belief_sum = oppo_reach_probabilities.sum()
    if game.action_id2str[last_action] == game.FOLD:
        values = belief_sum * np.ones_like(oppo_reach_probabilities)
    else:
        values = compute_win_probability(
            game=game, board_cards=board_cards, beliefs=oppo_reach_probabilities
        )
    # payoff: probability(win) * pot_size + probability(lose) * (-pot_size)
    # values <- ((values / belief_sum) * 2 - 1) * belief_sum * pot_size
    values[...] = (values * 2 - belief_sum) * pot_size
    if inverse:
        values *= -1
    return values


def compute_win_probability(
    game: Game, board_cards: List[Card], beliefs: List[float]
) -> Tuple[List[float], List[float]]:
    """computes probabilities to win the game for each possible hand assuming
    that the oponents hands are distributed according to beliefs
    """
    assert len(board_cards) == 5
    hand_evals = []
    for hole_cards in game.hands:
        hand_evals.append(HandEvaluator.eval_hand(
            hole_cards=hole_cards, board_cards=board_cards
        ))
    hand_evals = np.array(hand_evals)
    values = np.zeros_like(hand_evals, dtype=np.float)
    for i, player_hand_eval in enumerate(hand_evals):
        wins = player_hand_eval > hand_evals
        ties = player_hand_eval == hand_evals
        values[i] = beliefs[wins].sum()
        values[i] += beliefs[ties].sum() / 2
    return values


# def compute_win_probability(
#     game: Game, board_cards: List[Card], beliefs: List[float]
# ) -> Tuple[List[float], List[float]]:
#     """computes probabilities to win the game for each possible hand assuming
#     that the oponents hands are distributed according to beliefs
#     """
#     assert len(board_cards) == 5
#     hand_evals = []
#     for hole_cards in game.hands:
#         hand_evals.append(HandEvaluator.eval_hand(
#             hole_cards=hole_cards, board_cards=board_cards
#         ))
#     cnt = Counter(hand_evals)
#     ascending_values = sorted(cnt, reverse=False)
#     ascending_regions = {}
#     begin = 0
#     for v in ascending_values:
#         end = begin + cnt[v]
#         ascending_regions[v] = begin, end
#         begin = end
#     hand_evals = np.array(hand_evals)
#     ascending_hand_indices = np.argsort(hand_evals)
#     values = np.zeros_like(hand_evals, dtype=np.float)
#     for i, player_hand_eval in enumerate(hand_evals):
#         begin, end = ascending_regions[player_hand_eval]
#         wins = ascending_hand_indices[: begin]
#         ties = ascending_hand_indices[begin: end]
#         values[i] = beliefs[wins].sum()
#         values[i] += beliefs[ties].sum() / 2
#     return values
