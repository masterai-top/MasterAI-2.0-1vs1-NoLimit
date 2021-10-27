import logging
import numpy as np
import random
import threading
from collections import deque
from copy import copy, deepcopy
from typing import Callable, List, Text, Union, Tuple

from .net import INet
from .subgame_solving import ISubgameSolver, Pair, SubgameSolvingParams, \
    TreeStrategy, REACH_SMOOTHING_EPS, build_solver, get_initial_beliefs, \
    init_nd
from .texas_holdem.texas_holdem_hunl import Action, Game, PartialPublicState
from .tree import Tree, unroll_tree
from .utils import normalize_probabilities_safe, sampling

SubgameSolverBuilder = Callable[
    [Game, int, PartialPublicState, Pair],
    ISubgameSolver
]


class RecursiveSolvingParams:
    def __init__(self) -> None:
        self._num_hole_cards: int = 2
        self._num_deck_cards: int = 52
        self._num_card_suits: int = 4
        self._num_card_ranks: int = 13
        self._num_actions: int = 9
        self._actions: List[Text] = ["fold", "call & check", "raise", "allin"]
        self._raise_ratios: List[float] = [0.333, 0.5, 0.667, 1, 1.5, 2]
        self._max_raise_times: int = 10
        self._stack_size: int = 200  # 100 BB
        self._small_blind: int = 1
        self._big_blind: int = 2
        self._random_action_prob: float = 1.0
        self._sample_leaf: bool = False
        self._subgame_params: SubgameSolvingParams = SubgameSolvingParams()

    @property
    def num_hole_cards(self) -> int:
        return self._num_hole_cards

    @num_hole_cards.setter
    def num_hole_cards(self, x: int) -> None:
        self._num_hole_cards = x

    @property
    def num_deck_cards(self) -> int:
        return self._num_deck_cards

    @num_deck_cards.setter
    def num_deck_cards(self, x: int) -> None:
        self._num_deck_cards = x

    @property
    def num_card_suits(self) -> int:
        return self._num_card_suits

    @num_card_suits.setter
    def num_card_suits(self, x: int) -> None:
        self._num_card_suits = x

    @property
    def num_card_ranks(self) -> int:
        return self._num_card_ranks

    @num_card_ranks.setter
    def num_card_ranks(self, x: int) -> None:
        self._num_card_ranks = x

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @num_actions.setter
    def num_actions(self, x: int) -> None:
        self._num_actions = x

    @property
    def actions(self) -> List[Text]:
        return self._actions

    @actions.setter
    def actions(self, x: List[Text]) -> None:
        self._actions = x

    @property
    def raise_ratios(self) -> List[float]:
        return self._raise_ratios

    @raise_ratios.setter
    def raise_ratios(self, x: List[float]) -> None:
        self._raise_ratios = x

    @property
    def max_raise_times(self) -> int:
        return self._max_raise_times

    @max_raise_times.setter
    def max_raise_times(self, x: int) -> None:
        self._max_raise_times = x

    @property
    def stack_size(self) -> int:
        return self._stack_size

    @stack_size.setter
    def stack_size(self, x: int) -> None:
        self._stack_size = x

    @property
    def small_blind(self) -> int:
        return self._small_blind

    @small_blind.setter
    def small_blind(self, x: int) -> None:
        self._small_blind = x

    @property
    def big_blind(self) -> int:
        return self._big_blind

    @big_blind.setter
    def big_blind(self, x: int) -> None:
        self._big_blind = x

    @property
    def random_action_prob(self) -> float:
        return self._random_action_prob

    @random_action_prob.setter
    def random_action_prob(self, x: float) -> None:
        self._random_action_prob = x

    @property
    def sample_leaf(self) -> bool:
        return self._sample_leaf

    @sample_leaf.setter
    def sample_leaf(self, x: bool) -> None:
        self._sample_leaf = x

    @property
    def subgame_params(self) -> SubgameSolvingParams:
        return self._subgame_params

    @subgame_params.setter
    def subgame_params(self, params: SubgameSolvingParams) -> None:
        self._subgame_params = params


class RlRunner:
    def __init__(
        self,
        params: Union[RecursiveSolvingParams, SubgameSolvingParams],
        net: INet,
        seed: int,
        game: Game = None
    ) -> None:
        if isinstance(params, SubgameSolvingParams) and isinstance(game, Game):
            # deprecated
            params = self._build_params(game=game, fp_params=params)
        self._game: Game = Game(
            num_hole_cards=params.num_hole_cards,
            num_deck_cards=params.num_deck_cards,
            stack_size=params.stack_size,
            small_blind=params.small_blind,
            big_blind=params.big_blind,
            max_raise_times=params.max_raise_times,
            raise_ratios=params.raise_ratios
        )
        self._subgame_params: SubgameSolvingParams = params.subgame_params
        self._random_action_prob: float = params.random_action_prob
        self._sample_leaf: bool = params.sample_leaf
        self._net: INet = net
        # current state
        self._state: PartialPublicState = None
        # buffer to the beliefs
        self._beliefs: Pair = []
        random.seed(seed)

    def _build_params(
        self,
        game: Game,
        fp_params: SubgameSolvingParams
    ) -> RecursiveSolvingParams:
        params: RecursiveSolvingParams = RecursiveSolvingParams()
        params.subgame_params = fp_params
        params.num_hole_cards = game.num_hole_cards
        params.num_deck_cards = game.num_deck_cards
        return params

    def step(self) -> None:
        self._state = self._game.get_initial_state()
        self._beliefs = [
            np.ones(shape=(self._game.num_hands, )) * 1 / self._game.num_hands
            for _ in range(self._game.num_players)
        ]
        thread = threading.current_thread()
        # print("state: %s" % self._game.state_to_string(self._state))
        while not self._game.is_terminal(self._state):
            logging.debug(
                msg="[thread_loop, data_gen] tid: %d, name: %s, state: %s"
                % (thread.ident, thread.name, self._game.state_to_string(self._state))
            )
            solver: ISubgameSolver = build_solver(
                game=self._game,
                params=self._subgame_params,
                root=self._state,
                beliefs=self._beliefs,
                net=self._net
            )
            act_iters = random.randint(0, self._subgame_params.num_iters)
            for iter_ in range(act_iters):
                solver.step(traverser=iter_ % self._game.num_players)
            # sample a new state to explore
            self.sample_state(solver=solver)
            for iter_ in range(act_iters, self._subgame_params.num_iters):
                solver.step(traverser=iter_ % self._game.num_players)
            solver.update_value_network()

    def sample_state(self, solver: ISubgameSolver) -> None:
        """sampling new state from the solver and update beliefs
        """
        if self._sample_leaf:
            self.sample_state_to_leaf(solver=solver)
        else:
            self.sample_state_single(solver=solver)

    def sample_state_single(self, solver: ISubgameSolver) -> None:
        """sampling new state from the solver and update beliefs
        """
        root_id = 0
        br_sampler = random.choice(range(self._game.num_players))
        eps = random.random()
        feasible_actions = self._game.get_feasible_actions(
            state=self._state
        )
        feasible_actions_ = [
            action for action, is_feasible in enumerate(feasible_actions)
            if is_feasible
        ]
        # explore
        if (
            (self._state.player_id == br_sampler)
            and (eps < self._random_action_prob)
        ):
            action = random.choice(feasible_actions_)
        # exploit
        else:
            beliefs = self._beliefs[self._state.player_id]
            hand = sampling(probs=beliefs)
            policy = solver.get_sampling_strategy()[root_id, hand]
            action = sampling(unnormed_probs=policy)
            assert feasible_actions[action]
        # update beliefs
        # policy[hand, action] := P(action | hand)
        policy = solver.get_belief_propogation_strategy()[root_id]
        # P^{t+1}(hand | action) \propto P^t(action | hand) P^t(hand)
        # assuming that the policy has zeros outside of the range
        self._beliefs[self._state.player_id] *= policy[:, action]
        _normalize_beliefs_inplace(
            p_beliefs=self._beliefs[self._state.player_id]
        )
        self._state = self._game.act(self._state, action)

    def sample_state_to_leaf(self, solver: ISubgameSolver) -> None:
        """sampling new state from the solver and update beliefs
        """
        tree = solver.get_tree()
        # list of (node_id, action) pairs
        path: List[Tuple[int, Action]] = []
        node_id = 0
        br_sampler = random.choice(range(self._game.num_players))
        strategy = solver.get_sampling_strategy()
        sampling_beliefs = deepcopy(self._beliefs)
        while tree[node_id].num_children:
            eps = random.random()
            state = tree[node_id].state
            feasible_actions = self._game.get_feasible_actions(
                state=state
            )
            feasible_actions_ = [
                action for action, is_feasible in enumerate(feasible_actions)
                if is_feasible
            ]
            # explore
            if (
                (state.player_id == br_sampler)
                and (eps < self._random_action_prob)
            ):
                action = random.choice(feasible_actions_)
            # exploit
            else:
                beliefs = sampling_beliefs[state.player_id]
                hand = sampling(unnormed_probs=beliefs)
                policy = strategy[node_id, hand]
                action = sampling(unnormed_probs=policy)
                assert feasible_actions[action]
            # update beliefs
            # policy[hand, action] := P(action | hand)
            policy = strategy[node_id]
            # P^{t+1}(hand | action) \propto P^t(action | hand) P^t(hand)
            # assuming that the policy has zeros outside of the range
            sampling_beliefs[state.player_id] *= policy[:, action]
            _normalize_beliefs_inplace(
                p_beliefs=sampling_beliefs[state.player_id]
            )
            path.append((node_id, action))
            node_id = tree[node_id].children_begin \
                + feasible_actions_.index(action)
        # do another pass over the path to compute beliefs accroding to
        # `get_belief_propogation_strategy` that could differ from the sampling
        # strategy
        for node_id, action in path:
            feasible_actions = self._game.get_feasible_actions(
                state=self._state
            )
            feasible_actions_ = [
                action for action, is_feasible in enumerate(feasible_actions)
                if is_feasible
            ]
            policy = solver.get_belief_propogation_strategy()[node_id]
            self._beliefs[self._state.player_id] *= policy[:, action]
            _normalize_beliefs_inplace(
                p_beliefs=self._beliefs[self._state.player_id]
            )
            child_node_id = tree[node_id].children_begin \
                + feasible_actions_.index(action)
            self._state = tree[child_node_id].state


def _normalize_beliefs_inplace(p_beliefs: List[float]) -> None:
    p_beliefs[...] = normalize_probabilities_safe(
        unnormed_probs=p_beliefs, eps=REACH_SMOOTHING_EPS
    )


def _compute_strategy_recursive(
    game: Game,
    tree: Tree,
    node_id: int,
    beliefs: Pair,
    solver_builder: SubgameSolverBuilder,
    p_strategy: TreeStrategy
) -> None:
    """compute strategies for this node and all children
    """
    node = tree[node_id]
    state = node.state
    if game.is_terminal(state):
        return
    solver: ISubgameSolver = solver_builder(game, node_id, state, beliefs)
    solver.multistep()
    p_strategy[node_id] = solver.get_strategy()[0]
    feasible_actions = game.get_feasible_actions(state=state)
    child_node_id = node.children_begin
    for action, is_feasible in enumerate(feasible_actions):
        if is_feasible:
            assert child_node_id < node.children_end
            new_beliefs = deepcopy(beliefs)
            # update beliefs
            # P^{t+1}(hand|action) \propto P^t(action|hand) P^t(hand)
            # assuming that the policy has zeros outside of the range
            new_beliefs[state.player_id] *= p_strategy[node_id, :, action]
            _normalize_beliefs_inplace(p_beliefs=new_beliefs)
            _compute_strategy_recursive(
                game=game,
                tree=tree,
                node_id=child_node_id,
                beliefs=new_beliefs,
                solver_builder=solver_builder,
                p_strategy=p_strategy
            )
            child_node_id += 1


def _compute_strategy_recursive_to_leaf(
    game: Game,
    tree: Tree,
    node_id: int,
    beliefs: Pair,
    solver_builder: SubgameSolverBuilder,
    use_samplig_strategy: bool,
    p_strategy: TreeStrategy
) -> None:
    node = tree[node_id]
    state = node.state
    if game.is_terminal(state):
        return
    solver: ISubgameSolver = solver_builder(game, node_id, state, beliefs)
    solver.multistep()
    # tree traversal queue storing tuples:
    # (full_node_id, partial_node_id, unnormalized beliefs at the node)
    # do BFS traversal, for each node:
    # - copy the policy from the partial (solver) tree to strategy
    # - add children to the queue with propoer believes
    # - for non-termial leaves of the solver tree, do a recursive call
    traversal_queue = deque()
    traversal_queue.append((node_id, 0, beliefs))
    partial_strategy: TreeStrategy = \
        solver.get_sampling_strategy() if use_samplig_strategy \
        else solver.get_strategy()
    partial_belief_strategy: TreeStrategy = \
        solver.get_belief_propogation_strategy() if use_samplig_strategy \
        else solver.get_strategy()
    partial_tree = solver.get_tree()
    while len(traversal_queue) > 0:
        full_node_id, partial_node_id, node_reaches = traversal_queue.popleft()
        p_strategy[full_node_id] = partial_strategy[partial_node_id]
        full_node = tree[full_node_id]
        partial_node = partial_tree[partial_node_id]
        assert (partial_node.num_children == 0) \
            or (partial_node.num_children == full_node.num_children)
        assert partial_node.state == full_node.state
        feasible_actions = game.get_feasible_actions(state=full_node.state)
        offset = 0
        for action, is_feasible in enumerate(feasible_actions):
            if is_feasible:
                child_reaches = deepcopy(node_reaches)
                pid = full_node.state.player_id
                child_reaches[pid] *= \
                    partial_belief_strategy[partial_node_id, :, action]
                traversal_queue.append((
                    full_node.children_begin + offset,
                    partial_node.children_begin + offset,
                    child_reaches
                ))
        if (partial_node.num_children == 0) and (full_node.num_children != 0):
            _normalize_beliefs_inplace(p_beliefs=node_reaches[game.PLAYER1])
            _normalize_beliefs_inplace(p_beliefs=node_reaches[game.PLAYER2])
            _compute_strategy_recursive_to_leaf(
                game=game,
                tree=tree,
                node_id=full_node_id,
                beliefs=node_reaches,
                solver_builder=solver_builder,
                use_samplig_strategy=use_samplig_strategy,
                p_strategy=p_strategy
            )


def _compute_strategy_with_solver(
    game: Game, solver_builder: SubgameSolverBuilder
) -> TreeStrategy:
    tree: Tree = unroll_tree(game=game)
    strategy: TreeStrategy = init_nd(
        shape=(len(tree), game.num_hands, game.num_actions),
        value=0
    )
    beliefs: Pair = get_initial_beliefs(game=game)
    _compute_strategy_recursive(
        game=game,
        tree=tree,
        node_id=0,
        beliefs=beliefs,
        solver_builder=solver_builder,
        p_strategy=strategy
    )
    return strategy


def _compute_strategy_with_solver_to_leaf(
    game: Game,
    solver_builder: SubgameSolverBuilder,
    use_samplig_strategy: bool = False
):
    tree: Tree = unroll_tree(game=game)
    strategy: TreeStrategy = init_nd(
        shape=(len(tree), game.num_hands, game.num_actions),
        value=0
    )
    beliefs: Pair = get_initial_beliefs(game=game)
    _compute_strategy_recursive_to_leaf(
        game=game,
        tree=tree,
        node_id=0,
        beliefs=beliefs,
        solver_builder=solver_builder,
        use_samplig_strategy=use_samplig_strategy,
        p_strategy=strategy
    )
    return strategy


def compute_strategy_recursive(
    game: Game, subgame_params: SubgameSolvingParams, net: INet
) -> TreeStrategy:
    """compute strategy by recursively solving subgames
    use only the strategy at root of the game for the full tree,
    and proceed to its children
    """
    def solver_builder(
        game: Game, node_id: int, state: PartialPublicState, beliefs: Pair
    ) -> ISubgameSolver:
        return build_solver(
            game=game,
            params=subgame_params,
            root=state,
            beliefs=beliefs,
            net=net
        )
    return _compute_strategy_with_solver(
        game=game, solver_builder=solver_builder
    )


def compute_strategy_recursive_to_leaf(
    game: Game, subgame_params: SubgameSolvingParams, net: INet
) -> TreeStrategy:
    """compute strategy by recursively solving subgames
    use strategy for all non-leaf subgame nodes as for full game strategy
    and proceed with leaf nodes in the subgame
    """
    def solver_builder(
        game: Game, node_id: int, state: PartialPublicState, beliefs: Pair
    ) -> ISubgameSolver:
        return build_solver(
            game=game,
            params=subgame_params,
            root=state,
            beliefs=beliefs,
            net=net
        )
    return _compute_strategy_with_solver_to_leaf(
        game=game, solver_builder=solver_builder
    )


def compute_sampled_strategy_recursive_to_leaf(
    game: Game,
    subgame_params: SubgameSolvingParams,
    net: INet,
    seed: int,
    root_only: bool = False
) -> TreeStrategy:
    """compute strategy by recursively solving subgames in way that mimics
    training:
    1. sample random iteration with linear weigting
    2. copy the sampling strategy for the solver to the full game strategy
    3. compute beliefs in leaves using belief_propogation_strategy start
    recursively
    """
    random.seed(seed)
    # emulate linear weigting: choose only even iterations
    iteration_weights: List[float] = []
    for i in range(subgame_params.num_iters):
        iteration_weights.append(0. if i % 2 else i / 2 + 1)

    def solver_builder(
        game: Game, node_id: int, state: PartialPublicState, beliefs: Pair
    ) -> ISubgameSolver:
        act_iteration = sampling(unnormed_probs=iteration_weights)
        params = copy(subgame_params)
        params.num_iters = act_iteration
        if root_only and (node_id != 0):
            params.max_depth = int(1e5)
        return build_solver(
            game=game,
            params=params,
            root=state,
            beliefs=beliefs,
            net=net
        )
    return _compute_strategy_with_solver_to_leaf(
        game=game, solver_builder=solver_builder, use_samplig_strategy=True
    )
