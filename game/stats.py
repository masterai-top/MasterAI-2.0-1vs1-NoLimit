import torch
from typing import List

from game.net import INet
from game.texas_holdem.texas_holdem_hunl import Game, PartialPublicState
from game.utils import normalize_probabilities
from .subgame_solving import Pair, SubgameSolvingParams, TreeStrategy, \
    TreeStrategyStats, build_solver, compute_strategy_stats, get_query
from .tree import ChildrenIt, Tree, unroll_tree


def compute_depths(
    tree: Tree,
    p_depths: List[int],
    index: int = 0,
    depth: int = 0
):
    if not p_depths:
        p_depths += [0] * len(tree)
    p_depths[index] = depth
    for child_id in ChildrenIt(tree[index]):
        compute_depths(
            tree=tree, p_depths=p_depths, index=child_id, depth=depth + 1
        )


def eval_net(
    game: Game,
    net_strategy: TreeStrategy,
    full_strategy: TreeStrategy,
    mdp_depth: int,
    fp_iters: int,
    net: INet,
    traverse_by_net: bool,
    verbose: bool = False
) -> float:
    full_tree: Tree = unroll_tree(game=game)
    net_stats: TreeStrategyStats = compute_strategy_stats(
        game=game, strategy=net_strategy
    )
    true_stats: TreeStrategyStats = compute_strategy_stats(
        game=game, strategy=full_strategy
    )
    if verbose:
        if traverse_by_net:
            print("using net policy to define beliefs")
        else:
            print("using fp policy to define beliefs")
    traversing_stats: TreeStrategyStats = \
        net_stats if traverse_by_net else true_stats
    node_reach: List[float] = traversing_stats.node_reach
    # get non-terminal nodes at depth mdp_depth and mdp_depth * 2
    depths: List[int] = []
    compute_depths(tree=full_tree, p_depths=depths)
    top_node_ids: List[int] = []
    for i in range(len(node_reach)):
        if (depths[i] == mdp_depth) or (depths[i] == 2 * mdp_depth):
            if not game.is_terminal(state=full_tree[i].state):
                top_node_ids.append(i)
    # sort in descending order
    top_node_ids.sort(
        key=lambda i: node_reach[i], reverse=True
    )
    MIN_REACH = 1e-6
    if verbose:
        print(
            "non-terminal nodes at depth %d : %d"
            % (mdp_depth, len(top_node_ids))
        )
    if len(top_node_ids) == 0:
        print("empty list, exiting")
        return 0.
    while node_reach[top_node_ids[-1]] < MIN_REACH:
        top_node_ids.pop()
    if verbose:
        print(
            "after filtering with reach < %f : %d"
            % (MIN_REACH, len(top_node_ids))
        )
        print("min reach: %f" % node_reach[top_node_ids[-1]])
        print("max reach: %f" % node_reach[top_node_ids[0]])
    total_true_reach = 0.
    total_net_reach = 0.
    for node_id in top_node_ids:
        total_true_reach += true_stats.node_reach[node_id]
        total_net_reach += net_stats.node_reach[node_id]
    if verbose:
        print(
            "total reach: true = %f, net = %f"
            % (total_true_reach, total_net_reach)
        )
    if len(top_node_ids) == 0:
        # that's odd
        return 0.
    mses: List[float] = []
    for node_id in top_node_ids:
        beliefs: Pair = [
            normalize_probabilities(
                traversing_stats.reach_probabilities[player_id][node_id]
            ) for player_id in range(game.num_players)
        ]
        state: PartialPublicState = full_tree[node_id].state
        params = SubgameSolvingParams()
        params.num_iters = fp_iters
        params.max_depth = int(1e5)
        params.linear_update = True
        fp = build_solver(
            game=game, params=params, root=state, beliefs=beliefs
        )
        fp.multistep()
        for traverser in range(game.num_players):
            query = torch.tensor(get_query(
                game=game,
                traverser=traverser,
                state=state,
                reaches1=beliefs[game.PLAYER1],
                reaches2=beliefs[game.PLAYER2]
            ))
            reach_tensor = torch.tensor(beliefs[traverser])
            net_value: torch.Tensor = net.compute_values(
                query=query.unsqueeze(dim=0)
            )
            net_value = (net_value.squeeze() * reach_tensor).sum().item()
            br_value: float = torch.tensor(
                fp.get_hand_values(player_id=traverser) * reach_tensor
            ).sum().item()
            blueprint_value: float = true_stats.node_values[traverser][node_id]
            if verbose:
                print((
                    "%s \tnet_reach = %f, true_reach = %f, "
                    + "net_value = %f, br_value = %f"
                ) % (
                    game.state_to_string(state=state),
                    net_stats.node_reach[node_id],
                    true_stats.node_reach[node_id],
                    net_value, br_value
                ), end="")
                if not traverse_by_net:
                    print(" blue_value = %s" % blueprint_value)
                print()
            mses.append((net_value - br_value) ** 2)
    mse: float = sum(mses) / len(mses)
    if verbose:
        print("final mse: %f" % mse)
    return mse
