import random
from game.tree import unroll_tree
from game.texas_holdem.texas_holdem_hunl import Game, PartialPublicState, \
    draw_cards


class TestTree:
    def test_unroll_tree(self):
        game = Game(num_deck_cards=52, num_hole_cards=2, stack_size=800)
        # preflop
        root = game.get_initial_state()
        tree, pseudo_leaves_indices = ut_tree(game, root, 0)
        # flop
        root = tree[random.choice(pseudo_leaves_indices)].state
        root.board_cards += draw_cards(3, root)
        tree, pseudo_leaves_indices = ut_tree(game, root, 1)
        # turn
        root = tree[random.choice(pseudo_leaves_indices)].state
        root.board_cards += draw_cards(1, root)
        tree, pseudo_leaves_indices = ut_tree(game, root, 2)
        # river
        root = tree[random.choice(pseudo_leaves_indices)].state
        root.board_cards += draw_cards(1, root)
        tree, pseudo_leaves_indices = ut_tree(game, root, 3)
        assert len(pseudo_leaves_indices) == 0


def ut_tree(game: Game, root: PartialPublicState, street: int):
    max_depth = game.get_max_depth(state=root)
    tree = unroll_tree(game=game, root=root)
    pseudo_leaves_indices_ = []
    pseudo_leaves_indices = []
    tree_max_depth = 0
    for node_id in range(len(tree)):
        node = tree[node_id]
        if tree_max_depth < node.depth:
            tree_max_depth = node.depth
        assert game.get_street(node.state) == street
        assert node.depth < max_depth
        if (not node.num_children) and (not game.is_terminal(node.state)):
            pseudo_leaves_indices.append(node_id)
        if game.is_pseudo_terminal(node.state):
            if node_id > 0:
                pseudo_leaves_indices_.append(node_id)
    assert tree_max_depth <= max_depth - 1
    assert pseudo_leaves_indices == pseudo_leaves_indices_
    return tree, pseudo_leaves_indices


def test_building_tree():
    stack_size = 800
    num_deck_cards = 52
    num_hole_cards = 2
    max_raise_times = 6
    small_blind = 1
    big_blind = 2

    game = Game(
        num_hole_cards=num_hole_cards,
        num_deck_cards=num_deck_cards,
        stack_size=stack_size,
        small_blind=small_blind,
        big_blind=big_blind,
        max_raise_times=max_raise_times
    )
    root = game.get_initial_state()

    def _exp_seq(a, n):
        return (a ** (n + 1) - 1) / (a - 1)

    def _ut(max_depth, num_nodes=None):
        tree = unroll_tree(
            game=game, root=root, max_depth=max_depth, is_subgame=True
        )
        num_nodes_ = len(tree)
        if num_nodes is not None:
            assert num_nodes == num_nodes_
        else:
            assert (num_nodes_ > _exp_seq(6, max_depth)) \
                and (num_nodes_ < _exp_seq(7, max_depth))
        max_depth_ = 0
        for node in tree:
            max_depth_ = node.depth if max_depth_ < node.depth else max_depth_
            print(node)
        assert max_depth_ == max_depth
    _ut(max_depth=1, num_nodes=9)
    _ut(max_depth=2, num_nodes=59)
    _ut(max_depth=3)
    _ut(max_depth=4)
    # full tree
    # _ut(max_depth=10000000)
