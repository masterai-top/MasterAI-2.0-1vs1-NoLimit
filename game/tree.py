import logging
import torch.multiprocessing as mp
from typing import Iterator, List, Text, Tuple
from game.texas_holdem.texas_holdem_hunl import Action, Game, PartialPublicState


class UnrolledTreeNode:
    """the nodes are expected to be stored in a vector, with children_begin,
    children_end, and parent being indices in the vector.
    """

    def __init__(
        self,
        state: PartialPublicState,
        children_begin: int = 0,
        children_end: int = 0,
        parent_id: int = -1,
        depth: int = 2
    ) -> None:
        self._state = state
        self._children_begin = children_begin
        self._children_end = children_end
        self._parent_id = parent_id
        self._depth = depth

    @property
    def num_children(self) -> int:
        return self._children_end - self._children_begin

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def state(self) -> PartialPublicState:
        return self._state

    @property
    def parent_id(self) -> int:
        return self._parent_id

    @property
    def children_begin(self) -> int:
        return self._children_begin

    @children_begin.setter
    def children_begin(self, offset: int):
        self._children_begin = offset

    @property
    def children_end(self) -> int:
        return self._children_end

    @children_end.setter
    def children_end(self, offset: int) -> None:
        self._children_end = offset

    def get_children(self) -> List[int]:
        children = [self.children_begin + i for i in range(self.num_children)]
        return children

    def __str__(self) -> Text:
        ret = ""
        ret += "parent_id: %d, " % self._parent_id
        ret += "num_children: %d, " % (
            self._children_end - self._children_begin
        )
        ret += "depth: %d, " % self._depth
        ret += "state info: [%s]" % str(self._state)
        return ret


class ChildrenActionIt:
    """creates iterator over children nodes and corresponding actions
    usage:
        for (child_id, action) in ChildrenActionIt(node, game)
        {
            do stuff
        }
    """

    def __init__(self, node: UnrolledTreeNode, game: Game) -> None:
        self._game = game
        self._feasible_actions = game.get_feasible_actions(node.state)
        self._action = 0
        self._child = node.children_begin

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Tuple[int, Action]:
        if self._action < self._game.num_actions:
            is_feasible = self._feasible_actions[self._action]
            while not is_feasible:
                self._action += 1
                is_feasible = self._feasible_actions[self._action]
            action = self._action
            child = self._child
            self._action += 1
            self._child += 1
            return child, action
        else:
            raise StopIteration


class ChildrenIt:
    """ creates iterator over children nodes and corresponding actions
    usage:
        for child_id in ChildrenIt(node)
        {
            do stuff
        }
    """

    def __init__(self, node: UnrolledTreeNode) -> None:
        self._node = node
        self._index = 0

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> int:
        if self._index < self._node.num_children:
            child = self._node.children_begin + self._index
            self._index += 1
            return child
        else:
            raise StopIteration


Tree = List[UnrolledTreeNode]


def unroll_tree(
    game: Game,
    root: PartialPublicState = None,
    max_depth: int = None,
    is_subgame: bool = False
) -> Tree:
    """builds a bfs tree of this depth as a linear table

    for max_depth = 0, the tree will contain only the root
    for max_depth = 1, the tree will contain root and its children
    and so on

    is_subgame: set to be false if max_depth > 1e5
    """
    if root is None:
        root = game.get_initial_state()
    if max_depth is None:
        max_depth = game.get_max_depth(state=root)
    if max_depth > 1e5:
        is_subgame = False
    assert max_depth >= 0, "error: can not build an empty tree"
    nodes: Tree = []
    nodes.append(
        UnrolledTreeNode(
            state=root, children_begin=0, children_end=0, parent_id=-1, depth=0
        )
    )
    node_id = 0
    while (node_id < len(nodes)) and (nodes[node_id].depth < max_depth):
        feasible_actions = game.get_feasible_actions(
            state=nodes[node_id].state
        )
        node = nodes[node_id]
        node.children_begin = len(nodes)
        node.children_end = node.children_begin
        # build a subgame
        if is_subgame and game.is_pseudo_terminal(state=node.state):
            # root is not a pseudo leaf
            if node_id > 0:
                node_id += 1
                node.state.board_cards = nodes[node.parent_id].state.board_cards
                continue
        for action, is_feasible in enumerate(feasible_actions):
            if is_feasible:
                next_state = game.act(state=node.state, action=action)
                nodes.append(UnrolledTreeNode(
                    state=next_state,
                    children_begin=0,
                    children_end=0,
                    parent_id=node_id,
                    depth=node.depth + 1
                ))
                node.children_end += 1
        node_id += 1
    proc = mp.current_process()
    logging.debug(
        msg="[tree] pid: %d, name: %s, # of nodes: %d"
        % (proc.pid, proc.name, len(nodes))
    )
    return nodes


def print_tree(tree: Tree) -> None:
    for node in tree:
        print(node)
