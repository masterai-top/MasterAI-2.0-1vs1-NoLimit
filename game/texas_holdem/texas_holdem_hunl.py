from copy import copy
from itertools import combinations
from typing import Dict, Iterator, List, Text, Tuple
from .engine.deck import Deck
from .engine.card import Card
from .utils import n_combinations

Action = int


class UnpackedAction:
    def __init__(self, action, bet_size):
        self._action: Action = 0
        self._bet_size: int = 0


class PartialPublicState:
    def __init__(
        self,
        last_action: Action,
        board_cards: List[Card],
        player_bet_to: int,
        oppo_bet_to: int,
        player_id: int,
        is_first_action: bool,
        is_terimnal: bool,
        raise_times: int
    ):
        """
        args:

        is_first_action: if the action is the first in each street
        """
        # last action
        self._last_action = last_action
        self.board_cards = board_cards
        self._player_bet_to = player_bet_to
        self._oppo_bet_to = oppo_bet_to
        self._is_first_action = is_first_action
        # player to make decision next
        self._pid = player_id
        self._is_terminal = is_terimnal
        self._raise_times = raise_times

    def __eq__(self, state):
        return (
            self.player_bet_to == state.player_bet_to
            and self.oppo_bet_to == state.oppo_bet_to
            and self.board_cards == state.board_cards
            and self.player_id == state.player_id
            # and self.last_action == state.last_action
            and self.is_first_action == state.is_first_action
            and self.is_terminal == state.is_terminal
        )

    @property
    def pot_size(self) -> int:
        return self._player_bet_to + self._oppo_bet_to

    @property
    def player_bet_to(self) -> int:
        return self._player_bet_to

    @property
    def oppo_bet_to(self) -> int:
        return self._oppo_bet_to

    @property
    def board_cards(self) -> List[Card]:
        return self._board_cards

    @board_cards.setter
    def board_cards(self, board_cards: List[Card]):
        # def sort_(card): return card.rank
        # self._board_cards = sorted(board_cards, key=sort_, reverse=True)
        self._board_cards = board_cards

    @property
    def player_id(self) -> int:
        return self._pid

    @property
    def last_action(self) -> Action:
        return self._last_action

    @last_action.setter
    def last_action(self, action: Action) -> None:
        self._last_action = action

    @property
    def is_first_action(self) -> bool:
        return self._is_first_action

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def raise_times(self) -> int:
        return self._raise_times

    def __str__(self) -> Text:
        ret = ""
        ret += "player_id: %d, " % self._pid
        ret += "last_action: %d, " % self._last_action
        ret += "player_bet_to: %d, " % self._player_bet_to
        ret += "oppo_bet_to: %d, " % self._oppo_bet_to
        ret += "pot_size: %d, " % self.pot_size
        ret += "is_first_action: {}, ".format(self._is_first_action)
        ret += "is_terminal: {}, ".format(self._is_first_action)
        ret += "raise_times: %d, " % self._raise_times
        ret += "board_cards: ["
        for card in self._board_cards:
            ret += "%s, " % str(card)
        ret += "]"
        return ret


class Game:
    """HUNL

    player_id = 0 is always the banker (SB)
    """
    SB = 1
    BB = 2 * SB
    DEFALT_STACK_SIZE = 200 * BB
    NUM_PLAYERS = 2
    PLAYER1 = 0
    PLAYER2 = 1
    BANKER = PLAYER1

    # streets
    PRE_FLOP = 0
    FLOP = 1
    TRUN = 2
    RIVER = 3
    NUM_STREETS = 4
    NUM_HOLE_CARDS = 2
    NUM_DECK_CARDS = 52
    NUM_BOARD_CARDS_PER_STREET = [0, 3, 1, 1]
    NUM_BOARD_CARDS = sum(NUM_BOARD_CARDS_PER_STREET)
    # actions
    INITIAL_ACTION = "start"
    FOLD = "fold"
    CALL = "check&call"
    RAISE = "raise"
    # raise_size = ratio * pot_size + call_size
    RAISE_RATIOS: List[float] = [1 / 3, 1 / 2, 2 / 3, 1, 3 / 2, 2]
    RAISES: List[Text] = [
        "raise_%.2f" % ratio for ratio in RAISE_RATIOS
    ]
    MAX_RAISE_TIMES = 10
    ALLIN = "allin"
    ACTIONS: List[Text] = [FOLD, CALL] + RAISES + [ALLIN]
    ACTION_STR2ID: Dict[Text, Action] = {
        k: v for v, k in enumerate(ACTIONS)
    }
    ACTION_STR2ID[INITIAL_ACTION] = -1
    ACTION_ID2STR: Dict[Action, Text] = {
        v: k for k, v in ACTION_STR2ID.items()
    }
    ACTION_ID2STR[-1] = INITIAL_ACTION
    DECK_CARDS = Deck().cards

    def __init__(
        self,
        num_hole_cards: int = None,
        num_deck_cards: int = None,
        stack_size: int = None,
        small_blind: int = None,
        big_blind: int = None,
        max_raise_times: int = None,
        raise_ratios: List[float] = None,
        deck_cards: List[Card] = None
    ) -> None:
        self._num_hole_cards = num_hole_cards if num_hole_cards \
            else self.NUM_HOLE_CARDS
        self._num_deck_cards = num_deck_cards if num_deck_cards \
            else self.NUM_DECK_CARDS
        self._num_hands = n_combinations(
            self._num_deck_cards, self._num_hole_cards
        )
        self._stack_size = stack_size if stack_size else self.DEFALT_STACK_SIZE
        self._small_blind = small_blind if small_blind else self.SB
        self._big_blind = big_blind if big_blind else self.BB
        self._max_raise_times = max_raise_times if max_raise_times else self.MAX_RAISE_TIMES
        self._config_actions(raise_ratios=raise_ratios)
        self._deck_cards = deck_cards if deck_cards else self.DECK_CARDS

    def _config_actions(self, raise_ratios: List[float] = None) -> None:
        if raise_ratios is None:
            self._actions = self.ACTIONS
            self._action_id2str = self.ACTION_ID2STR
            self._action_str2id = self.ACTION_STR2ID
            return
        raises = ["raise_%.2f" % ratio for ratio in raise_ratios]
        self._actions = [self.FOLD, self.CALL] + raises + [self.ALLIN]
        self._action_str2id: Dict[Text, Action] = {
            k: v for v, k in enumerate(self._actions)
        }
        self._action_str2id[self.INITIAL_ACTION] = -1
        self._action_id2str: Dict[Action, Text] = {
            v: k for k, v in self._action_str2id.items()
        }
        self._action_id2str[-1] = self.INITIAL_ACTION

    def act(
        self, state: PartialPublicState, action: Action
    ) -> PartialPublicState:
        assert not state.is_terminal, (
            "fatal: no action are feasible at terminal nodes"
        )
        assert action in self._action_id2str, (
            "fatal: unknown action id (%d)" % action
        )
        action = self._action_id2str[action]
        allin_size = self.stack_size - state.player_bet_to
        assert allin_size > 0, (
            "fatal: the player has already allin and no more action are feasible"
        )
        last_action = self._action_id2str[state.last_action]
        # allin, for allin is always feasible
        if action == self.ALLIN:
            next_state = self._allin_child(state=state)
        # fold
        elif action == self.FOLD:
            assert last_action != self.CALL, (
                "warning: if the opponent has acted check & call, fold is strictly dominated by check"
            )
            next_state = self._fold_child(state=state)
        # check & call, raise
        else:
            call_size = state.oppo_bet_to - state.player_bet_to
            assert call_size >= 0, (
                "fatal: oppo_bet_to (%d) must be greater than or equal to "
                + "player_bet_to (%d)"
                % (state.oppo_bet_to, state.player_bet_to)
            )
            assert call_size < allin_size, (
                "fatal: the player's left chips (%d) are less than "
                + "call_size (%d), only allin is feasible"
                % (allin_size, call_size)
            )

            # check & call
            if action == self.CALL:
                next_state = self._call_child(state=state)
            # raise
            else:
                idx = self.RAISES.index(action)
                bet_size = call_size + int(
                    (state.pot_size + call_size) * self.RAISE_RATIOS[idx] + 0.5
                )
                assert bet_size < allin_size, (
                    "fatal: raise action (%s) is not feasible for "
                    + "raise_size (%d) is greater than or equal to "
                    + "allin_size (%d)" % (action, bet_size, allin_size)
                )
                next_state = self._raise_child(
                    state=state, bet_size=bet_size, action=action
                )
        return next_state

    def _call_child(self, state: PartialPublicState) -> PartialPublicState:
        last_action = self._action_id2str[state.last_action]
        next_bet_to = state.oppo_bet_to
        next_player_id = 1 - state.player_id
        is_first_action = True
        # if the opponent has allin and the player acts call, the child is
        # terminal and board cards are drawn to five
        if last_action == self.ALLIN:
            is_terimnal = True
            board_cards = copy(state.board_cards)
            num_drawn_cards = self.NUM_BOARD_CARDS - len(board_cards)
            board_cards += draw_cards(
                num_drawn_cards=num_drawn_cards, state=state
            )
        else:
            is_terimnal = False
            board_cards = state.board_cards
            # the first action in each street
            if state.is_first_action:
                is_first_action = False
            # go to the next street
            else:
                street = self.get_street(state=state) + 1
                if street <= self.RIVER:
                    num_drawn_cards = self.NUM_BOARD_CARDS_PER_STREET[street]
                    board_cards = copy(state.board_cards)
                    board_cards += draw_cards(
                        num_drawn_cards=num_drawn_cards, state=state
                    )
                    # the banker always acts at last except pre-flop
                    next_player_id = 1 - self.BANKER
                else:
                    is_terimnal = True
        return PartialPublicState(
            board_cards=board_cards,
            last_action=self._action_str2id[self.CALL],
            player_bet_to=next_bet_to,
            oppo_bet_to=next_bet_to,
            player_id=next_player_id,
            is_first_action=is_first_action,
            is_terimnal=is_terimnal,
            raise_times=0
        )

    def _fold_child(self, state: PartialPublicState) -> PartialPublicState:
        return PartialPublicState(
            board_cards=state.board_cards,
            last_action=self._action_str2id[self.FOLD],
            player_bet_to=state.oppo_bet_to,
            oppo_bet_to=state.player_bet_to,
            player_id=1 - state.player_id,
            is_first_action=False,
            is_terimnal=True,
            raise_times=0
        )

    def _raise_child(
        self, state: PartialPublicState, bet_size: int, action: Text
    ) -> PartialPublicState:
        last_action = self._action_id2str[state.last_action]
        if last_action == self.ALLIN:
            is_terimnal = True
        else:
            is_terimnal = False
        return PartialPublicState(
            board_cards=state.board_cards,
            last_action=self._action_str2id[action],
            player_bet_to=state.oppo_bet_to,
            oppo_bet_to=bet_size + state.player_bet_to,
            player_id=1 - state.player_id,
            is_first_action=False,
            is_terimnal=is_terimnal,
            raise_times=state.raise_times + 1
        )

    def _allin_child(self, state: PartialPublicState) -> PartialPublicState:
        last_action = self._action_id2str[state.last_action]
        board_cards = copy(state.board_cards)
        if last_action == self.ALLIN:
            # go to the river street directly, it is a terminal node
            is_terimnal = True
            num_drawn_cards = sum(self.NUM_BOARD_CARDS_PER_STREET) \
                - len(state.board_cards)
            board_cards += draw_cards(
                num_drawn_cards=num_drawn_cards, state=state
            )
        else:
            is_terimnal = False
        return PartialPublicState(
            board_cards=board_cards,
            last_action=self._action_str2id[self.ALLIN],
            player_bet_to=state.oppo_bet_to,
            oppo_bet_to=self.stack_size,
            player_id=1 - state.player_id,
            is_first_action=False,
            is_terimnal=is_terimnal,
            raise_times=0
        )

    def get_feasible_actions(self, state: PartialPublicState) -> List[bool]:
        # terminal node
        if state.is_terminal:
            return [False] * len(self.actions)
        call_size = state.oppo_bet_to - state.player_bet_to
        min_raise_size = max(self._big_blind, call_size) + call_size
        allin_size = self.stack_size - state.player_bet_to
        feasible_acitons = [True] * len(self.actions)
        # fold
        if call_size == 0:
            # when check is feasible, fold is strictly dominated
            feasible_acitons[self._action_str2id[self.FOLD]] = False
        # check & call
        if call_size >= allin_size:
            # check & call is always feasible unless player's left chips are
            # less than or equal to call_size
            feasible_acitons[self._action_str2id[self.CALL]] = False
        # raise
        if state.raise_times >= self._max_raise_times:
            # all players can raise up to max_raise_times times in total
            for r in self.RAISES:
                feasible_acitons[self._action_str2id[r]] = False
        elif min_raise_size >= allin_size:
            # the player's chips can not afford a raise
            for r in self.RAISES:
                feasible_acitons[self._action_str2id[r]] = False
        else:
            for i, ratio in enumerate(self.RAISE_RATIOS):
                raise_size = call_size + int(
                    ratio * (state.pot_size + call_size) + 0.5
                )
                if min_raise_size <= raise_size < allin_size:
                    continue
                r = self.RAISES[i]
                feasible_acitons[self._action_str2id[r]] = False
        # allin
        if False:
            # allin is always feasible
            feasible_acitons[self._action_str2id[self.ALLIN]] = False
        return feasible_acitons

    def get_initial_state(self) -> PartialPublicState:
        return PartialPublicState(
            last_action=-1,
            board_cards=[],
            player_bet_to=self._small_blind,
            oppo_bet_to=self._big_blind,
            is_first_action=True,
            player_id=self.BANKER,
            is_terimnal=False,
            raise_times=0
        )

    def unpack_action(self, action: Action, pot_size: int) -> UnpackedAction:
        raise NotImplementedError
        unpacked_action = UnpackedAction(action, bet_size)
        return unpacked_action

    @staticmethod
    def is_terminal(state: PartialPublicState) -> bool:
        return state.is_terminal

    @staticmethod
    def is_pseudo_terminal(state: PartialPublicState) -> bool:
        """a subgame starts at the beginning of one street and ends at the start
        of the next street
        """
        return state.is_first_action and not state.is_terminal

    def state_to_string(self, state: PartialPublicState) -> Text:
        # last_action = "start" if self.INITIAL_ACTION == state.last_action \
        #     else self.action_to_string(action=state.last_action)
        last_action = self.action_to_string(action=state.last_action)
        return "(pid = %d, pot_size = %d, last_action = %s)" % (
            state.player_id, state.pot_size, last_action
        )

    def action_to_string_short(self, state: PartialPublicState) -> Text:
        last_action = "start" if self.INITIAL_ACTION == state.last_action \
            else self.action_to_string_short(action=state.last_action)
        raise NotImplementedError

    def action_to_string(self, action: Action) -> Text:
        return self._action_id2str[action]

    def action_to_string_short(self, action: Action) -> Text:
        raise NotImplementedError

    @classmethod
    def get_street(cls, state: PartialPublicState) -> int:
        num_board_cards = len(state.board_cards)
        if num_board_cards == 0:
            return cls.PRE_FLOP
        elif num_board_cards == 3:
            return cls.FLOP
        elif num_board_cards == 4:
            return cls.TRUN
        elif num_board_cards == 5:
            return cls.RIVER
        else:
            raise ValueError("fatal: # of board cards is %d" % num_board_cards)

    def get_max_depth(self, state: PartialPublicState) -> int:
        depth = 0
        player_bet_to = state.player_bet_to
        oppo_bet_to = state.oppo_bet_to
        pot_size = player_bet_to + oppo_bet_to
        call_size = oppo_bet_to - player_bet_to
        while player_bet_to < self.stack_size:
            depth += 1
            player_bet_to += call_size + max(
                int((pot_size + call_size) * min(self.RAISE_RATIOS) + 0.5),
                self._big_blind
            )
            oppo_bet_to, player_bet_to = player_bet_to, oppo_bet_to
            pot_size = player_bet_to + oppo_bet_to
            call_size = oppo_bet_to - player_bet_to
        n_board_cards = len(state.board_cards)
        if n_board_cards == 0:
            depth += 2 * 4
        elif n_board_cards == 3:
            depth += 2 * 3
        elif n_board_cards == 4:
            depth += 2 * 2
        elif n_board_cards == 5:
            depth += 2
        else:
            raise ValueError("fatal: # of board cards is %d" % n_board_cards)
        return depth

    def get_hand_index(self, hand: Tuple[Card, Card]) -> int:
        hand = sorted(hand, key=lambda card: card.id)
        return self.hands.index(hand)

    @property
    def actions(self) -> List[Text]:
        return self._actions

    @property
    def action_str2id(self) -> Dict[Text, int]:
        return self._action_str2id

    @property
    def action_id2str(self) -> Dict[int, Text]:
        return self._action_id2str

    @property
    def num_actions(self) -> int:
        return len(self.actions)

    @property
    def num_deck_cards(self) -> int:
        return self._num_deck_cards

    @property
    def num_hole_cards(self) -> int:
        return self._num_hole_cards

    @property
    def num_hands(self) -> int:
        return self._num_hands

    @property
    def num_players(self) -> int:
        return self.NUM_PLAYERS

    @property
    def hands(self) -> Iterator[Tuple[Card, Card]]:
        combs = combinations(self._deck_cards, self._num_hole_cards)
        return [sorted(list(item), key=lambda card: card.id) for item in combs]

    @property
    def stack_size(self) -> int:
        return self._stack_size

    @property
    def small_blind(self) -> int:
        return self._small_blind

    @property
    def big_blind(self) -> int:
        return self._big_blind

    def __str__(self) -> Text:
        ret = ""
        ret += "stack_size: %d, " % self.stack_size
        ret += "small_blind: %d, " % self.small_blind
        ret += "big_blind: %d, " % self.big_blind
        ret += "max_raise_times: %d" % self._max_raise_times
        return ret


def draw_cards(
    num_drawn_cards: int, state: PartialPublicState
) -> List[int]:
    deck = Deck(drawn_cards=state.board_cards)
    deck.shuffle()
    return deck.draw_cards(num_drawn_cards)
