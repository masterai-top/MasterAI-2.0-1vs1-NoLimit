from game.texas_holdem.engine.card import Card
import logging
import time
from collections import Counter
from random import seed
from typing import Dict, List, Text, Tuple

from game.subgame_solving import TreeStrategy
from game.texas_holdem.texas_holdem_hunl import Action, Game, PartialPublicState

from ..conf import ALLIN, CALL, CHECK, FOLD, RAISE, card_id2str, sort_cards
from ..utils import sample_action
from ..gto_mc.mc import Simulator
from .solver import compute_strategy

RANKS = "".join([rank for rank in Card.RANK_MAP.values()])
SUITS = "".join([suit for suit in Card.SUIT_MAP.values()])

# # of plays to simulate
LOOPS = 10000
EPS = 1e-6
MC_SIM = Simulator("./robot/gto_mc/lookup_tablev3.bin")


ACT2STR = {
    -1: Game.INITIAL_ACTION,
    FOLD: Game.FOLD,
    CHECK: Game.CALL,
    CALL: Game.CALL,
    RAISE: Game.RAISE,
    ALLIN: Game.ALLIN
}


def get_game_state(req) -> Tuple[List[bool], Dict]:
    logging.debug((
        "[request msg] "
        + "uid={:d}, small_blind={:d}, big_blind={:d}, is_dealer={}, "
        + "hole_card={}, board_card={}, street={:d}, "
        + "self_bet_to={:d}, oppo_bet_to={:d}, "
        + "self_init_chips={:d}, oppo_init_chips={:d}, "
        + "self_street_bet_to={:d}, oppo_street_bet_to={:d}, "
        + "self_allow_actions={:#x}, runGameID={:d}, action_list={}, "
        + "action_num={:d}"
    ).format(
        req.uid,
        req.small_blind,
        req.big_blind,
        req.is_dealer,
        req.hole_card,
        req.board_card,
        req.street,
        req.self_bet_to,
        req.oppo_bet_to,
        req.self_init_chips,
        req.oppo_init_chips,
        req.self_street_bet_to,
        req.oppo_street_bet_to,
        req.self_allow_actions,
        req.runGameID,
        req.action_list,
        req.action_num
    ))
    uid = req.uid
    is_banker = req.is_dealer
    board_cards = req.board_card
    hole_cards = req.hole_card
    player_bet_to = req.self_bet_to
    oppo_bet_to = req.oppo_bet_to
    player_init_chips = req.self_init_chips
    oppo_init_chips = req.oppo_init_chips
    player_allow_actions = req.self_allow_actions
    small_blind = req.small_blind
    big_blind = req.big_blind
    hole_cards = sort_cards(hole_cards)
    board_cards = sort_cards(board_cards)
    hole_cards = [card_id2str(card) for card in hole_cards]
    board_cards = [card_id2str(card) for card in board_cards]
    stack_size = min(player_init_chips, oppo_init_chips)
    game_state = {"uid": uid}
    game_state["game"] = Game(
        stack_size=stack_size,
        small_blind=small_blind,
        big_blind=big_blind
    )
    is_first_action, raise_times, last_action = parse_action_seq(
        req.action_list, req.street, game_state["game"]
    )
    game_state["pocket"] = hole_cards
    game_state["state"] = PartialPublicState(
        last_action=last_action,
        board_cards=[Card.str2card(card) for card in board_cards],
        player_bet_to=player_bet_to,
        oppo_bet_to=oppo_bet_to,
        player_id=Game.PLAYER1 if is_banker else Game.PLAYER2,
        is_first_action=is_first_action,
        is_terimnal=False,
        raise_times=raise_times
    )
    actions = valid_actions(
        game=game_state["game"],
        state=game_state["state"],
        player_allow_actions=player_allow_actions
    )
    return actions, game_state


def valid_actions(
    game: Game, state: PartialPublicState, player_allow_actions: int
) -> List[bool]:
    call_bet = state.oppo_bet_to - state.player_bet_to
    try:
        assert call_bet >= 0
    except:
        logging.fatal(
            "fatal: oppo_bet_to (%d) cann't be smaller than player_bet_to (%d)"
            % (state.oppo_bet_to, state.player_bet_to)
        )
    feasible_actions = game.get_feasible_actions(state=state)
    actions = []
    for action, is_feasible in enumerate(feasible_actions):
        if is_feasible:
            actions.append(action)
    min_bet = call_bet + max(call_bet, game.big_blind)
    allin_bet = game.stack_size - state.player_bet_to
    actions = {}
    # actions allowed
    if (player_allow_actions & FOLD) and (not player_allow_actions & CHECK):
        actions["fold"] = True
    else:
        actions["fold"] = False
    if (player_allow_actions & CHECK) or (player_allow_actions & CALL):
        actions["check&call"] = True
    else:
        actions["check&call"] = False
    if (player_allow_actions & RAISE) and (min_bet < allin_bet):
        actions["raise"] = True
    else:
        actions["raise"] = False
    if player_allow_actions & ALLIN:
        actions["allin"] = True
    else:
        actions["allin"] = False
    return actions


def parse_action_seq(
    action_seq: Dict[int, List[int]], street: int, game: Game
) -> Tuple[bool, int, Action]:
    action_seq_ = action_seq[street].action
    is_first_action = len(action_seq_) == 0
    if street > 0:
        # flop, turn, river
        if is_first_action:
            last_action = action_seq[street - 1].action[-1]
        else:
            last_action = action_seq_[-1]
    else:
        # pre-flop
        if is_first_action:
            last_action = -1
        else:
            last_action = action_seq_[-1]
    raise_times = Counter(action_seq_).get(RAISE)
    raise_times = 0 if raise_times is None else raise_times

    def req2game_action_id(action):
        action = ACT2STR[action]
        if action != game.RAISE:
            action = game.action_str2id[action]
        else:
            action = 2
        return action
    last_action = req2game_action_id(last_action)
    return is_first_action, raise_times, last_action


def do_action(actions: List[bool], game_state: Dict) -> Tuple[int, int]:
    """given the status of a game, returns the best action"""
    seed(time.time())
    allowed_actions = [k for k in actions if actions[k] is not False]
    uid = game_state["uid"]
    hand = game_state["pocket"]
    game: Game = game_state["game"]
    state: PartialPublicState = game_state["state"]
    hand_id = game.get_hand_index(hand=[Card.str2card(card) for card in hand])
    logging.info(
        "req: uid={}, state=[{}], game=[{}], pocket={}, allowed actions={}"
        .format(uid, state, game, hand, allowed_actions)
    )
    # -----------------
    # tolerating faults
    # -----------------
    # 1. player has no more money
    if game.stack_size <= state.player_bet_to:
        logging.warn(
            "player has no more money, stack_size=%d, player_bet_to=%d"
            % (game.stack_size, state.player_bet_to)
        )
        return CHECK, 0
    assert game.stack_size > state.player_bet_to, \
        "fatal: player has no more money"
    # 2. sometimes, oppo_bet_to == stack_size, however, last_action is not allin
    if game.stack_size == state.oppo_bet_to:
        if state.last_action != game.action_str2id["allin"]:
            logging.warn(
                "opponent has allin, however, last_action=%s"
                % (game.action_id2str[state.last_action])
            )
            state.last_action = game.action_str2id["allin"]
        assert state.last_action == game.action_str2id["allin"]
    # -----------------
    # call cfr decision
    # -----------------
    strategy: TreeStrategy = compute_strategy(
        game=game, state=state
    )
    action_probs = strategy[hand_id]
    logging.debug("rsp: uid={}, strategy={}".format(uid, action_probs))
    # only for debug
    # import numpy as np
    # action_probs = np.ones(9)
    action_probs[action_probs < 0] = 0
    for action, is_allowed in actions.items():
        if not is_allowed:
            if action == "fold":
                action_probs[0] = 0
            elif action == "check&call":
                action_probs[1] = 0
            elif action == "raise":
                action_probs[2: -1] = 0
            elif action == "allin":
                action_probs[-1] = 0
            else:
                pass
    # ------------------
    # sampling an action
    # ------------------
    call_size = state.oppo_bet_to - state.player_bet_to  # call size
    # in the case that there are no valid actions,
    # choose either check or fold&call depending on whether check is feasible
    if action_probs.sum() < EPS:
        if call_size == 0:
            action_probs[1] = 1
        else:
            action_probs[0: 2] = 0.5
    # normalization
    action_probs /= action_probs.sum()
    action_id = sample_action(probs=action_probs)
    action: Text = game.actions[action_id]
    if action == "fold":
        action = FOLD
        action_size = 0
    elif action == "check&call":
        if call_size > 0:
            action = CALL
        else:
            action = CHECK
        action_size = call_size
    elif action == "allin":
        action = ALLIN
        action_size = game.stack_size - state.oppo_bet_to
    else:
        _, ratio = action.split("_")
        action = RAISE
        ratio = float(ratio)
        action_size = int(state.pot_size * ratio + 0.5) + call_size
    # ---------------------
    # mc for the last guard
    # ---------------------
    odds = calculate_odds(
        hand, game_state["state"].board_cards, game.num_players
    )
    if len(game_state["state"].board_cards) == 0:
        if odds < 0.55:
            action = CHECK if call_size == 0 else FOLD
            action_size = 0
    if len(game_state["state"].board_cards) == 3:
        if odds < 0.65:
            action = CHECK if call_size == 0 else FOLD
            action_size = 0
        elif (odds > 0.90) and (action == FOLD):
            action = CHECK if call_size == 0 else CALL
            action_size = call_size
    if len(game_state["state"].board_cards) == 4:
        if odds < 0.75:
            action = CHECK if call_size == 0 else FOLD
            action_size = 0
        elif (odds > 0.90) and (action == FOLD):
            action = CHECK if call_size == 0 else CALL
            action_size = call_size
    if len(game_state["state"].board_cards) == 5:
        if odds < 0.80:
            action = CHECK if call_size == 0 else FOLD
            action_size = 0
        elif (odds > 0.90) and (action == FOLD):
            action = CHECK if call_size == 0 else CALL
            action_size = call_size

    if call_size == 0 and action == FOLD:
        action = CHECK
        action_size = 0

    logging.info(
        "rsp: uid={}, action={}, bet_size={}".format(uid, action, action_size)
    )
    return action, action_size


def calculate_odds(hole_cards, board_cards, num_players):
    """given a hand and the community cards, returns the strenght of the hand
    """
    t_start = time.time()
    suit_map = {"♦": "d", "♣": "c", "♥": "h", "♠": "s"}
    rank_map = {k: k for k in RANKS}
    rank_map["T"] = "10"
    hole_cards_ = [[rank_map[x[0]] + suit_map[x[1]] for x in hole_cards]]
    board_cards_ = [str(x) for x in board_cards]
    board_cards_ = [rank_map[x[0]] + suit_map[x[1]] for x in board_cards_]
    num_wins, _ = MC_SIM.compute_probabilities(
        LOOPS, board_cards_, hole_cards_, num_players - 1
    )
    t_elapsed = time.time() - t_start
    logging.info("message: mc take %.2fs, odds=%.2f%%" % (t_elapsed, num_wins))
    return num_wins / 100
