import logging
import time
from random import random, seed, shuffle
from game.texas_holdem.engine.card import Card
from game.texas_holdem.engine.hand_eval import HandEvaluator
from .mc import Simulator
# from .poker import poker
from ..conf import ALLIN, CALL, CHECK, FOLD, RAISE, card_id2str, sort_cards

RANKS = "".join([rank for rank in Card.RANK_MAP.values()])
SUITS = "".join([suit for suit in Card.SUIT_MAP.values()])

# # of plays to simulate
LOOPS = 10000
EPS = 1e-3
EV_THRESHOLD = 0.1
PRE_FLOP_ODDS = [EPS, 0.2, 0.5, 0.7, 0.85]
MC_SIM = Simulator("./robot/gto_mc/lookup_tablev3.bin")


def get_game_state(
    uid,
    is_banker,
    board_cards,
    hole_cards,
    player_bet_to,
    oppo_bet_to,
    player_street_bet_to,
    oppo_street_bet_to,
    player_init_chips,
    oppo_init_chips,
    player_allow_actions,
    is_first_action: bool,
    action_seq = None,
    small_blind: int = 1,
    big_blind: int = 2,
    last_action=None
):
    hole_cards = sort_cards(hole_cards)
    board_cards = sort_cards(board_cards)
    hole_cards = [card_id2str(card) for card in hole_cards]
    board_cards = [card_id2str(card) for card in board_cards]
    state = {
        "uid": uid,
        "banker": is_banker,
        "pocket": hole_cards,
        "community": board_cards,
        "player_bet_to": player_bet_to,
        "oppo_bet_to": oppo_bet_to,
        "players": 2
    }
    actions = valid_actions(
        player_allow_actions,
        player_bet_to,
        oppo_bet_to,
        player_init_chips,
        oppo_init_chips,
        big_blind
    )
    return state, actions


def valid_actions(
    player_allow_actions,
    player_bet_to,
    oppo_bet_to,
    player_init_chips,
    oppo_init_chips,
    big_blind,
):
    call_bet = oppo_bet_to - player_bet_to
    try:
        assert call_bet >= 0
    except:
        logging.fatal(
            "fatal: oppo_bet_to (%d) cann't be smaller than player_bet_to (%d)"
            % (oppo_bet_to, player_bet_to)
        )
    min_bet = call_bet + max(call_bet, big_blind)
    allin_bet = min(player_init_chips, oppo_init_chips) - player_bet_to
    actions = {}
    # actions allowed
    if (player_allow_actions & FOLD) and (not player_allow_actions & CHECK):
        actions["fold"] = True
    else:
        actions["fold"] = False
    if player_allow_actions & CHECK:
        actions["check"] = True
    else:
        actions["check"] = False
    if player_allow_actions & CALL:
        actions["call"] = call_bet
    else:
        actions["call"] = False
    if (player_allow_actions & RAISE) and (min_bet < allin_bet):
        actions["raise"] = [min_bet, allin_bet]
    else:
        actions["raise"] = False
    if player_allow_actions & ALLIN:
        actions["allin"] = allin_bet
    else:
        actions["allin"] = False
    return actions


def do_action(actions, state):
    """given the status of a game, returns the best action"""
    seed(time.time())
    uid = state["uid"]
    is_banker = state["banker"]
    hole_cards = state["pocket"]
    board_cards = state["community"]
    player_bet_to = state["player_bet_to"]
    oppo_bet_to = state["oppo_bet_to"]
    num_players = state["players"]
    pot_size = player_bet_to + oppo_bet_to
    allowed_actions = [k for k in actions if actions[k] is not False]
    logging.info(
        "req: uid={}, banker={}, pocket={}, community={}, pot={}, player={}, oppo={}, allowed actions={}"
        .format(
            uid,
            is_banker,
            hole_cards,
            board_cards,
            pot_size,
            player_bet_to,
            oppo_bet_to,
            allowed_actions
        )
    )

    if len(board_cards) < 3:
        # odds = min(EPS + pre_flop_strength(hole_cards) * 0.25, 1 - EPS)
        odds = PRE_FLOP_ODDS[pre_flop_strength(hole_cards)]
    else:
        odds = calculate_odds(hole_cards, board_cards, num_players)

    evs = []
    # check, call, raise, allin
    bet_actions = []
    bet_action_sizes = []
    bet_ev = odds * pot_size

    # ------------------------------
    # expected value for each action
    # ------------------------------
    def expected_value(odds, bet_size):
        return bet_ev - (1 - odds) * bet_size

    # fold
    # if actions["fold"]:
    #     ev = expected_value(odds, -actions["call"])
    #     if ev >= EV_THRESHOLD:
    #         bet_actions.append(FOLD)
    #         bet_action_sizes.append(0)
    #         evs.append(ev)
    # check
    if actions["check"]:
        bet_actions.append(CHECK)
        bet_action_sizes.append(0)
        evs.append(expected_value(odds, 0))
    # call
    if not isinstance(actions["call"], bool):
        ev = expected_value(odds, actions["call"])
        if ev >= EV_THRESHOLD:
            bet_actions.append(CALL)
            bet_action_sizes.append(actions["call"])
            evs.append(ev)
    # raise
    if not isinstance(actions["raise"], bool):
        # bet to allin
        for bet_size in range(*actions["raise"]):
            ev = expected_value(odds, bet_size)
            if ev >= EV_THRESHOLD:
                bet_actions.append(RAISE)
                bet_action_sizes.append(bet_size)
                evs.append(ev)
                # print(
                #     "odds: %f, pot: %d, bet_size: %d, ev: %f"
                #     % (odds, pot_size, bet_size, ev)
                # )
    # allin
    if not isinstance(actions["allin"], bool):
        # allin
        ev = expected_value(odds, actions["allin"])
        if ev >= EV_THRESHOLD:
            bet_actions.append(ALLIN)
            bet_action_sizes.append(actions["allin"])
            evs.append(ev)
    # ----------------
    # sample an action
    # ----------------

    def bet_or_fold(odds, bet_size):
        bet_prob = bet_ev / (bet_ev + (1 - odds) * bet_size)
        return (bet_prob, 1 - bet_prob)
    if evs:
        # 1. a bet action
        # normalization
        evs = [v + EPS for v in evs]
        sum_evs = sum(evs)
        probs = [v / sum_evs for v in evs]
        idx = sample_action(probs)
        action = bet_actions[idx]
        action_size = bet_action_sizes[idx]
        # 2. bet or fold
        # fold
        if actions["fold"]:
            bet_or_fold_prob = bet_or_fold(odds, action_size)
            idx = sample_action(bet_or_fold_prob)
            if idx == 1:
                action = FOLD
                action_size = 0
    else:
        action = FOLD
        action_size = 0
    logging.info(
        "rsp: uid={}, action={}, bet_size={}, odds={}"
        .format(
            uid, action, action_size, odds
        )
    )
    return action, action_size


def sample_action(probs):
    rand = random()
    cum_prob = 0
    for idx, prob in enumerate(probs):
        cum_prob += prob
        if rand < cum_prob:
            return idx


# def expected_value(odds, pot_size, bet_size):
#     return odds * pot_size - bet_size * (1 - odds)


def pre_flop_strength(hand):
    """return if the strength of the pocket cards is high
    """
    highs = {}
    highs[4] = [
        "AA", "AKs", "AQs", "AJs", "ATs", "AKo", "KK", "KQs", "KJs", "AQo",
        "QQ", "QJs", "JJ", "TT"
    ]
    highs[3] = [
        "A5s", "A4s", "A3s", "KTs", "KQo", "QTs", "AJo", "JTs", "T9s", "99",
        "98s", "88", "87s", "77", "66"
    ]
    highs[2] = [
        "A9s", "A8s", "A7s", "A6s", "A2s", "K9s", "K8s", "Q9s", "KJo", "QJo",
        "J9s", "ATo", "KTo", "QTo", "JTo", "T8s", "A9o", "J9o", "T9o", "97s",
        "98o", "86s", "76s", "75s", "65s", "55", "44", "33", "22"
    ]
    highs[1] = [
        "K7s", "K6s", "K5s", "K4s", "K3s", "Q8s", "Q7s", "Q6s", "Q5s", "Q4s",
        "J8s", "J7s", "J6s", "J5s", "T7s", "T6s", "K9o", "Q9o", "96s", "A8o",
        "K8o", "Q8o", "J8o", "T8o", "85s", "A7o", "K7o", "Q7o", "T7o", "97o",
        "87o", "74s", "A6o", "K6o", "86o", "76o", "64s", "63s", "A5o", "75o",
        "65o", "54s", "53s", "A4o", "43s", "A3o"
    ]
    card0, card1 = hand
    if card0[0] == card1[0]:
        pair = "".join([card0[0], card1[0]])
    elif card0[1] == card1[1]:
        pair = "".join([card0[0], card1[0], "s"])
    else:
        pair = "".join([card0[0], card1[0], "o"])
    for strenght in highs:
        if pair in highs[strenght]:
            return strenght
    return 0


# def calculate_odds(hole_cards, board_cards, num_players):
#     """given a hand and the community cards, returns the strenght of the hand
#     """
#     num_wins = 0
#     num_ties = 0
#     if len(board_cards) < 3:
#         return 0
#     t_start = time.time()
#     for _ in range(LOOPS):
#         # sort of deepcopy
#         board_cards_ = [x for x in board_cards]
#         deck = [r + s for r in RANKS for s in SUITS]
#         # remove the pocket cards of the deck
#         deck.remove(hole_cards[0])
#         deck.remove(hole_cards[1])
#         # remove the table cards of the deck
#         for card in board_cards_:
#             deck.remove(card)
#         # shuffle and fill the table with cards
#         shuffle(deck)
#         while(len(board_cards_) < 5):
#             board_cards_.append(deck.pop())
#         # generate hands
#         player_strength = generate_hands(hole_cards, board_cards_)
#         oppos_strength = []
#         for _ in range(num_players - 1):
#             oppo_hole_cards = []
#             oppo_hole_cards.append(deck.pop())
#             oppo_hole_cards.append(deck.pop())
#             oppos_strength.append(generate_hands(
#                 oppo_hole_cards, board_cards_))
#         # check whether the player wins
#         oppos_strength = sorted(oppos_strength, reverse=True)
#         if player_strength > oppos_strength[0]:
#             num_wins += 1
#         # if player_strength == oppos_strength[0]:
#         #     num_ties += 1
#     t_elapsed = time.time() - t_start
#     logging.info("message: mc take %.2fs" % t_elapsed)
#     return num_wins / LOOPS

def calculate_odds(hole_cards, board_cards, num_players):
    """given a hand and the community cards, returns the strenght of the hand
    """
    if len(board_cards) < 3:
        return 0
    t_start = time.time()
    suit_map = {"♦": "d", "♣": "c", "♥": "h", "♠": "s"}
    rank_map = {k: k for k in RANKS}
    rank_map["T"] = "10"
    hole_cards_ = [[rank_map[x[0]] + suit_map[x[1]] for x in hole_cards]]
    board_cards_ = [rank_map[x[0]] + suit_map[x[1]] for x in board_cards]
    num_wins, _ = MC_SIM.compute_probabilities(
        LOOPS, board_cards_, hole_cards_, num_players - 1
    )
    t_elapsed = time.time() - t_start
    logging.info("message: mc take %.2fs" % t_elapsed)
    return num_wins / 100


def generate_hands(hand, board):
    """return the winner combination of the avaiable cards
    """
    hand = [Card.str2card(card) for card in hand]
    board = [Card.str2card(card) for card in board]
    strength = HandEvaluator.eval_hand(hand, board)
    return strength
