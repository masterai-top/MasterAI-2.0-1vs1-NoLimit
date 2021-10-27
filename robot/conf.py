from typing import List
from game.texas_holdem.engine.card import Card

RANKS = Card.RANK_MAP
SUITS = {i: suit for i, suit in enumerate(Card.SUIT_MAP.values())}
SUIT_IDS = {suit_id: i for i, suit_id in enumerate(Card.SUIT_MAP.keys())}

# constants defined by pb
FOLD = 16
CHECK = 32
CALL = 64
RAISE = 128
ALLIN = 256


def sort_cards(card_ids: List[int]):
    return sorted(card_ids, key=lambda x: x & 0x0F, reverse=True)


def card_id2str(card_id: int):
    rank = card_id & 0x0F
    suit = (card_id & 0xF0) >> 4
    return RANKS[rank] + SUITS[suit]


def card_id2card(card_id: int):
    rank = card_id & 0x0F
    suit = (card_id & 0xF0) >> 4
    return Card(suit=suit, rank=rank)


def card2card_id(card: Card):
    """mapping a card object to its id for pb
    """
    return (SUIT_IDS[card.suit] << 4) + card.rank
