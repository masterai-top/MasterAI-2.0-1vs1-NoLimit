from typing import Text

class Card(object):
    """a poker card

    card id: 1 - 52
    """
    CLUB = 2
    DIAMOND = 4
    HEART = 8
    SPADE = 16
    SUIT_MAP = {
        2   : "♦",
        4   : "♣",
        8   : "♥",
        16  : "♠"
    }
    RANK_MAP = {
        2   : "2",
        3   : "3",
        4   : "4",
        5   : "5",
        6   : "6",
        7   : "7",
        8   : "8",
        9   : "9",
        10  : "T",
        11  : "J",
        12  : "Q",
        13  : "K",
        14  : "A"
    }

    def __init__(self, suit: int, rank: int):
        self._suit = suit
        self._rank = 14 if rank == 1 else rank

    def __eq__(self, card) -> bool:
        return self._suit == card.suit and self._rank == card.rank

    def __str__(self):
        suit = self.SUIT_MAP[self._suit]
        rank = self.RANK_MAP[self._rank]
        return "{0}{1}".format(rank, suit)

    @property
    def id(self) -> int:
        rank = 1 if self._rank == 14 else self._rank
        num = 0
        tmp = self._suit >> 1
        while tmp & 1 != 1:
            num += 1
            tmp >>= 1
        return rank + 13 * num

    @property
    def suit(self) -> int:
        return self._suit

    @property
    def rank(self) -> int:
        return self._rank

    @classmethod
    def id2card(cls, card_id: int):
        suit, rank = 2, card_id
        while rank > 13:
            suit <<= 1
            rank -= 13
        return cls(suit, rank)

    @classmethod
    def str2card(cls, card_str: Text):
        assert len(card_str) == 2, "error: card_str is invalid"
        inverse = lambda hsh: {v: k for k, v in hsh.items()}
        suit = inverse(cls.SUIT_MAP)[card_str[1]]
        rank = inverse(cls.RANK_MAP)[card_str[0]]
        return cls(suit, rank)
