import random
import time
from functools import reduce
from typing import List

from .card import Card


class Deck(object):
    def __init__(self, drawn_cards: List[Card] = []):
        random.seed(time.time())
        self._drawn_cards = drawn_cards
        self._deck_cards = self._build_cards()

    def __len__(self) -> int:
        return len(self._deck_cards)

    def _build_cards(self) -> List[Card]:
        deck_cards = [Card.id2card(card_id) for card_id in range(1, 53)]
        for card in self._drawn_cards:
            deck_cards.remove(card)
        return deck_cards

    def restore(self) -> None:
        self._deck_cards = self._build_cards()

    def shuffle(self) -> None:
        random.shuffle(self._deck_cards)

    def draw_one_card(self) -> Card:
        return self._deck_cards.pop()

    def draw_cards(self, num_cards: int) -> List[Card]:
        return reduce(
            lambda cards, _: cards + [self.draw_one_card()],
            range(num_cards),
            []
        )

    @property
    def cards(self) -> List[Card]:
        return self._deck_cards


if __name__ == "__main__":
    from itertools import combinations
    board_cards = [Card.id2card(id_) for id_ in [1, 3, 5]]
    deck = Deck(drawn_cards=board_cards)
    hands = combinations(deck.cards, 2)
    print(len(deck))
    for card in deck.cards:
        print(card)
